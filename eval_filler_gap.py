import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from litgpt import Tokenizer

from eval_test import load_checkpoint, resolve_checkpoint_file, step_from_checkpoint


def compute_gap_surprisals_batch(
    items: List[Tuple[str, str]],
    model, tokenizer, device, batch_size: int = 32,
) -> List[Tuple[float, float]]:
    """
    Batched version: given a list of (pre_gap, post_gap) pairs, return
    [(surprisal_first, surprisal_mean), ...] for each pair.
    """
    results = []

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]

        encoded = []
        for pre_gap, post_gap in batch:
            context_ids = tokenizer.encode(pre_gap, bos=False, eos=False).tolist()
            suffix_ids  = tokenizer.encode(" " + post_gap, bos=False, eos=False).tolist()
            encoded.append((context_ids, suffix_ids))

        max_len = max(len(c) + len(s) for c, s in encoded)
        pad_id  = 0

        padded = [
            c + s + [pad_id] * (max_len - len(c) - len(s))
            for c, s in encoded
        ]
        x = torch.tensor(padded, device=device)  # (B, max_len)

        with torch.no_grad():
            logits = model(x)  # (B, max_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, max_len, vocab)

        for i, (context_ids, suffix_ids) in enumerate(encoded):
            if not suffix_ids:
                results.append((float("nan"), float("nan")))
                continue
            context_len = len(context_ids)
            surprisals = [
                -log_probs[i, context_len - 1 + j, tok_id].item()
                for j, tok_id in enumerate(suffix_ids)
            ]
            results.append((surprisals[0], sum(surprisals) / len(surprisals)))

    return results


def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else float("nan")


def print_metrics(results: List[dict], surprisal_col: str, label: str) -> None:
    by_condition = defaultdict(list)
    for r in results:
        key = (int(r["filler"]), int(r["gap"]))
        v = r[surprisal_col]
        if v == v:  # skip NaN
            by_condition[key].append(v)

    s = {k: _mean(v) for k, v in by_condition.items()}
    s_f0g1 = s.get((0, 1), float("nan"))
    s_f1g1 = s.get((1, 1), float("nan"))
    s_f0g0 = s.get((0, 0), float("nan"))
    s_f1g0 = s.get((1, 0), float("nan"))

    diff_gap    = s_f0g1 - s_f1g1
    diff_nogap  = s_f1g0 - s_f0g0 
    interaction = diff_gap + diff_nogap

    print(f"\n  [{label}]")
    print(f"  surp(filler=0,gap=1) - surp(filler=1,gap=1) = {diff_gap:+.4f}  "
          f"({s_f0g1:.4f} - {s_f1g1:.4f})")
    print(f"  surp(filler=1,gap=0) - surp(filler=0,gap=0) = {diff_nogap:+.4f}  "
          f"({s_f1g0:.4f} - {s_f0g0:.4f})")
    print(f"  Interaction (gap_diff + nogap_diff)          = {interaction:+.4f}")


def run_checkpoint(ckpt_file: Path, tokenizer: Tokenizer, rows: List[dict],
                   max_seq_length: int, device: torch.device,
                   batch_size: int = 32) -> List[dict]:
    step = step_from_checkpoint(ckpt_file)
    print(f"\n=== step {step} ===")
    model = load_checkpoint(ckpt_file, device, max_seq_length)

    # Resolve post_gap for rows where it's empty (wh +gap: derive from sentence)
    items = []
    for row in rows:
        pre_gap  = row["pre_gap_text"]
        post_gap = row["post_gap_text"]
        if not post_gap:
            post_gap = row["sentence"][len(pre_gap):].strip()
        items.append((pre_gap, post_gap))

    surprisals = []
    for start in tqdm(range(0, len(items), batch_size), desc="  batches", leave=False):
        batch_items = items[start : start + batch_size]
        surprisals.extend(compute_gap_surprisals_batch(batch_items, model, tokenizer, device, batch_size))

    results = [
        {**row, "step": step,
         "surprisal_first": round(s_first, 6),
         "surprisal_mean":  round(s_mean,  6)}
        for row, (s_first, s_mean) in zip(rows, surprisals)
    ]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print_metrics(results, "surprisal_first", "surprisal_first")
    print_metrics(results, "surprisal_mean",  "surprisal_mean")
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute gap-site surprisal for filler-gap / wh-movement factorial CSVs."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--input-csv", type=Path, required=True,
                   help="filler_gap_factorial.csv or wh_movement_factorial.csv")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--max-seq-length", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    input_fields = list(rows[0].keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(args.tokenizer_dir)

    if args.checkpoint is not None:
        ckpts = [resolve_checkpoint_file(args.checkpoint)]
    else:
        ckpts = sorted(
            args.checkpoint_dir.glob("step-*/lit_model.pth"),
            key=lambda x: int(x.parent.name.split("-")[1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No step checkpoints found in {args.checkpoint_dir}")

    all_results = []
    for ckpt_file in ckpts:
        all_results.extend(run_checkpoint(ckpt_file, tokenizer, rows, args.max_seq_length, device, args.batch_size))

    out_fields = input_fields + ["step", "surprisal_first", "surprisal_mean"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
