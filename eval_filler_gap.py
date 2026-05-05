import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from litgpt import Tokenizer

from eval_test import load_checkpoint, resolve_checkpoint_file, step_from_checkpoint


def compute_gap_surprisals(
    pre_gap: str, post_gap: str, model, tokenizer, device
) -> Tuple[float, float]:
    """
    Run model on pre_gap context and return surprisal of post_gap tokens.
    Returns (surprisal_first_token, mean_surprisal_all_tokens).
    """
    context_ids = tokenizer.encode(pre_gap, bos=False, eos=False).tolist()
    suffix_ids = tokenizer.encode(" " + post_gap, bos=False, eos=False).tolist()

    if not suffix_ids:
        return float("nan"), float("nan")

    all_ids = context_ids + suffix_ids
    x = torch.tensor(all_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    log_probs = torch.log_softmax(logits[0], dim=-1)

    context_len = len(context_ids)
    surprisals = [
        -log_probs[context_len - 1 + i, tok_id].item()
        for i, tok_id in enumerate(suffix_ids)
    ]
    return surprisals[0], sum(surprisals) / len(surprisals)


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
    diff_nogap  = s_f0g0 - s_f1g0
    interaction = diff_gap - diff_nogap

    print(f"\n  [{label}]")
    print(f"  surp(filler=0,gap=1) - surp(filler=1,gap=1) = {diff_gap:+.4f}  "
          f"({s_f0g1:.4f} - {s_f1g1:.4f})")
    print(f"  surp(filler=0,gap=0) - surp(filler=1,gap=0) = {diff_nogap:+.4f}  "
          f"({s_f0g0:.4f} - {s_f1g0:.4f})")
    print(f"  Interaction (gap_diff - nogap_diff)          = {interaction:+.4f}")


def run_checkpoint(ckpt_file: Path, tokenizer: Tokenizer, rows: List[dict],
                   max_seq_length: int, device: torch.device) -> List[dict]:
    step = step_from_checkpoint(ckpt_file)
    print(f"\n=== step {step} ===")
    model = load_checkpoint(ckpt_file, device, max_seq_length)

    results = []
    for row in tqdm(rows, desc="  sentences", leave=False):
        pre_gap = row["pre_gap_text"]
        post_gap = row["post_gap_text"]
        if not post_gap:
            # Derive suffix from sentence (e.g. "?" for wh +gap)
            post_gap = row["sentence"][len(pre_gap):].strip()

        s_first, s_mean = compute_gap_surprisals(pre_gap, post_gap, model, tokenizer, device)
        results.append({**row, "step": step, "surprisal_first": round(s_first, 6),
                        "surprisal_mean": round(s_mean, 6)})

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
        all_results.extend(run_checkpoint(ckpt_file, tokenizer, rows, args.max_seq_length, device))

    out_fields = input_fields + ["step", "surprisal_first", "surprisal_mean"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
