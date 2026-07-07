#!/usr/bin/env python3
"""Evaluate language model checkpoints on the BLiMP benchmark.

BLiMP consists of 67 .jsonl files (one per paradigm), each containing 1000
minimal pairs with "sentence_good" and "sentence_bad". A model scores a pair
correctly when it assigns a higher log-probability to sentence_good.

Usage
-----
# Single checkpoint, all paradigms in a directory:
python eval_blimp.py \\
    --checkpoint-dir checkpoints/my_model \\
    --blimp-dir /data/blimp/data \\
    --output-csv results/blimp_results.csv

# Single checkpoint file, specific paradigm:
python eval_blimp.py \\
    --checkpoint checkpoints/my_model/step-10000 \\
    --blimp-dir /data/blimp/data \\
    --output-csv results/blimp_results.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Sentence log-probability
# ---------------------------------------------------------------------------

def sentence_log_prob(
    sentence_ids: List[int],
    log_probs: torch.Tensor,
) -> float:
    """Sum token log-probs for a sequence (given pre-computed log_probs tensor).

    log_probs: (seq_len, vocab) — already shifted so that log_probs[t] gives
    the distribution used to predict token at position t+1 (i.e. the output of
    the model at position t). We want sum_{t=1}^{T-1} log P(x_t | x_{<t}),
    which corresponds to positions 0..T-2 in log_probs.
    """
    total = 0.0
    for t, tok_id in enumerate(sentence_ids[1:]):  # skip first token (no context)
        total += log_probs[t, tok_id].item()
    return total


def compute_pair_scores_batch(
    pairs: List[Tuple[List[int], List[int]]],
    model,
    device: torch.device,
    batch_size: int = 32,
) -> List[Tuple[float, float]]:
    """Return (log_prob_good, log_prob_bad) for each pair.

    Each element of `pairs` is (good_ids, bad_ids).
    Both sentences in each pair are processed in the same forward pass by
    concatenating them into a single batch.
    """
    # Flatten pairs into individual sequences
    seqs: List[List[int]] = []
    for good_ids, bad_ids in pairs:
        seqs.append(good_ids)
        seqs.append(bad_ids)

    log_probs_list: List[float] = []

    for start in range(0, len(seqs), batch_size):
        batch_seqs = seqs[start : start + batch_size]

        max_len = max(len(s) for s in batch_seqs)
        pad_id = 0

        padded = [
            s + [pad_id] * (max_len - len(s))
            for s in batch_seqs
        ]
        x = torch.tensor(padded, device=device)  # (B, max_len)

        with torch.no_grad():
            logits = model(input_ids=x).logits  # (B, max_len, vocab)
        lp = torch.log_softmax(logits, dim=-1)  # (B, max_len, vocab)

        for i, s in enumerate(batch_seqs):
            lp_cpu = lp[i].cpu()
            log_probs_list.append(sentence_log_prob(s, lp_cpu))

    # Re-pair: odd indices are bad, even are good
    results = []
    for k in range(0, len(log_probs_list), 2):
        results.append((log_probs_list[k], log_probs_list[k + 1]))
    return results


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    ckpt_dir: Path,
    tokenizer,
    paradigms: List[dict],
    device: torch.device,
    batch_size: int = 32,
    model=None,
) -> Tuple[List[dict], List[dict]]:
    """Return (pair_rows, paradigm_rows) for one checkpoint."""
    name = ckpt_dir.name
    step = int(name.split("-")[-1]) if "-" in name else 0
    print(f"\n=== {name} (step {step}) ===")
    loaded_here = False
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, local_files_only=True
        ).to(device)
        model.eval()
        loaded_here = True

    pair_rows: List[dict] = []
    paradigm_rows: List[dict] = []

    for par in paradigms:
        uid = par["uid"]
        field = par["field"]
        linguistics_term = par["linguistics_term"]
        raw_pairs = par["pairs"]  # list of (good_ids, bad_ids, pair_id, sentence_good, sentence_bad)

        token_pairs = [(p[0], p[1]) for p in raw_pairs]

        scores = compute_pair_scores_batch(token_pairs, model, device, batch_size)

        correct = 0
        for (good_ids, bad_ids, pair_id, sent_good, sent_bad), (lp_good, lp_bad) in zip(raw_pairs, scores):
            is_correct = int(lp_good > lp_bad)
            correct += is_correct
            pair_rows.append({
                "step": step,
                "uid": uid,
                "field": field,
                "linguistics_term": linguistics_term,
                "pair_id": pair_id,
                "sentence_good": sent_good,
                "sentence_bad": sent_bad,
                "log_prob_good": round(lp_good, 6),
                "log_prob_bad": round(lp_bad, 6),
                "correct": is_correct,
            })

        n = len(raw_pairs)
        accuracy = correct / n if n > 0 else float("nan")
        paradigm_rows.append({
            "step": step,
            "uid": uid,
            "field": field,
            "linguistics_term": linguistics_term,
            "n_pairs": n,
            "n_correct": correct,
            "accuracy": round(accuracy, 6),
        })
        print(f"  {uid:50s}  acc={accuracy:.3f}  ({correct}/{n})")

    # Overall accuracy across all paradigms
    total_correct = sum(r["n_correct"] for r in paradigm_rows)
    total_n = sum(r["n_pairs"] for r in paradigm_rows)
    overall_acc = total_correct / total_n if total_n > 0 else float("nan")
    print(f"\n  Overall accuracy: {overall_acc:.4f}  ({total_correct}/{total_n})")

    if loaded_here:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pair_rows, paradigm_rows


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paradigms(blimp_dir: Path, tokenizer) -> List[dict]:
    """Load all .jsonl files in blimp_dir and tokenize sentences."""
    jsonl_files = sorted(blimp_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {blimp_dir}")

    paradigms = []
    print(f"Loading {len(jsonl_files)} BLiMP paradigm(s) from {blimp_dir} ...")
    for jf in jsonl_files:
        pairs = []
        uid = field = linguistics_term = None
        with open(jf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if uid is None:
                    uid = item["UID"]
                    field = item["field"]
                    linguistics_term = item["linguistics_term"]
                sent_good = item["sentence_good"]
                sent_bad = item["sentence_bad"]
                good_ids = tokenizer.encode(sent_good)
                bad_ids = tokenizer.encode(sent_bad)
                pairs.append((good_ids, bad_ids, item["pairID"], sent_good, sent_bad))

        paradigms.append({
            "uid": uid,
            "field": field,
            "linguistics_term": linguistics_term,
            "pairs": pairs,
        })

    return paradigms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate GPT checkpoints on BLiMP minimal pairs."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path,
                   help="Path to a single checkpoint file or step directory.")
    g.add_argument("--checkpoint-dir", type=Path,
                   help="Directory containing step-*/lit_model.pth checkpoints.")

    p.add_argument("--blimp-dir", type=Path, required=True,
                   help="Directory containing BLiMP .jsonl paradigm files.")
    p.add_argument("--tokenizer-dir", type=Path, default=None,
                   help="HF tokenizer directory. Defaults to --checkpoint or --checkpoint-dir.")
    p.add_argument("--output-csv", type=Path, required=True,
                   help="Output CSV for per-pair results.")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = args.tokenizer_dir or args.checkpoint or args.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)

    # Resolve checkpoints
    if args.checkpoint is not None:
        ckpts = [args.checkpoint]
    else:
        ckpts = sorted(
            [d for d in args.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint-* directories found in {args.checkpoint_dir}")

    # Load & tokenize BLiMP data once
    paradigms = load_paradigms(args.blimp_dir, tokenizer)

    all_pair_rows: List[dict] = []
    all_paradigm_rows: List[dict] = []

    for ckpt_file in ckpts:
        pair_rows, paradigm_rows = evaluate_checkpoint(
            ckpt_file, tokenizer, paradigms,
            device, args.batch_size,
        )
        all_pair_rows.extend(pair_rows)
        all_paradigm_rows.extend(paradigm_rows)

    # Save per-pair results
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pair_fields = [
        "step", "uid", "field", "linguistics_term",
        "pair_id", "sentence_good", "sentence_bad",
        "log_prob_good", "log_prob_bad", "correct",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(all_pair_rows)
    print(f"\nPer-pair results saved to: {args.output_csv}")

    # Save per-paradigm accuracy summary
    paradigm_csv = args.output_csv.with_stem(args.output_csv.stem + "_paradigms")
    paradigm_fields = [
        "step", "uid", "field", "linguistics_term",
        "n_pairs", "n_correct", "accuracy",
    ]
    with open(paradigm_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=paradigm_fields)
        writer.writeheader()
        writer.writerows(all_paradigm_rows)
    print(f"Per-paradigm accuracy saved to: {paradigm_csv}")


if __name__ == "__main__":
    main()
