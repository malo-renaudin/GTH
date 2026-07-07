#!/usr/bin/env python3
"""
eval_semantic_distractors_orc.py

At the ORC gap site (right after the embedded verb), compare probability mass of:
  1. the moved NP  (the filler, e.g. "the boy")
  2. semantically plausible same-verb distractors of the opposite animacy:
       - animate moved NP  → verb-specific inanimate distractors
         e.g. after "eats": "the pizza", "the cake", "the bread"
       - inanimate moved NP → animate NPs from the standard vocab

The key question: does the model correctly predict the filler at the gap
even when the embedded verb strongly selects for semantically incompatible objects?

All probabilities use the autoregressive geomean over NP tokens, median
aggregated across sentences, matching the logic of eval_gap_np_orc.py.

Usage
-----
python eval_semantic_distractors_orc.py \\
    --checkpoint-dir checkpoints/my_model \\
    --input-csv data/orc_semantic_distractors.csv \\
    --result-name results/semantic_distractors.csv
"""
import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


CSV_FIELDS = [
    "step", "structure", "gap_verb",
    "moved_np", "moved_np_animacy",
    "moved_np_mass", "moved_np_mass_median",
    "distractor_mass", "distractor_mass_median",
    "moved_minus_distractor",
    "roi_count", "checkpoint",
]

# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def compute_one_checkpoint(
    ckpt_dir:   Path,
    tokenizer,
    csv_rows:   List[dict],
    batch_size: int = 32,
) -> List[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AutoModelForCausalLM.from_pretrained(ckpt_dir, local_files_only=True).to(device)
    model.eval()

    # Pre-tokenize: single-token noun IDs for moved NP and distractors.
    # P(noun | pre_gap_text) compared directly at the gap site.
    items: List[Tuple[List[int], int, List[int], str, str]] = []
    skipped = 0
    for row in csv_rows:
        pre_gap_ids = tokenizer.encode(row["pre_gap_text"], add_special_tokens=False)

        moved_noun_id = tokenizer.encode(" " + row["moved_np"].split()[-1], add_special_tokens=False)
        distractor_noun_ids = [
            tokenizer.encode(" " + obj.strip().split()[-1], add_special_tokens=False)
            for obj in row["plausible_objects"].split("|")
            if obj.strip()
        ]
        # skip if any noun is multi-token
        if (not pre_gap_ids or len(moved_noun_id) != 1
                or any(len(d) != 1 for d in distractor_noun_ids)):
            skipped += 1
            continue

        verb = row["pre_gap_text"].strip().split()[-1]
        items.append((
            pre_gap_ids,
            moved_noun_id[0],
            [d[0] for d in distractor_noun_ids],
            verb,
            row["moved_np_animacy"],
        ))

    if skipped:
        print(f"  [warn] {skipped} row(s) skipped (empty tokenisation)")

    # Per-verb accumulator: verb → {moved, distractor}
    by_verb: Dict[str, Dict[str, List[float]]] = {}

    for batch_start in tqdm(range(0, len(items), batch_size), desc="  batches", leave=False):
        batch = items[batch_start: batch_start + batch_size]

        contexts    = [it[0] for it in batch]
        ctx_lengths = [len(c) for c in contexts]
        max_ctx     = max(ctx_lengths)
        x_batch     = torch.tensor(
            [c + [0] * (max_ctx - len(c)) for c in contexts], device=device
        )
        with torch.no_grad():
            logits_batch = model(input_ids=x_batch).logits

        for i, (ctx_with_the, moved_noun_id, distractor_noun_ids, verb, animacy) in enumerate(batch):
            sp = torch.softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)

            # direct single-token lookups — P(noun | ctx + "the")
            moved_mass       = sp[moved_noun_id].item()
            distractor_probs = [sp[d].item() for d in distractor_noun_ids]
            if not distractor_probs:
                continue

            by_verb.setdefault(verb, {"moved": [], "distractor": [], "animacy": animacy})
            by_verb[verb]["moved"].append(moved_mass)
            by_verb[verb]["distractor"].append(statistics.mean(distractor_probs))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    name = ckpt_dir.name
    step = int(name.split("-")[-1]) if "-" in name else 0

    # Build one row per verb (+ one "all" aggregate row)
    rows = []
    all_moved, all_distractor = [], []

    for verb, vals in sorted(by_verb.items()):
        n = len(vals["moved"])
        if n == 0:
            continue
        t_avg = statistics.mean(vals["moved"])
        t_med = statistics.median(vals["moved"])
        d_avg = statistics.mean(vals["distractor"])
        d_med = statistics.median(vals["distractor"])

        all_moved.extend(vals["moved"])
        all_distractor.extend(vals["distractor"])

        animacy_label = vals.get("animacy", "animate")
        print(
            f"  step {step} | {verb:10s}: "
            f"moved={t_avg:.6f} (med={t_med:.6f}), "
            f"distractor={d_avg:.6f} (med={d_med:.6f}), n={n}"
        )
        rows.append({
            "step": step,
            "structure": "orc",
            "gap_verb": verb,
            "moved_np": "filler",
            "moved_np_animacy": animacy_label,
            "moved_np_mass": round(t_avg, 6),
            "moved_np_mass_median": round(t_med, 6),
            "distractor_mass": round(d_avg, 6),
            "distractor_mass_median": round(d_med, 6),
            "moved_minus_distractor": round(t_avg - d_avg, 6),
            "roi_count": n,
            "checkpoint": str(ckpt_dir),
        })

    # Aggregate "all verbs" row
    if all_moved:
        t_avg = statistics.mean(all_moved)
        t_med = statistics.median(all_moved)
        d_avg = statistics.mean(all_distractor)
        d_med = statistics.median(all_distractor)
        print(
            f"  step {step} | {'ALL':10s}: "
            f"moved={t_avg:.6f} (med={t_med:.6f}), "
            f"distractor={d_avg:.6f} (med={d_med:.6f}), n={len(all_moved)}"
        )
        rows.append({
            "step": step,
            "structure": "orc",
            "gap_verb": "ALL",
            "moved_np": "filler",
            "moved_np_animacy": "mixed",
            "moved_np_mass": round(t_avg, 6),
            "moved_np_mass_median": round(t_med, 6),
            "distractor_mass": round(d_avg, 6),
            "distractor_mass_median": round(d_med, 6),
            "moved_minus_distractor": round(t_avg - d_avg, 6),
            "roi_count": len(all_moved),
            "checkpoint": str(ckpt_dir),
        })

    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_rows(rows: List[dict], result_name: Path) -> None:
    result_name.parent.mkdir(parents=True, exist_ok=True)
    with open(result_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "ORC gap-site semantic distractor evaluation: "
            "moved NP mass vs verb-compatible opposite-animacy distractors."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--tokenizer-dir", type=Path, default=None,
                   help="HF tokenizer directory. Defaults to --checkpoint or --checkpoint-dir.")
    p.add_argument("--input-csv", type=Path, required=True,
                   help="CSV produced by generate_simple_datasets/orc_semantic_distractors.py")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--result-name", type=Path,
                   default=Path("results/eval_semantic_distractors_orc.csv"))
    args = p.parse_args()

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    print(f"Loaded {len(csv_rows)} rows from {args.input_csv}")
    n_anim = sum(1 for r in csv_rows if r["moved_np_animacy"] == "animate")
    n_inan = sum(1 for r in csv_rows if r["moved_np_animacy"] == "inanimate")
    print(f"  animate filler: {n_anim}, inanimate filler: {n_inan}")

    tok_path  = args.tokenizer_dir or args.checkpoint or args.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)

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

    all_rows = []
    for ckpt_dir in tqdm(ckpts, desc="checkpoints"):
        rows = compute_one_checkpoint(ckpt_dir, tokenizer, csv_rows, args.batch_size)
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["step"], r["gap_verb"]))
    write_rows(all_rows, args.result_name)
    print(f"\nResults saved to: {args.result_name}")


if __name__ == "__main__":
    main()
