#!/usr/bin/env python3
"""
eval_transitivity_orc.py

At the ORC gap site (right after the embedded verb), measure the probability
mass of the moved NP under two conditions:

  transitive:   embedded verb takes an object  → gap is syntactically licensed
  intransitive: embedded verb cannot take an object → no gap

Key comparison:
  transitive_minus_intransitive > 0  →  model is sensitive to gap licensing
  ≈ 0                                →  positional/surface copy, verb type ignored

Input CSV (produced by generate_simple_datasets/orc_transitivity.py):
  sentence, pre_gap_text, moved_np, verb, verb_type, moved_np_animacy

Probability method: autoregressive geomean over NP tokens (same as all other
eval scripts), mean and median aggregated across sentences.

Usage
-----
python eval_transitivity_orc.py \\
    --checkpoint-dir checkpoints/my_model \\
    --input-csv data/orc_transitivity.csv \\
    --result-name results/eval_transitivity_orc.csv
"""
import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from litgpt import Tokenizer

from utils import (
    load_checkpoint,
    normalize_word,
    np_chain_logprob,
    resolve_checkpoint_file,
    step_from_checkpoint,
    _word_token_spans,
)

# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "step",
    "verb_type",          # transitive | intransitive | ALL
    "verb",               # specific verb or ALL
    "moved_np_mass",
    "moved_np_mass_median",
    "roi_count",
    "transitive_minus_intransitive",   # filled only in summary rows
    "checkpoint",
]

# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def compute_one_checkpoint(
    ckpt_file:      Path,
    tokenizer_dir:  Path,
    csv_rows:       List[dict],
    max_seq_length: int,
    batch_size:     int = 32,
) -> List[dict]:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(tokenizer_dir)
    model     = load_checkpoint(ckpt_file, device, max_seq_length)

    # Pre-tokenize moved_np for every row (" the noun" with leading space)
    def np_token_ids(np_str: str) -> List[int]:
        # np_str is e.g. "the boy" — add leading space for subword tokenisers
        return tokenizer.encode(" " + np_str, bos=False, eos=False).tolist()

    # Build list of (pre_gap_token_ids, moved_np_ids, verb, verb_type) per row
    items: List[Tuple[List[int], List[int], str, str]] = []
    skipped = 0
    for row in csv_rows:
        pre_gap_ids = tokenizer.encode(row["pre_gap_text"], bos=True, eos=False).tolist()
        moved_ids   = np_token_ids(row["moved_np"])
        if not pre_gap_ids or not moved_ids:
            skipped += 1
            continue
        items.append((pre_gap_ids, moved_ids, row["verb"], row["verb_type"]))

    if skipped:
        print(f"  [warn] {skipped} row(s) skipped (empty tokenisation)")

    # Accumulator: (verb, verb_type) → list of geomean values
    by_key: Dict[Tuple[str, str], List[float]] = {}

    def geomean_prob(x_ctx: torch.Tensor, ids: List[int], sp: torch.Tensor) -> float:
        lp = np_chain_logprob(model, x_ctx, ids, device, first_step_probs=sp)
        return lp ** (1.0 / len(ids))

    for batch_start in tqdm(range(0, len(items), batch_size), desc="  batches", leave=False):
        batch = items[batch_start: batch_start + batch_size]

        ctx_lists  = [it[0] for it in batch]
        ctx_lengths = [len(c) for c in ctx_lists]
        max_ctx    = max(ctx_lengths)
        x_batch    = torch.tensor(
            [c + [0] * (max_ctx - len(c)) for c in ctx_lists], device=device
        )
        with torch.no_grad():
            logits_batch = model(x_batch)

        for i, (pre_gap_ids, moved_ids, verb, verb_type) in enumerate(batch):
            sp    = torch.softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)
            x_ctx = torch.tensor(ctx_lists[i], device=device).unsqueeze(0)
            mass  = geomean_prob(x_ctx, moved_ids, sp)

            key = (verb, verb_type)
            by_key.setdefault(key, []).append(mass)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step = step_from_checkpoint(ckpt_file)

    # -----------------------------------------------------------------------
    # Build output rows
    # -----------------------------------------------------------------------
    rows: List[dict] = []

    # Per-verb rows
    # Also accumulate per verb_type for condition-level rows
    by_type: Dict[str, List[float]] = {}
    all_vals: List[float] = []

    for (verb, verb_type), vals in sorted(by_key.items()):
        n     = len(vals)
        avg   = statistics.mean(vals)
        med   = statistics.median(vals)
        by_type.setdefault(verb_type, []).extend(vals)
        all_vals.extend(vals)
        print(f"  step {step} | {verb_type:12s} | {verb:10s}: "
              f"moved_mass={avg:.6f} (med={med:.6f}), n={n}")
        rows.append({
            "step":                        step,
            "verb_type":                   verb_type,
            "verb":                        verb,
            "moved_np_mass":               round(avg, 6),
            "moved_np_mass_median":        round(med, 6),
            "roi_count":                   n,
            "transitive_minus_intransitive": "",
            "checkpoint":                  str(ckpt_file),
        })

    # Per-condition aggregate rows (verb = "ALL")
    cond_avgs: Dict[str, float] = {}
    for verb_type, vals in sorted(by_type.items()):
        n   = len(vals)
        avg = statistics.mean(vals)
        med = statistics.median(vals)
        cond_avgs[verb_type] = avg
        print(f"  step {step} | {verb_type:12s} | {'ALL':10s}: "
              f"moved_mass={avg:.6f} (med={med:.6f}), n={n}")
        rows.append({
            "step":                        step,
            "verb_type":                   verb_type,
            "verb":                        "ALL",
            "moved_np_mass":               round(avg, 6),
            "moved_np_mass_median":        round(med, 6),
            "roi_count":                   n,
            "transitive_minus_intransitive": "",
            "checkpoint":                  str(ckpt_file),
        })

    # Overall aggregate row with the key comparison
    if all_vals:
        n   = len(all_vals)
        avg = statistics.mean(all_vals)
        med = statistics.median(all_vals)
        diff = (
            round(cond_avgs.get("transitive", float("nan")) -
                  cond_avgs.get("intransitive", float("nan")), 6)
            if "transitive" in cond_avgs and "intransitive" in cond_avgs
            else ""
        )
        print(f"  step {step} | {'ALL':12s} | {'ALL':10s}: "
              f"moved_mass={avg:.6f} (med={med:.6f}), n={n}, "
              f"transitive-intransitive={diff}")
        rows.append({
            "step":                        step,
            "verb_type":                   "ALL",
            "verb":                        "ALL",
            "moved_np_mass":               round(avg, 6),
            "moved_np_mass_median":        round(med, 6),
            "roi_count":                   n,
            "transitive_minus_intransitive": diff,
            "checkpoint":                  str(ckpt_file),
        })

    return rows


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
            "ORC gap-site transitivity evaluation: "
            "compare moved NP mass under transitive vs intransitive embedded verbs."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint",     type=Path)
    g.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--tokenizer-dir",  type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--input-csv",      type=Path, required=True,
                   help="CSV produced by generate_simple_datasets/orc_transitivity.py")
    p.add_argument("--max-seq-length", type=int,  default=0)
    p.add_argument("--batch-size",     type=int,  default=32)
    p.add_argument("--result-name",    type=Path,
                   default=Path("results/eval_transitivity_orc.csv"))
    args = p.parse_args()

    csv_rows = load_csv(args.input_csv)
    print(f"Loaded {len(csv_rows)} rows from {args.input_csv}")
    n_trans = sum(1 for r in csv_rows if r["verb_type"] == "transitive")
    n_intr  = sum(1 for r in csv_rows if r["verb_type"] == "intransitive")
    print(f"  transitive: {n_trans}, intransitive: {n_intr}")

    if args.checkpoint is not None:
        ckpts = [resolve_checkpoint_file(args.checkpoint)]
    else:
        ckpts = sorted(
            args.checkpoint_dir.glob("step-*/lit_model.pth"),
            key=lambda x: int(x.parent.name.split("-")[1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No step checkpoints found under {args.checkpoint_dir}")

    all_rows = []
    for ckpt_file in tqdm(ckpts, desc="checkpoints"):
        rows = compute_one_checkpoint(
            ckpt_file, args.tokenizer_dir, csv_rows, args.max_seq_length, args.batch_size
        )
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["step"], r["verb_type"], r["verb"]))
    write_rows(all_rows, args.result_name)
    print(f"\nResults saved to: {args.result_name}")


if __name__ == "__main__":
    main()
