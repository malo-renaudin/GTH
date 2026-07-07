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

from transformers import AutoTokenizer, AutoModelForCausalLM


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
    ckpt_dir:   Path,
    tokenizer,
    csv_rows:   List[dict],
    batch_size: int = 32,
) -> List[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AutoModelForCausalLM.from_pretrained(ckpt_dir, local_files_only=True).to(device)
    model.eval()

    items: List[Tuple[List[int], int, str, str]] = []
    skipped = 0
    for row in csv_rows:
        pre_gap_ids = tokenizer.encode(row["pre_gap_text"], add_special_tokens=False)
        # moved_np is "the N1"; N1 is single-token so we take only the last token
        noun_id = tokenizer.encode(" " + row["moved_np"].split()[-1], add_special_tokens=False)
        if not pre_gap_ids or len(noun_id) != 1:
            skipped += 1
            continue
        items.append((pre_gap_ids, noun_id[0], row["verb"], row["verb_type"]))

    if skipped:
        print(f"  [warn] {skipped} row(s) skipped (empty tokenisation)")

    # Accumulator: (verb, verb_type) → list of P(noun | ctx) values
    by_key: Dict[Tuple[str, str], List[float]] = {}

    for batch_start in tqdm(range(0, len(items), batch_size), desc="  batches", leave=False):
        batch = items[batch_start: batch_start + batch_size]

        ctx_lists   = [it[0] for it in batch]
        ctx_lengths = [len(c) for c in ctx_lists]
        max_ctx     = max(ctx_lengths)
        x_batch     = torch.tensor(
            [c + [0] * (max_ctx - len(c)) for c in ctx_lists], device=device
        )
        with torch.no_grad():
            logits_batch = model(input_ids=x_batch).logits

        for i, (pre_gap_ids, noun_id, verb, verb_type) in enumerate(batch):
            sp   = torch.softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)
            mass = sp[noun_id].item()  # single-token: direct lookup

            key = (verb, verb_type)
            by_key.setdefault(key, []).append(mass)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    name = ckpt_dir.name
    step = int(name.split("-")[-1]) if "-" in name else 0

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
            "checkpoint":                  str(ckpt_dir),
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
            "checkpoint":                  str(ckpt_dir),
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
            "checkpoint":                  str(ckpt_dir),
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


def run(ckpt_dir: str, csv_path: str, out_csv_path: str) -> None:
    """Callable entry-point used by the training callback."""
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    rows = compute_one_checkpoint(Path(ckpt_dir), tokenizer, load_csv(Path(csv_path)))
    write_rows(rows, Path(out_csv_path))


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
    p.add_argument("--tokenizer-dir",  type=Path, default=None,
                   help="HF tokenizer directory. Defaults to --checkpoint or --checkpoint-dir.")
    p.add_argument("--input-csv",      type=Path, required=True,
                   help="CSV produced by generate_simple_datasets/orc_transitivity.py")
    p.add_argument("--batch-size",     type=int,  default=32)
    p.add_argument("--result-name",    type=Path,
                   default=Path("results/eval_transitivity_orc.csv"))
    args = p.parse_args()

    csv_rows = load_csv(args.input_csv)
    print(f"Loaded {len(csv_rows)} rows from {args.input_csv}")
    n_trans = sum(1 for r in csv_rows if r["verb_type"] == "transitive")
    n_intr  = sum(1 for r in csv_rows if r["verb_type"] == "intransitive")
    print(f"  transitive: {n_trans}, intransitive: {n_intr}")

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

    all_rows.sort(key=lambda r: (r["step"], r["verb_type"], r["verb"]))
    write_rows(all_rows, args.result_name)
    print(f"\nResults saved to: {args.result_name}")


if __name__ == "__main__":
    main()
