#!/usr/bin/env python3
"""
eval_semantic_distractors_orc.py

At the ORC gap site (right after the embedded verb), compare probability mass of:
  1. the moved NP  (the filler, e.g. "the boy")
  2. semantically plausible same-verb distractors of the opposite animacy:
       - animate moved NP  → verb-specific inanimate distractors
         e.g. after "eats": "the pizza", "the cake", "the bread"
       - inanimate moved NP → animate NPs from the standard vocab
  3. other animate vocab NPs (non-filler, always computed for reference)

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

from litgpt import Tokenizer

from utils import (
    load_checkpoint,
    normalize_word,
    np_chain_logprob,
    resolve_checkpoint_file,
    step_from_checkpoint,
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Animate vocabulary — used both as comparators and for moved-NP identification
ANIMATE_NOUNS = [
    "student", "doctor", "pilot", "officer", "athlete", "artist",
    "child", "girl", "boy", "patient", "client", "tourist",
    "scientist", "engineer",
]
ANIMATE_SET = set(ANIMATE_NOUNS)


CSV_FIELDS = [
    "step", "structure", "gap_verb",
    "moved_np", "moved_np_animacy",
    "moved_np_mass", "moved_np_mass_median",
    "distractor_mass", "distractor_mass_median",
    "animate_other_mass", "animate_other_mass_median",
    "moved_minus_distractor",
    "moved_minus_animate_other",
    "roi_count", "checkpoint",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_np_token_lists(
    nouns: List[str], tokenizer: Tokenizer
) -> List[Tuple[str, List[int]]]:
    """[(noun, token_ids_for_'the noun'), ...]"""
    return [
        (noun, tokenizer.encode(f" the {noun}", bos=False, eos=False).tolist())
        for noun in nouns
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

    # Pre-build animate NP token lists for animate_other computation
    animate_np_list = _build_np_token_lists(ANIMATE_NOUNS, tokenizer)

    # Pre-tokenize every row from the CSV
    # Each item: (pre_gap_ids, moved_ids, distractor_ids_list, verb, moved_np_animacy)
    items: List[Tuple[List[int], List[int], List[List[int]], str, str]] = []
    skipped = 0
    for row in csv_rows:
        pre_gap_ids = tokenizer.encode(row["pre_gap_text"], bos=True, eos=False).tolist()
        moved_ids   = tokenizer.encode(" " + row["moved_np"], bos=False, eos=False).tolist()
        # plausible_objects is "the pizza|the cake|..." — tokenise each
        distractor_ids_list = [
            tokenizer.encode(" " + obj.strip(), bos=False, eos=False).tolist()
            for obj in row["plausible_objects"].split("|")
            if obj.strip()
        ]
        if not pre_gap_ids or not moved_ids or not distractor_ids_list:
            skipped += 1
            continue
        verb = row["pre_gap_text"].strip().split()[-1]  # last word of pre_gap_text
        items.append((pre_gap_ids, moved_ids, distractor_ids_list, verb, row["moved_np_animacy"]))

    if skipped:
        print(f"  [warn] {skipped} row(s) skipped (empty tokenisation)")

    # Per-verb accumulator: verb → {moved, distractor, animate_other}
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
            logits_batch = model(x_batch)

        for i, (pre_gap_ids, moved_ids, distractor_ids_list, verb, animacy) in enumerate(batch):
            sp    = torch.softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)
            x_ctx = torch.tensor(contexts[i], device=device).unsqueeze(0)

            def geomean_prob(ids: List[int]) -> float:
                if not ids:
                    return float("nan")
                lp = np_chain_logprob(model, x_ctx, ids, device, first_step_probs=sp)
                return lp ** (1.0 / len(ids))

            # --- Moved NP ---
            moved_mass = geomean_prob(moved_ids)

            # --- Semantic distractors (from CSV plausible_objects) ---
            distractor_probs = [geomean_prob(ids) for ids in distractor_ids_list if ids]
            if not distractor_probs:
                continue

            # --- Other animate NPs (non-filler, for reference) ---
            moved_np_str   = tokenizer.decode(torch.tensor(moved_ids)).strip().lower()
            moved_head     = moved_np_str.split()[-1]  # e.g. "the boy" → "boy"
            animate_other_probs = [
                geomean_prob(ids)
                for noun, ids in animate_np_list
                if normalize_word(noun) != moved_head and ids
            ]

            by_verb.setdefault(verb, {"moved": [], "distractor": [], "animate_other": [],
                                       "animacy": animacy})
            by_verb[verb]["moved"].append(moved_mass)
            by_verb[verb]["distractor"].append(statistics.mean(distractor_probs))
            if animate_other_probs:
                by_verb[verb]["animate_other"].append(statistics.mean(animate_other_probs))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step = step_from_checkpoint(ckpt_file)

    # Build one row per verb (+ one "all" aggregate row)
    rows = []
    all_moved, all_distractor, all_animate_other = [], [], []

    for verb, vals in sorted(by_verb.items()):
        n = len(vals["moved"])
        if n == 0:
            continue
        t_avg = statistics.mean(vals["moved"])
        t_med = statistics.median(vals["moved"])
        d_avg = statistics.mean(vals["distractor"])
        d_med = statistics.median(vals["distractor"])
        a_avg = statistics.mean(vals["animate_other"]) if vals["animate_other"] else float("nan")
        a_med = statistics.median(vals["animate_other"]) if vals["animate_other"] else float("nan")

        all_moved.extend(vals["moved"])
        all_distractor.extend(vals["distractor"])
        all_animate_other.extend(vals["animate_other"])

        animacy_label = vals.get("animacy", "animate")
        print(
            f"  step {step} | {verb:10s}: "
            f"moved={t_avg:.6f} (med={t_med:.6f}), "
            f"distractor={d_avg:.6f} (med={d_med:.6f}), "
            f"animate_other={a_avg:.6f} (med={a_med:.6f}), n={n}"
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
            "animate_other_mass": round(a_avg, 6) if a_avg == a_avg else float("nan"),
            "animate_other_mass_median": round(a_med, 6) if a_med == a_med else float("nan"),
            "moved_minus_distractor": round(t_avg - d_avg, 6),
            "moved_minus_animate_other": round(t_avg - a_avg, 6) if a_avg == a_avg else float("nan"),
            "roi_count": n,
            "checkpoint": str(ckpt_file),
        })

    # Aggregate "all verbs" row
    if all_moved:
        t_avg = statistics.mean(all_moved)
        t_med = statistics.median(all_moved)
        d_avg = statistics.mean(all_distractor)
        d_med = statistics.median(all_distractor)
        a_avg = statistics.mean(all_animate_other) if all_animate_other else float("nan")
        a_med = statistics.median(all_animate_other) if all_animate_other else float("nan")
        print(
            f"  step {step} | {'ALL':10s}: "
            f"moved={t_avg:.6f} (med={t_med:.6f}), "
            f"distractor={d_avg:.6f} (med={d_med:.6f}), "
            f"animate_other={a_avg:.6f} (med={a_med:.6f}), n={len(all_moved)}"
        )
        rows.append({
            "step": step,
            "structure": "orc",
            "gap_verb": "ALL",
            "moved_np": "filler",
            "moved_np_animacy": "animate",
            "moved_np_mass": round(t_avg, 6),
            "moved_np_mass_median": round(t_med, 6),
            "distractor_mass": round(d_avg, 6),
            "distractor_mass_median": round(d_med, 6),
            "animate_other_mass": round(a_avg, 6) if a_avg == a_avg else float("nan"),
            "animate_other_mass_median": round(a_med, 6) if a_med == a_med else float("nan"),
            "moved_minus_distractor": round(t_avg - d_avg, 6),
            "moved_minus_animate_other": round(t_avg - a_avg, 6) if a_avg == a_avg else float("nan"),
            "roi_count": len(all_moved),
            "checkpoint": str(ckpt_file),
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
    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--input-csv", type=Path, required=True,
                   help="CSV produced by generate_simple_datasets/orc_semantic_distractors.py")
    p.add_argument("--max-seq-length", type=int, default=0)
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

    all_rows.sort(key=lambda r: (r["step"], r["gap_verb"]))
    write_rows(all_rows, args.result_name)
    print(f"\nResults saved to: {args.result_name}")


if __name__ == "__main__":
    main()
