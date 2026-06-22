#!/usr/bin/env python3
"""
generate_simple_datasets/orc_transitivity.py

Generate ORC sentences for the transitivity evaluation and write a CSV.

Structure:  The N1 [that|who] the N2 VERB CONTINUATION.

Two conditions (matched on N1, N2, relativizer, continuation):
  transitive:   "The boy that the girl sees is old."
                → gap is licensed: the verb takes an object, which should be N1
  intransitive: "The boy that the girl sleeps is old."
                → gap is unlicensed: the verb cannot take an object

Key question: does the model assign higher probability mass to the moved NP
(N1) at the gap site when the verb is transitive vs intransitive?

  - Syntactic copy model: moved_mass(transitive) > moved_mass(intransitive)
  - Surface/positional copy model: moved_mass similar in both conditions

CSV columns:
  sentence          full surface string
  pre_gap_text      context up to and including the verb (gap probed here)
  moved_np          "the N1"
  verb              the embedded verb
  verb_type         "transitive" | "intransitive"
  moved_np_animacy  always "animate" (N1 is always animate)

Usage
-----
python generate_simple_datasets/orc_transitivity.py \\
    --output-csv data/orc_transitivity.csv \\
    --n-sentences 1000 \\
    --seed 42
"""
import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

ANIMATE_N1 = ["boy", "girl", "student", "doctor", "artist", "athlete", "pilot", "officer"]
ANIMATE_N2 = ["girl", "boy", "child", "scientist", "engineer", "patient", "client", "tourist"]

# Transitive verbs: clearly take a direct object (gap-licensing).
TRANSITIVE_VERBS = ["sees", "knows", "likes", "helps", "follows", "watches", "visits", "calls"]

# Intransitive verbs: cannot take a direct object (no gap licensing).
# Excluded ambitransitives: runs (runs a company), walks (walks the dog).
INTRANSITIVE_VERBS = ["sleeps", "laughs", "cries", "sneezes", "yawns", "sits", "waits", "smiles"]

RELATIVIZERS = ["that", "who", ""]  # animate N1: overt relativizer or zero

CONTINUATIONS = [
    "is here",
    "is old",
    "is new",
    "is nearby",
    "is interesting",
    "matters",
    "is well known",
    "is important",
]

# ---------------------------------------------------------------------------
# Row generation
# ---------------------------------------------------------------------------

def make_row(n1: str, n2: str, verb: str, verb_type: str, rel: str, cont: str) -> dict:
    if rel:
        core = f"The {n1} {rel} the {n2} {verb}"
    else:
        core = f"The {n1} the {n2} {verb}"
    sentence     = f"{core} {cont}.".capitalize()
    pre_gap_text = core.capitalize()
    return {
        "sentence":         sentence,
        "pre_gap_text":     pre_gap_text,
        "moved_np":         f"the {n1}",
        "verb":             verb,
        "verb_type":        verb_type,
        "relativizer":      rel if rel else "zero",
        "moved_np_animacy": "animate",
    }


def generate_all() -> List[dict]:
    rows: List[dict] = []
    seen: set = set()

    for verb_list, verb_type in [
        (TRANSITIVE_VERBS,   "transitive"),
        (INTRANSITIVE_VERBS, "intransitive"),
    ]:
        for n1, n2, verb, rel, cont in itertools.product(
            ANIMATE_N1, ANIMATE_N2, verb_list, RELATIVIZERS, CONTINUATIONS
        ):
            if n1 == n2:
                continue
            key = (n1, n2, verb, rel, cont)
            if key in seen:
                continue
            seen.add(key)
            rows.append(make_row(n1, n2, verb, verb_type, rel, cont))

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate ORC transitivity CSV."
    )
    p.add_argument("--output-csv",    type=Path, default=Path("data/orc_transitivity.csv"))
    p.add_argument("--n-sentences",   type=int,  default=1000)
    p.add_argument("--seed",          type=int,  default=42)
    args = p.parse_args()

    random.seed(args.seed)
    print("Generating sentences ...")
    all_rows = generate_all()

    trans_rows = [r for r in all_rows if r["verb_type"] == "transitive"]
    intr_rows  = [r for r in all_rows if r["verb_type"] == "intransitive"]
    print(f"Total unique rows: {len(all_rows)}  "
          f"(transitive: {len(trans_rows)}, intransitive: {len(intr_rows)})")

    if len(all_rows) <= args.n_sentences:
        selected = all_rows
        print(f"[info] Fewer rows than requested — writing all {len(selected)}.")
    else:
        half = args.n_sentences // 2
        sel_trans = random.sample(trans_rows, min(half, len(trans_rows)))
        sel_intr  = random.sample(intr_rows,  min(args.n_sentences - len(sel_trans), len(intr_rows)))
        selected  = sel_trans + sel_intr
        random.shuffle(selected)
        print(f"Sampled {len(sel_trans)} transitive + {len(sel_intr)} intransitive = "
              f"{len(selected)} rows")

    selected.sort(key=lambda r: (r["verb_type"], r["sentence"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = ["sentence", "pre_gap_text", "moved_np", "verb", "verb_type", "relativizer", "moved_np_animacy"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {len(selected)} rows to {args.output_csv}")

    # Sanity check: one row per verb
    print("\nSample (first row per verb):")
    seen_verbs: set = set()
    for row in selected:
        if row["verb"] not in seen_verbs:
            print(f"  [{row['verb_type']:12s} | {row['verb']:10s}] {row['sentence']}")
            seen_verbs.add(row["verb"])


if __name__ == "__main__":
    main()
