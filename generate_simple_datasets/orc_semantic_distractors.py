#!/usr/bin/env python3
"""
generate_simple_datasets/orc_semantic_distractors.py

Generate ORC sentences for the semantic distractor evaluation and write a CSV.

Structure:  The N1 [that|who] the N2 VERB CONTINUATION.

Two conditions:
  1. animate N1 + inanimate-selecting VERB
       e.g. "The boy that the girl eats is old."
       plausible_objects = ["the pizza", "the cake", ...]
  2. inanimate N1 + animate-selecting VERB
       e.g. "The car that the girl interviews is old."
       plausible_objects = ["the doctor", "the student", ...]

Verbs are strictly unambiguous: each verb is labelled as selecting ONLY
inanimate objects or ONLY animate objects. Ambiguous verbs (finds, uses,
touches, paints, holds, carries, buys, cleans) are excluded.

No adjectives, no adverbs.

CSV columns:
  sentence            full surface string
  pre_gap_text        context up to and including the verb (gap is after this)
  moved_np            "the N1"  (the filler that should be recovered at the gap)
  plausible_objects   "|"-separated list of "the noun" distractors
  moved_np_animacy    "animate" | "inanimate"
  verb_expects_animate  "yes" | "no"

Usage
-----
python generate_simple_datasets/orc_semantic_distractors.py \\
    --output-csv data/orc_semantic_distractors.csv \\
    --n-sentences 1000 \\
    --seed 42
"""
import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Animate nouns used as N1 (filler) in condition 1, and as N2 (embedded subj)
# in both conditions.
ANIMATE_N1 = ["boy", "girl", "student", "doctor", "artist", "athlete", "pilot", "officer"]
ANIMATE_N2 = ["girl", "boy", "child", "scientist", "engineer", "patient", "client", "tourist"]

# Inanimate nouns used as N1 (filler) in condition 2.
INANIMATE_N1 = ["car", "house", "book", "phone", "computer", "table", "door", "ball",
                "stone", "bridge", "train", "cup"]

# Verbs that strongly select INANIMATE objects only.
# Excluded: uses (general), finds (general), touches (general), holds (can hold
# people), carries (can carry people), buys (can hire), paints (portraits), cleans.
INANIMATE_VERBS: Dict[str, List[str]] = {
    "eats":   ["pizza", "cake", "bread", "apple", "cookie"],
    "reads":  ["book", "newspaper", "letter", "magazine"],
    "builds": ["house", "bridge", "table", "door", "wall"],
    "drives": ["car", "truck", "bus", "train"],
    "fixes":  ["car", "door", "phone", "computer"],
    "opens":  ["door", "window", "bottle", "box"],
    "breaks": ["window", "door", "phone", "cup"],
    "lifts":  ["box", "stone", "bag", "table"],
    "throws": ["ball", "stone", "bag", "book"],
}

# Verbs that strongly select ANIMATE objects only.
ANIMATE_VERBS: Dict[str, List[str]] = {
    "interviews": ["doctor", "student", "pilot", "officer", "scientist"],
    "hires":      ["doctor", "engineer", "pilot", "officer", "scientist"],
    "meets":      ["doctor", "student", "pilot", "officer", "engineer"],
    "trusts":     ["doctor", "student", "pilot", "officer", "engineer"],
    "teaches":    ["student", "child", "engineer", "scientist", "artist"],
    "greets":     ["doctor", "student", "pilot", "officer", "tourist"],
    "thanks":     ["doctor", "student", "pilot", "officer", "engineer"],
    "advises":    ["doctor", "student", "engineer", "scientist", "officer"],
}

RELATIVIZERS_ANIMATE   = ["that", "who"]
RELATIVIZERS_INANIMATE = ["that"]

# Continuations neutral enough for both animate and inanimate subjects.
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

def make_row(
    n1: str, n2: str,
    verb: str, plausible_nouns: List[str],
    rel: str, cont: str,
    moved_np_animacy: str, verb_expects_animate: str,
) -> dict:
    sentence = f"The {n1} {rel} the {n2} {verb} {cont}.".capitalize()
    pre_gap_text = f"The {n1} {rel} the {n2} {verb}".capitalize()
    moved_np = f"the {n1}"
    plausible_objects = "|".join(f"the {noun}" for noun in plausible_nouns)
    return {
        "sentence": sentence,
        "pre_gap_text": pre_gap_text,
        "moved_np": moved_np,
        "plausible_objects": plausible_objects,
        "moved_np_animacy": moved_np_animacy,
        "verb_expects_animate": verb_expects_animate,
    }


def generate_all() -> List[dict]:
    rows: List[dict] = []
    seen: set = set()

    # Condition 1: animate N1, inanimate-selecting verb
    for n1, n2, (verb, distractors), rel, cont in itertools.product(
        ANIMATE_N1, ANIMATE_N2,
        INANIMATE_VERBS.items(),
        RELATIVIZERS_ANIMATE,
        CONTINUATIONS,
    ):
        if n1 == n2:
            continue
        key = (n1, n2, verb, rel, cont)
        if key in seen:
            continue
        seen.add(key)
        rows.append(make_row(n1, n2, verb, distractors, rel, cont,
                             moved_np_animacy="animate",
                             verb_expects_animate="no"))

    # Condition 2: inanimate N1, animate-selecting verb
    for n1, n2, (verb, distractors), cont in itertools.product(
        INANIMATE_N1, ANIMATE_N2,
        ANIMATE_VERBS.items(),
        CONTINUATIONS,
    ):
        rel = "that"  # inanimate N1 only takes "that"
        if n1 == n2:
            continue
        key = (n1, n2, verb, rel, cont)
        if key in seen:
            continue
        seen.add(key)
        rows.append(make_row(n1, n2, verb, distractors, rel, cont,
                             moved_np_animacy="inanimate",
                             verb_expects_animate="yes"))

    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate ORC semantic distractor CSV."
    )
    p.add_argument("--output-csv", type=Path,
                   default=Path("data/orc_semantic_distractors.csv"))
    p.add_argument("--n-sentences", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    print("Generating sentences ...")
    all_rows = generate_all()

    # Count per condition
    n_animate   = sum(1 for r in all_rows if r["moved_np_animacy"] == "animate")
    n_inanimate = sum(1 for r in all_rows if r["moved_np_animacy"] == "inanimate")
    print(f"Total unique rows: {len(all_rows)}  "
          f"(animate filler: {n_animate}, inanimate filler: {n_inanimate})")

    if len(all_rows) <= args.n_sentences:
        selected = all_rows
        print(f"[info] Fewer rows than requested — writing all {len(selected)}.")
    else:
        # Sample while preserving rough balance between conditions
        half = args.n_sentences // 2
        animate_rows   = [r for r in all_rows if r["moved_np_animacy"] == "animate"]
        inanimate_rows = [r for r in all_rows if r["moved_np_animacy"] == "inanimate"]
        sel_anim = random.sample(animate_rows,   min(half, len(animate_rows)))
        sel_inan = random.sample(inanimate_rows, min(args.n_sentences - len(sel_anim),
                                                     len(inanimate_rows)))
        selected = sel_anim + sel_inan
        random.shuffle(selected)
        print(f"Sampled {len(sel_anim)} animate + {len(sel_inan)} inanimate = "
              f"{len(selected)} rows")

    selected.sort(key=lambda r: r["sentence"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = ["sentence", "pre_gap_text", "moved_np", "plausible_objects",
              "moved_np_animacy", "verb_expects_animate"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {len(selected)} rows to {args.output_csv}")

    # Quick sanity check: one row per verb
    print("\nSample (first row per verb):")
    seen_verbs: set = set()
    for row in selected:
        # extract verb from pre_gap_text (last whitespace-delimited token)
        verb = row["pre_gap_text"].strip().rstrip(".").split()[-1]
        if verb not in seen_verbs:
            print(f"  [{verb:12s}] {row['sentence']}")
            print(f"              moved_np={row['moved_np']}  "
                  f"animacy={row['moved_np_animacy']}  "
                  f"verb_expects_animate={row['verb_expects_animate']}")
            seen_verbs.add(verb)


if __name__ == "__main__":
    main()
