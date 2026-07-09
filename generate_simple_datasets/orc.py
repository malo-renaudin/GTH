#!/usr/bin/env python3
"""
Generate ORC (Object Relative Clause) sentences at scale.

Design principles
-----------------
* "" is included directly in ADJ/ADV option lists, so every generated
  combination is unique by construction — no set-deduplication is needed.
* Each of the 20 n1-forms (10 nouns × singular/plural) is handled by an
  independent worker that streams sentences directly to a temp file using
  an 8 MB write buffer.  Workers run in parallel via multiprocessing.Pool.
* Temp files are concatenated into the final output.

Expected output (default vocabulary, passive enabled)
-----------------------------------------------------
  Active ORC  : ~1 164 000 000 sentences
  Passive ORC :   ~116 400 000 sentences
  Total       : ~1 280 400 000 sentences  (~70 GB plain text)

Estimated runtime : 2–6 min on a modern 4-core machine.
Estimated RAM     : ~200 MB (write buffers only).
"""

import argparse
import os
import shutil
import multiprocessing as mp
import time

# ---------------------------------------------------------------------------
# Verb–adverb compatibility
# ---------------------------------------------------------------------------
ADVERB_RESTRICTIONS: dict = {
    "follows": {"secretly"},   # following secretly is semantically odd
    "greets":  {"secretly"},   # greetings are by definition public
    "notices": {"openly"},     # noticing is perceptual, not a public act
}

PAST_PARTICIPLES: dict = {
    "visits":   "visited",
    "helps":    "helped",
    "avoids":   "avoided",
    "follows":  "followed",
    "greets":   "greeted",
    "contacts": "contacted",
    "guides":   "guided",
    "assists":  "assisted",
    "teaches":  "taught",
    "notices":  "noticed",
}

# ---------------------------------------------------------------------------
# Vocabulary
# "" at index 0 of ADJ/ADV lists means "no modifier".
# Including it here makes every combination unique — no dedup step needed.
# ---------------------------------------------------------------------------
N1_OPTS: list = [
    "boy", "student", "doctor", "artist", "athlete",
    "teacher", "lawyer", "farmer", "writer", "cook",
]
N2_OPTS: list = [
    "girl", "child", "pilot", "scientist", "engineer",
    "reporter", "officer", "manager", "dancer", "chef",
]
V_OPTS: list = [
    ("visits",   "visit"),
    ("helps",    "help"),
    ("avoids",   "avoid"),
    ("follows",  "follow"),
    ("greets",   "greet"),
    ("contacts", "contact"),
    ("guides",   "guide"),
    ("assists",  "assist"),
    ("teaches",  "teach"),
    ("notices",  "notice"),
]

ADJ1_OPTS: list = [
    "", "big", "tall", "young", "strong", "kind",
    "old", "quiet", "cheerful", "serious",
]  # 10 options: index 0 = no adjective
ADJ2_OPTS: list = [
    "", "beautiful", "smart", "brave", "famous", "honest",
    "wise", "gentle", "curious", "experienced",
]  # 10 options
ADV_OPTS: list = [
    "", "possibly", "apparently", "secretly", "always", "often",
    "rarely", "frequently", "openly", "regularly",
]  # 10 options

REL_OPTS: list = ["that", "who", ""]   # "" = zero-relative clause
AUX_LIST: list = [
    "", "did ", "could ", "couldn't ", "didn't ",
    "would ", "wouldn't ", "should ", "shouldn't ", "might ",
]  # 10 options; non-empty items carry a trailing space

CONT_SING: list = [
    "is eating an apple",    "is watching a movie",   "is reading a book",
    "likes to dance",        "enjoys music",           "likes climbing",
    "is taking a walk",      "is cooking dinner",      "is playing chess",
    "is practicing guitar",
]
CONT_PLUR: list = [
    "are eating an apple",   "are watching a movie",  "are reading a book",
    "like to dance",         "enjoy music",            "like climbing",
    "are taking a walk",     "are cooking dinner",     "are playing chess",
    "are practicing guitar",
]

_IRREGULARS: dict = {"child": "children", "man": "men", "woman": "women"}


def _pluralize(noun: str) -> str:
    return _IRREGULARS.get(noun, noun + "s")


# ---------------------------------------------------------------------------
# Worker — streams all sentences for one (n1_form, n1_is_plural) pair
# ---------------------------------------------------------------------------
def _worker(task: tuple) -> tuple:
    """
    Generate every ORC sentence whose head noun is *n1_form* and stream
    them to *tmp_path*.  Returns (sentence_count, tmp_path).
    """
    n1_form, n1_is_plural, tmp_path, include_passive = task
    conts = CONT_PLUR if n1_is_plural else CONT_SING
    n_conts = len(conts)
    count = 0

    with open(tmp_path, "w", buffering=8 * 1024 * 1024) as fh:
        for n2_base in N2_OPTS:
            for n2_is_plural in (False, True):
                n2_form = _pluralize(n2_base) if n2_is_plural else n2_base

                for v_sing, v_plur in V_OPTS:
                    v_pp = PAST_PARTICIPLES[v_sing]
                    restricted = ADVERB_RESTRICTIONS.get(v_sing, set())
                    valid_advs = [a for a in ADV_OPTS if a not in restricted]

                    for adj1 in ADJ1_OPTS:
                        a1 = adj1 + " " if adj1 else ""
                        for adj2 in ADJ2_OPTS:
                            a2 = adj2 + " " if adj2 else ""
                            for adv in valid_advs:
                                av = adv + " " if adv else ""
                                for rel in REL_OPTS:
                                    r = " " + rel if rel else ""

                                    # ── Active ORC ──────────────────────────
                                    # Template:
                                    #   The [adj1] N1 [rel] the [adj2] N2 [aux][adv] V_form CONT.
                                    for aux in AUX_LIST:
                                        v_form = (
                                            v_plur if aux                        # base form after auxiliary
                                            else (v_plur if n2_is_plural         # plural agreement
                                                  else v_sing)
                                        )
                                        prefix = (
                                            f"The {a1}{n1_form}{r}"
                                            f" the {a2}{n2_form}"
                                            f" {aux}{av}{v_form}"
                                        )
                                        for cont in conts:
                                            fh.write(prefix)
                                            fh.write(" ")
                                            fh.write(cont)
                                            fh.write(".\n")
                                        count += n_conts

                                    # ── Passive ORC ─────────────────────────
                                    if include_passive:
                                        if rel:
                                            # Full passive relative:
                                            #   "N1 that/who was/were [adv] V-pp by N2"
                                            aux_p = "were " if n1_is_plural else "was "
                                            prefix = (
                                                f"The {a1}{n1_form}{r}"
                                                f" {aux_p}{av}{v_pp}"
                                                f" by the {a2}{n2_form}"
                                            )
                                        else:
                                            # Zero-relative → reduced participial:
                                            #   "N1 [adv] V-pp by N2"  (no was/were)
                                            prefix = (
                                                f"The {a1}{n1_form}"
                                                f" {av}{v_pp}"
                                                f" by the {a2}{n2_form}"
                                            )
                                        for cont in conts:
                                            fh.write(prefix)
                                            fh.write(" ")
                                            fh.write(cont)
                                            fh.write(".\n")
                                        count += n_conts

    return count, tmp_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream ~1.28 B grammatical ORC sentences to disk.\n"
            "  Disk usage  : ~70 GB plain text\n"
            "  RAM usage   : ~200 MB\n"
            "  Runtime     : ~2-6 min on a 4-core machine"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_file", default="datasets/orc.txt",
                        help="Destination file (default: datasets/orc.txt).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = all CPU cores, default: 0).")
    parser.add_argument("--no-passive", dest="no_passive", action="store_true",
                        help="Omit passive ORC sentences (~116 M fewer sentences).")
    # kept for backwards-compatibility; value is ignored (always generates ORC)
    parser.add_argument("--structures", type=int, default=1,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    include_passive = not args.no_passive
    n_workers = args.workers or mp.cpu_count()

    # Build all (n1_form, n1_is_plural) pairs: 10 nouns x 2 = 20 tasks
    n1_items = []
    for noun in N1_OPTS:
        n1_items.append((noun, False))
        n1_items.append((_pluralize(noun), True))

    base = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)

    tasks = [
        (form, is_plur, f"{base}.part{i:03d}", include_passive)
        for i, (form, is_plur) in enumerate(n1_items)
    ]

    # Approximate expected count (informational)
    # 20 n1 x 20 n2 x 10 v x 10 adj1 x 10 adj2 x ~9.7 adv x 3 rel x 10 aux x 10 cont
    expected_active  = 1_164_000_000
    expected_passive = 116_400_000 if include_passive else 0
    expected_total   = expected_active + expected_passive
    expected_gb      = expected_total * 70 // 1_000_000_000

    print(f"Workers  : {n_workers}")
    print(f"Tasks    : {len(tasks)}  (one per head-noun form)")
    print(f"Passive  : {'yes' if include_passive else 'no'}")
    print(f"Expected : ~{expected_total:,} sentences  (~{expected_gb} GB)")
    print(f"Output   : {base}")
    print("Generating...")

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))

    gen_elapsed = time.time() - t0
    total = sum(c for c, _ in results)
    rate  = total / gen_elapsed / 1_000_000
    print(f"\nGenerated {total:,} sentences in {gen_elapsed:.1f}s  ({rate:.2f} M sent/s).")

    print("Merging parts into final file...")
    t1 = time.time()
    with open(base, "wb") as out:
        # sort by tmp path to keep a deterministic output order
        for _, tmp in sorted(results, key=lambda x: x[1]):
            with open(tmp, "rb") as inp:
                shutil.copyfileobj(inp, out, length=16 * 1024 * 1024)
            os.remove(tmp)

    merge_elapsed = time.time() - t1
    total_elapsed = time.time() - t0
    print(f"Merged   : {merge_elapsed:.1f}s")
    print(f"Total    : {total_elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()