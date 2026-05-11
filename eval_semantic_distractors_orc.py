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
    --sentences-file data/orc_eating_verbs.txt \\
    --result-name results/semantic_distractors.csv
"""
import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

from litgpt import Tokenizer

from utils import (
    ORC_REL_MARKERS,
    extract_orc_moved_np,
    load_checkpoint,
    normalize_word,
    np_chain_logprob,
    read_nonempty_lines,
    resolve_checkpoint_file,
    step_from_checkpoint,
    _word_token_spans,
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

# Full noun list used by extract_orc_moved_np to locate the filler head
NOUNS_ORC = [
    "boy", "boys", "student", "students", "doctor", "doctors",
    "artist", "artists", "athlete", "athletes", "girl", "girls",
    "child", "children", "pilot", "pilots", "scientist", "scientists",
    "engineer", "engineers",
]

# Verb → semantically plausible inanimate objects.
# Only nouns that genuinely co-occur with the verb in context are listed;
# these serve as distractors when the moved NP is animate.
# Nouns chosen from frequent items in train_baseline_2_augmented.txt.
VERB_INANIMATE_DISTRACTORS: Dict[str, List[str]] = {
    "eat":     ["pizza", "cake", "bread", "apple", "cookie"],
    "eats":    ["pizza", "cake", "bread", "apple", "cookie"],
    "ate":     ["pizza", "cake", "bread", "apple", "cookie"],
    "read":    ["book", "newspaper", "letter", "magazine"],
    "reads":   ["book", "newspaper", "letter", "magazine"],
    "carry":   ["bag", "box", "book", "key"],
    "carries": ["bag", "box", "book", "key"],
    "carried": ["bag", "box", "book", "key"],
    "build":   ["house", "bridge", "table", "door"],
    "builds":  ["house", "bridge", "table", "door"],
    "built":   ["house", "bridge", "table", "door"],
    "drive":   ["car", "truck", "bus", "train"],
    "drives":  ["car", "truck", "bus", "train"],
    "drove":   ["car", "truck", "bus", "train"],
    "fix":     ["car", "door", "phone", "computer"],
    "fixes":   ["car", "door", "phone", "computer"],
    "fixed":   ["car", "door", "phone", "computer"],
    "use":     ["computer", "phone", "key", "book"],
    "uses":    ["computer", "phone", "key", "book"],
    "used":    ["computer", "phone", "key", "book"],
    "buy":     ["book", "car", "phone", "house"],
    "buys":    ["book", "car", "phone", "house"],
    "bought":  ["book", "car", "phone", "house"],
    "sell":    ["car", "house", "book", "phone"],
    "sells":   ["car", "house", "book", "phone"],
    "sold":    ["car", "house", "book", "phone"],
    "clean":   ["car", "house", "table", "window"],
    "cleans":  ["car", "house", "table", "window"],
    "paint":   ["house", "door", "table", "car"],
    "paints":  ["house", "door", "table", "car"],
    "open":    ["door", "window", "bottle", "box"],
    "opens":   ["door", "window", "bottle", "box"],
    "break":   ["window", "door", "phone", "cup"],
    "breaks":  ["window", "door", "phone", "cup"],
    "broke":   ["window", "door", "phone", "cup"],
    "hold":    ["bag", "box", "cup", "key"],
    "holds":   ["bag", "box", "cup", "key"],
    "held":    ["bag", "box", "cup", "key"],
    "find":    ["key", "book", "phone", "bag"],
    "finds":   ["key", "book", "phone", "bag"],
    "found":   ["key", "book", "phone", "bag"],
    "lift":    ["box", "stone", "bag", "table"],
    "lifts":   ["box", "stone", "bag", "table"],
    "throw":   ["ball", "stone", "bag", "book"],
    "throws":  ["ball", "stone", "bag", "book"],
    "threw":   ["ball", "stone", "bag", "book"],
    "touch":   ["stone", "door", "table", "ball"],
    "touches": ["stone", "door", "table", "ball"],
}

# All unique inanimate nouns across all verb entries (for moved-NP animacy check)
INANIMATE_SET: Set[str] = {
    noun for nouns in VERB_INANIMATE_DISTRACTORS.values() for noun in nouns
}

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


def extract_orc_embedded_np(
    sentence: str, noun_candidates: Set[str]
) -> Tuple[List[str], Optional[str]]:
    """Extract the embedded subject NP immediately after the relativizer.

    e.g. "the girl that the boy eats" → (["the", "boy"], "boy")
    Returns ([], None) if not found.
    """
    words = sentence.split()
    rel_idx = None
    for i, w in enumerate(words):
        if normalize_word(w) in ORC_REL_MARKERS:
            rel_idx = i
            break
    if rel_idx is None or rel_idx + 2 >= len(words):
        return [], None
    if normalize_word(words[rel_idx + 1]) != "the":
        return [], None
    head = normalize_word(words[rel_idx + 2])
    if head not in noun_candidates:
        return [], None
    return ["the", head], head


def _noun_animacy(head: str) -> str:
    if head in ANIMATE_SET:
        return "animate"
    if head in INANIMATE_SET:
        return "inanimate"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def compute_one_checkpoint(
    ckpt_file: Path,
    tokenizer_dir: Path,
    sentences: List[str],
    max_seq_length: int,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(tokenizer_dir)
    model = load_checkpoint(ckpt_file, device, max_seq_length)

    noun_candidates = {
        normalize_word(w) for w in NOUNS_ORC if normalize_word(w) not in {"the", "a", "an"}
    }

    # Pre-build NP token lists for animate vocab
    animate_np_list = _build_np_token_lists(ANIMATE_NOUNS, tokenizer)

    # Pre-tokenize and find gap site for every sentence
    # gap_verb_set = all verbs we know about
    known_verbs = set(VERB_INANIMATE_DISTRACTORS.keys())

    valid_items: List[Tuple[List[int], int, str, str]] = []  # (tok_ids, roi_idx, sentence, gap_verb)
    skipped = 0
    for s in sentences:
        tok, spans = _word_token_spans(s, tokenizer)
        roi_idx: Optional[int] = None
        gap_verb: Optional[str] = None
        for word, start, end in spans:
            w_norm = normalize_word(word)
            if w_norm in known_verbs:
                roi_idx = end - 1
                gap_verb = w_norm
                break
        if roi_idx is not None and gap_verb is not None:
            valid_items.append((tok, roi_idx, s, gap_verb))
        else:
            skipped += 1

    if skipped:
        print(f"  [warn] {skipped} sentence(s) skipped (gap verb not in VERB_INANIMATE_DISTRACTORS)")

    # Per-verb accumulator: verb → {moved, distractor, animate_other}
    by_verb: Dict[str, Dict[str, List[float]]] = {}

    for batch_start in tqdm(range(0, len(valid_items), batch_size), desc="  batches", leave=False):
        batch = valid_items[batch_start: batch_start + batch_size]

        # Forward pass over context (up to and including the gap verb's last token)
        contexts = [tok[:roi_idx + 1] for tok, roi_idx, _, _ in batch]
        ctx_lengths = [len(c) for c in contexts]
        max_ctx = max(ctx_lengths)
        x_batch = torch.tensor(
            [c + [0] * (max_ctx - len(c)) for c in contexts], device=device
        )
        with torch.no_grad():
            logits_batch = model(x_batch)

        for i, (tok, roi_idx, s, gap_verb) in enumerate(batch):
            sp = torch.softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)
            x_ctx = torch.tensor(contexts[i], device=device).unsqueeze(0)

            def geomean_prob(ids: List[int]) -> float:
                if not ids:
                    return float("nan")
                lp = np_chain_logprob(model, x_ctx, ids, device, first_step_probs=sp)
                return lp ** (1.0 / len(ids))

            # --- Moved NP ---
            np_words, head = extract_orc_moved_np(s, set(NOUNS_ORC))
            if not np_words or head is None:
                continue
            moved_ids = [
                t for w in np_words
                for t in tokenizer.encode(" " + w, bos=False, eos=False).tolist()
            ]
            moved_mass = geomean_prob(moved_ids)
            animacy = _noun_animacy(normalize_word(head))

            # --- Semantic distractors (opposite animacy) ---
            if animacy == "animate":
                # Verb-specific inanimate distractors
                distractor_nouns = VERB_INANIMATE_DISTRACTORS[gap_verb]
                distractor_np_list = _build_np_token_lists(distractor_nouns, tokenizer)
                distractor_probs = [
                    geomean_prob(ids) for _, ids in distractor_np_list if ids
                ]
            else:
                # Inanimate moved NP → animate distractors
                moved_head_norm = normalize_word(head)
                distractor_probs = [
                    geomean_prob(ids)
                    for noun, ids in animate_np_list
                    if normalize_word(noun) != moved_head_norm and ids
                ]

            if not distractor_probs:
                continue

            # --- Other animate NPs (non-filler) ---
            moved_head_norm = normalize_word(head)
            # Also exclude embedded NP head from animate_other
            _, emb_head = extract_orc_embedded_np(s, noun_candidates)
            exclude = {moved_head_norm}
            if emb_head:
                exclude.add(normalize_word(emb_head))
            animate_other_probs = [
                geomean_prob(ids)
                for noun, ids in animate_np_list
                if normalize_word(noun) not in exclude and ids
            ]

            if gap_verb not in by_verb:
                by_verb[gap_verb] = {"moved": [], "distractor": [], "animate_other": []}
            by_verb[gap_verb]["moved"].append(moved_mass)
            by_verb[gap_verb]["distractor"].append(statistics.mean(distractor_probs))
            if animate_other_probs:
                by_verb[gap_verb]["animate_other"].append(statistics.mean(animate_other_probs))

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

        # Infer animacy from verb direction (all sentences with this verb should share direction)
        animacy_label = "animate"  # default; we tracked by verb so this is consistent
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
    p.add_argument("--sentences-file", type=Path, required=True,
                   help="One ORC sentence per line. Sentences whose embedded verb "
                        "is not in the known verb list are skipped.")
    p.add_argument("--max-seq-length", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--result-name", type=Path,
                   default=Path("results/eval_semantic_distractors_orc.csv"))
    args = p.parse_args()

    sentences = read_nonempty_lines(args.sentences_file)
    print(f"Loaded {len(sentences)} sentences from {args.sentences_file}")
    print(f"Known verbs: {sorted(set(v.rstrip('s') for v in VERB_INANIMATE_DISTRACTORS))}")

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
            ckpt_file, args.tokenizer_dir, sentences, args.max_seq_length, args.batch_size
        )
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["step"], r["gap_verb"]))
    write_rows(all_rows, args.result_name)
    print(f"\nResults saved to: {args.result_name}")


if __name__ == "__main__":
    main()
