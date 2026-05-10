"""
eval_gap_np_orc.py

At the ORC gap site (right after the embedded verb), measure:
  - target_mass : geomean P(moved NP)  — the filler NP that was displaced
  - vocab_mass  : mean geomean P(NP) over vocabulary NPs (WH noun forms),
                  excluding the actual moved NP from the average

This checks whether the model specifically recovers the moved NP at the gap,
beyond what it would predict for any generic NP.
"""
import argparse
import csv
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Set, Tuple

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

# Nouns used to identify the moved NP in ORC sentences
NOUNS_ORC = [
    "boy", "boys", "student", "students", "doctor", "doctors",
    "artist", "artists", "athlete", "athletes", "girl", "girls",
    "child", "children", "pilot", "pilots", "scientist", "scientists",
    "engineer", "engineers",
]

# NP vocabulary used as comparators (WH-question noun set)
VOCAB_NOUNS = [
    "student", "doctor", "pilot", "officer", "athlete", "artist",
    "child", "girl", "boy", "patient", "client", "tourist",
]

# Embedded verbs that mark the gap site in ORC
VERBS_ORC = [
    "visits", "visit", "helps", "help", "avoids", "avoid",
    "follows", "follow", "greets", "greet",
]

CSV_FIELDS = [
    "step", "structure",
    "target_label", "comparator_label",
    "target_mass", "target_mass_median",
    "vocab_mass", "vocab_mass_median",
    "target_minus_vocab",
    "roi_count", "checkpoint",
]


def _build_np_token_lists(nouns: List[str], tokenizer: Tokenizer) -> List[Tuple[str, List[int]]]:
    """Return [(noun, token_ids_for_'the noun'), ...] for every noun."""
    return [
        (noun, tokenizer.encode(f" the {noun}", bos=False, eos=False).tolist())
        for noun in nouns
    ]


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

    roi_verb_set: Set[str] = set(VERBS_ORC)
    vocab_np_list = _build_np_token_lists(VOCAB_NOUNS, tokenizer)

    # Pre-tokenize and locate gap site (last token of embedded verb)
    valid_items: List[Tuple[List[int], int, str]] = []
    for s in sentences:
        tok, spans = _word_token_spans(s, tokenizer)
        roi_idx: Optional[int] = None
        for _, (word, start, end) in enumerate(spans):
            if normalize_word(word) in roi_verb_set:
                roi_idx = end - 1
                break
        if roi_idx is not None:
            valid_items.append((tok, roi_idx, s))

    t_vals: List[float] = []
    v_vals: List[float] = []

    for batch_start in tqdm(range(0, len(valid_items), batch_size), desc="  batches", leave=False):
        batch = valid_items[batch_start: batch_start + batch_size]
        contexts = [tok[: roi_idx + 1] for tok, roi_idx, _ in batch]
        ctx_lengths = [len(c) for c in contexts]
        max_ctx = max(ctx_lengths)
        x_batch = torch.tensor(
            [c + [0] * (max_ctx - len(c)) for c in contexts], device=device
        )
        with torch.no_grad():
            logits_batch = model(x_batch)

        for i, (tok, roi_idx, s) in enumerate(batch):
            sp = torch.softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)
            x  = torch.tensor(contexts[i], device=device).unsqueeze(0)

            def geomean_prob(ids: List[int]) -> float:
                lp = np_chain_logprob(model, x, ids, device, first_step_probs=sp)
                return lp ** (1.0 / len(ids))

            # target: the moved NP
            np_words, head = extract_orc_moved_np(s, set(NOUNS_ORC))
            if not np_words or head is None:
                continue
            flat_ids = [
                t for w in np_words
                for t in tokenizer.encode(" " + w, bos=False, eos=False).tolist()
            ]
            target = geomean_prob(flat_ids)

            # comparator: vocab NPs, excluding the actual moved NP
            moved_head_norm = normalize_word(head)
            comparator_probs = [
                geomean_prob(ids)
                for noun, ids in vocab_np_list
                if normalize_word(noun) != moved_head_norm and ids
            ]
            if not comparator_probs:
                continue

            t_vals.append(target)
            v_vals.append(statistics.mean(comparator_probs))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step = step_from_checkpoint(ckpt_file)
    n = len(t_vals)
    if n == 0:
        t_avg = t_med = v_avg = v_med = 0.0
    else:
        t_avg, t_med = statistics.mean(t_vals), statistics.median(t_vals)
        v_avg, v_med = statistics.mean(v_vals), statistics.median(v_vals)

    print(
        f"  step {step}: moved_np={t_avg:.6f} (med={t_med:.6f}), "
        f"vocab_np={v_avg:.6f} (med={v_med:.6f}), n={n}"
    )

    return {
        "step": step,
        "structure": "orc",
        "target_label": "moved_np_geomean_prob",
        "comparator_label": "vocab_np_geomean_prob",
        "target_mass": round(t_avg, 6),
        "target_mass_median": round(t_med, 6),
        "vocab_mass": round(v_avg, 6),
        "vocab_mass_median": round(v_med, 6),
        "target_minus_vocab": round(t_avg - v_avg, 6),
        "roi_count": n,
        "checkpoint": str(ckpt_file),
    }


def worker_unpack(args: Tuple) -> dict:
    return compute_one_checkpoint(*args)


def write_rows(rows: List[dict], result_name: Path) -> None:
    result_name.parent.mkdir(parents=True, exist_ok=True)
    with open(result_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="ORC gap-site: moved NP probability vs vocabulary NP probability."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--sentences-file", type=Path, required=True,
                   help="One ORC sentence per line.")
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--result-name", type=Path, default=Path("results/eval_gap_np_orc.csv"))
    args = p.parse_args()

    sentences = read_nonempty_lines(args.sentences_file)

    if args.checkpoint is not None:
        ckpt = resolve_checkpoint_file(args.checkpoint)
        rows = [compute_one_checkpoint(
            ckpt, args.tokenizer_dir, sentences, args.max_seq_length, args.batch_size
        )]
        write_rows(rows, args.result_name)
        print(f"Results saved to: {args.result_name}")
        return

    ckpts = sorted(
        args.checkpoint_dir.glob("step-*/lit_model.pth"),
        key=lambda x: int(x.parent.name.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No step checkpoints found under {args.checkpoint_dir}")

    items = [
        (ckpt, args.tokenizer_dir, sentences, args.max_seq_length, args.batch_size)
        for ckpt in ckpts
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=args.num_processes) as ex:
        for r in tqdm(ex.map(worker_unpack, items), total=len(items), desc="checkpoints"):
            rows.append(r)

    rows.sort(key=lambda r: r["step"])
    write_rows(rows, args.result_name)
    print(f"Results saved to: {args.result_name}")


if __name__ == "__main__":
    main()