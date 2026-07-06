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

from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Helpers (previously in utils)
# ---------------------------------------------------------------------------

ORC_REL_MARKERS = {"that", "who", "which"}


def normalize_word(w: str) -> str:
    return w.strip().lower().rstrip(".,!?;:")


def read_nonempty_lines(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _word_token_spans(sentence: str, tokenizer) -> Tuple[List[int], List[Tuple[str, int, int]]]:
    """Tokenize sentence; return (all_ids, [(word, token_start, token_end), ...])."""
    words  = sentence.split()
    tokens: List[int] = []
    spans:  List[Tuple[str, int, int]] = []
    for word in words:
        prefix   = " " if tokens else ""
        word_ids = tokenizer.encode(prefix + word, add_special_tokens=False)
        start    = len(tokens)
        tokens.extend(word_ids)
        spans.append((word, start, len(tokens)))
    return tokens, spans


def extract_orc_moved_np(
    sentence: str, noun_candidates: Set[str]
) -> Tuple[List[str], Optional[str]]:
    """Return the initial NP (filler) in an ORC, e.g. (['The', 'girl'], 'girl')."""
    words = sentence.split()
    if len(words) < 2:
        return [], None
    if normalize_word(words[0]) != "the":
        return [], None
    head = normalize_word(words[1])
    if head not in noun_candidates:
        return [], None
    return [words[0], words[1]], head


def _np_chain_prob(
    model, x_ctx: torch.Tensor, ids: List[int],
    device: torch.device, first_step_probs=None,
) -> float:
    """Product of P(t_i | ctx, t_0..t_{i-1}) for each token id in ids."""
    prob = 1.0
    cur  = x_ctx.clone()
    for i, tid in enumerate(ids):
        if i == 0 and first_step_probs is not None:
            p = first_step_probs[tid].item()
        else:
            with torch.no_grad():
                p = torch.softmax(model(input_ids=cur).logits[0, -1], dim=-1)[tid].item()
        prob *= p
        cur = torch.cat([cur, torch.tensor([[tid]], device=device)], dim=1)
    return prob

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
    "target_label", "embedded_label", "comparator_label",
    "target_mass", "target_mass_median",
    "embedded_mass", "embedded_mass_median",
    "vocab_mass", "vocab_mass_median",
    "target_minus_vocab",
    "target_minus_embedded",
    "embedded_minus_vocab",
    "roi_count", "checkpoint",
]


def _build_np_token_lists(nouns: List[str], tokenizer) -> List[Tuple[str, List[int]]]:
    """Return [(noun, token_ids_for_'the noun'), ...] for every noun."""
    return [
        (noun, tokenizer.encode(f" the {noun}", add_special_tokens=False))
        for noun in nouns
    ]


def extract_orc_embedded_np(sentence: str, noun_candidates: Set[str]) -> Tuple[List[str], Optional[str]]:
    """Extract the embedded subject NP (e.g. 'the boy' in 'the girl that the boy likes').

    Looks for the NP immediately following the relativizer (that/who).
    Returns (np_words, head_noun) or ([], None) if not found.
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


def compute_one_checkpoint(
    ckpt_dir:      Path,
    tokenizer_dir: Path,
    sentences:     List[str],
    batch_size:    int = 32,
) -> dict:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    model     = AutoModelForCausalLM.from_pretrained(ckpt_dir, local_files_only=True).to(device)
    model.eval()

    roi_verb_set: Set[str] = set(VERBS_ORC)
    vocab_np_list = _build_np_token_lists(VOCAB_NOUNS, tokenizer)
    noun_candidates_set = {
        normalize_word(w) for w in NOUNS_ORC if normalize_word(w) not in {"the", "a", "an"}
    }

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
    e_vals: List[float] = []
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
            logits_batch = model(input_ids=x_batch)

        for i, (tok, roi_idx, s) in enumerate(batch):
            sp = torch.softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)
            x  = torch.tensor(contexts[i], device=device).unsqueeze(0)

            def geomean_prob(ids: List[int]) -> float:
                prob = _np_chain_prob(model, x, ids, device, first_step_probs=sp)
                return prob ** (1.0 / len(ids))

            # target: the moved NP (e.g. "the girl")
            np_words, head = extract_orc_moved_np(s, set(NOUNS_ORC))
            if not np_words or head is None:
                continue
            flat_ids = [
                t for w in np_words
                for t in tokenizer.encode(" " + w, add_special_tokens=False)
            ]
            target = geomean_prob(flat_ids)

            # embedded NP: the subject inside the relative clause (e.g. "the boy")
            emb_words, emb_head = extract_orc_embedded_np(s, noun_candidates_set)
            if emb_words and emb_head is not None:
                emb_flat_ids = [
                    t for w in emb_words
                    for t in tokenizer.encode(" " + w, add_special_tokens=False)
                ]
                embedded = geomean_prob(emb_flat_ids)
                e_vals.append(embedded)
            else:
                emb_head = None
                embedded = float("nan")

            # comparator: vocab NPs, excluding both the moved NP and the embedded NP
            moved_head_norm = normalize_word(head)
            exclude_heads = {moved_head_norm}
            if emb_head is not None:
                exclude_heads.add(normalize_word(emb_head))
            comparator_probs = [
                geomean_prob(ids)
                for noun, ids in vocab_np_list
                if normalize_word(noun) not in exclude_heads and ids
            ]
            if not comparator_probs:
                continue

            t_vals.append(target)
            v_vals.append(statistics.mean(comparator_probs))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    name = ckpt_dir.name
    step = int(name.split("-")[-1]) if "-" in name else 0
    n = len(t_vals)
    if n == 0:
        t_avg = t_med = e_avg = e_med = v_avg = v_med = 0.0
    else:
        t_avg, t_med = statistics.mean(t_vals), statistics.median(t_vals)
        v_avg, v_med = statistics.mean(v_vals), statistics.median(v_vals)
        e_avg = statistics.mean(e_vals) if e_vals else float("nan")
        e_med = statistics.median(e_vals) if e_vals else float("nan")

    print(
        f"  step {step}: moved_np={t_avg:.6f} (med={t_med:.6f}), "
        f"embedded_np={e_avg:.6f} (med={e_med:.6f}), "
        f"vocab_np={v_avg:.6f} (med={v_med:.6f}), n={n}"
    )

    return {
        "step": step,
        "structure": "orc",
        "target_label": "moved_np_geomean_prob",
        "embedded_label": "embedded_np_geomean_prob",
        "comparator_label": "vocab_np_geomean_prob",
        "target_mass": round(t_avg, 6),
        "target_mass_median": round(t_med, 6),
        "embedded_mass": round(e_avg, 6) if e_avg == e_avg else float("nan"),
        "embedded_mass_median": round(e_med, 6) if e_med == e_med else float("nan"),
        "vocab_mass": round(v_avg, 6),
        "vocab_mass_median": round(v_med, 6),
        "target_minus_vocab": round(t_avg - v_avg, 6),
        "target_minus_embedded": round(t_avg - e_avg, 6) if e_avg == e_avg else float("nan"),
        "embedded_minus_vocab": round(e_avg - v_avg, 6) if e_avg == e_avg else float("nan"),
        "roi_count": n,
        "checkpoint": str(ckpt_dir),
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
    p.add_argument("--tokenizer-dir", type=Path, default=None,
                   help="HF tokenizer directory. Defaults to --checkpoint or --checkpoint-dir.")
    p.add_argument("--sentences-file", type=Path, required=True,
                   help="One ORC sentence per line.")
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--result-name", type=Path, default=Path("results/eval_gap_np_orc.csv"))
    args = p.parse_args()

    sentences = read_nonempty_lines(args.sentences_file)

    tok_path = args.tokenizer_dir or args.checkpoint or args.checkpoint_dir

    if args.checkpoint is not None:
        ckpt = args.checkpoint
        rows = [compute_one_checkpoint(ckpt, tok_path, sentences, args.batch_size)]
        write_rows(rows, args.result_name)
        print(f"Results saved to: {args.result_name}")
        return

    ckpts = sorted(
        [d for d in args.checkpoint_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-* directories found in {args.checkpoint_dir}")

    items = [
        (ckpt, tok_path, sentences, args.batch_size)
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