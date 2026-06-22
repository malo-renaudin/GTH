#!/usr/bin/env python3
"""Evaluate noun-vs-verb next-token preference after verb ROIs."""
# one function to load checkpoints
# one function that takes a vocabulary list and a tokenizer as input, and maps one to the other
# run the latter on lists on verbs and nouns that i will hardcode
# one functions that takes a sentence, and tokenizes it
# one function that takes the tokenized sentence, and runs the model on it up onto the verb for instance
# one function that compares the probability mass of this model on different dictionary of voc-token
# one function to run all this on several checkpoints in parallel, and gather results in a csv

import argparse
import csv
import hashlib
import json
import pickle
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

from litgpt import Tokenizer
from litgpt.config import Config
from litgpt.model import GPT


CacheValue = Tuple[List[int], List[Tuple[int, int]], List[str]]


@dataclass
class RoiScore:
    sentence: str
    verb_word: str
    verb_word_index: int
    predicted_token_index: int
    actual_next_word: str
    actual_next_token_id: int
    actual_next_token_prob: float
    noun_mass: float
    verb_mass: float
    noun_minus_verb: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI noun-vs-verb mass evaluation.")
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--checkpoint", type=Path)
    m.add_argument("--checkpoint-dir", type=Path)

    p.add_argument("--eval-file", type=Path)
    p.add_argument("--eval-orc-file", type=Path)
    p.add_argument("--eval-wh-file", type=Path)

    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--vocab-file", type=Path, default=Path("data/english_data/vocab.txt"))
    p.add_argument("--vocab-orc-file", type=Path, default=Path("data/english_data/orc.txt"))
    p.add_argument("--vocab-wh-file", type=Path, default=Path("data/english_data/wh.txt"))

    p.add_argument("--output-file", type=Path, default=Path("results/eval_roi_scores.json"))
    p.add_argument("--csv-output-file", type=Path, default=Path("results/eval_checkpoint_scan.csv"))
    p.add_argument("--cache-file", type=Path, default=Path("results/eval_tokenization_cache.pkl"))
    p.add_argument("--sample-size", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-seq-length", type=int, default=0)
    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "lit_model.pth"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find lit_model.pth in: {path}")


# one function to load checkpoints
def load_checkpoint(checkpoint: Path, device: torch.device, max_seq_length: int) -> GPT:
    ckpt_file = resolve_checkpoint_file(checkpoint)
    model = GPT(Config.from_checkpoint(ckpt_file.parent))
    raw = torch.load(ckpt_file, map_location="cpu")
    state = raw.get("model") if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else raw
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_file}")
    model.load_state_dict(state, strict=True)
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    model.to(device).eval()
    return model


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# one function that takes a vocabulary list and a tokenizer as input, and maps one to the other
def vocab_words_to_token_ids(tokenizer: Tokenizer, words: Iterable[str]) -> List[int]:
    ids = {
        toks[0]
        for w in words
        for toks in [tokenizer.encode(" " + w, bos=False, eos=False).tolist()]
        if len(toks) == 1
    }
    if not ids:
        raise ValueError("No single-token ids extracted from vocabulary words.")
    return sorted(ids)


# run the latter on lists on verbs and nouns that i will hardcode
def hardcoded_lexicons(vocab_file: Path) -> Tuple[Set[str], Set[str]]:
    vocab = {w.lower() for w in read_nonempty_lines(vocab_file)}
    verb_seeds = {
        "be", "am", "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must", "visit", "see",
        "like", "know", "chase", "admire", "help", "ask", "say", "think", "want", "find", "tell",
    }
    nouns = {w for w in vocab if w.endswith(("tion", "ment", "ness", "ity", "er", "or", "ist", "ship", "age"))}
    verbs = {w for w in vocab if w in verb_seeds or w.endswith("ed") or w.endswith("ing")}
    if not nouns:
        nouns = {w for w in vocab if w not in verb_seeds and not w.endswith(("ed", "ing"))}
    if not verbs:
        verbs = set(verb_seeds & vocab)
    if not nouns or not verbs:
        raise ValueError(f"Could not derive noun/verb lexicons from {vocab_file}")
    return nouns, verbs


# one functions that takes a sentence, and tokenizes it
def tokenize_with_word_spans(tokenizer: Tokenizer, sentence: str) -> CacheValue:
    token_ids: List[int] = []
    spans: List[Tuple[int, int]] = []
    words: List[str] = []
    for i, word in enumerate(sentence.split()):
        piece = word if i == 0 else " " + word
        ids = tokenizer.encode(piece, bos=False, eos=False).tolist()
        if not ids:
            continue
        start = len(token_ids)
        token_ids.extend(ids)
        spans.append((start, len(token_ids)))
        words.append(word)
    return token_ids, spans, words


def cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> Dict[str, CacheValue]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def save_cache(path: Path, cache: Dict[str, CacheValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(cache, f)


# one function that takes the tokenized sentence, and runs the model on it up onto the verb for instance
# one function that compares the probability mass of this model on different dictionary of voc-token
def score_sentence_rois(
    model: GPT,
    token_ids: List[int],
    spans: List[Tuple[int, int]],
    words: List[str],
    verbs_lexicon: Set[str],
    noun_ids: torch.Tensor,
    verb_ids: torch.Tensor,
    device: torch.device,
) -> List[RoiScore]:
    if len(token_ids) < 2 or len(token_ids) > model.max_seq_length:
        return []

    roi_word_idx = [i for i, w in enumerate(words[:-1]) if w.lower() in verbs_lexicon and i + 1 < len(spans)]
    if not roi_word_idx:
        return []

    with torch.no_grad():
        logits = model(torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0))[0]

    out: List[RoiScore] = []
    for i in roi_word_idx:
        pred_idx = spans[i][1] - 1
        next_tok_start = spans[i + 1][0]
        if pred_idx < 0 or pred_idx >= logits.shape[0] or next_tok_start >= len(token_ids):
            continue
        dist = logits[pred_idx]
        probs = torch.softmax(dist.float(), dim=-1)
        noun_mass = probs.index_select(0, noun_ids).sum().item()
        verb_mass = probs.index_select(0, verb_ids).sum().item()
        out.append(
            RoiScore(
                sentence=" ".join(words),
                verb_word=words[i],
                verb_word_index=i,
                predicted_token_index=pred_idx,
                actual_next_word=words[i + 1],
                actual_next_token_id=token_ids[next_tok_start],
                actual_next_token_prob=probs[token_ids[next_tok_start]].item(),
                noun_mass=noun_mass,
                verb_mass=verb_mass,
                noun_minus_verb=noun_mass - verb_mass,
            )
        )
    return out


def evaluate_dataset(
    model: GPT,
    tokenizer: Tokenizer,
    sentences: Sequence[str],
    verbs_lexicon: Set[str],
    noun_ids: torch.Tensor,
    verb_ids: torch.Tensor,
    device: torch.device,
    cache_file: Path,
) -> Tuple[List[RoiScore], Dict[str, int]]:
    cache = load_cache(cache_file)
    stats = {
        "total_sentences": len(sentences),
        "processed_sentences": 0,
        "skipped_too_short_or_long": 0,
        "skipped_no_roi": 0,
        "roi_count": 0,
    }
    all_scores: List[RoiScore] = []

    for sent in sentences:
        key = cache_key(sent)
        token_ids, spans, words = cache.get(key, tokenize_with_word_spans(tokenizer, sent))
        cache.setdefault(key, (token_ids, spans, words))

        if len(token_ids) < 2 or len(token_ids) > model.max_seq_length:
            stats["skipped_too_short_or_long"] += 1
            continue

        scores = score_sentence_rois(
            model=model,
            token_ids=token_ids,
            spans=spans,
            words=words,
            verbs_lexicon=verbs_lexicon,
            noun_ids=noun_ids,
            verb_ids=verb_ids,
            device=device,
        )
        if not scores:
            stats["skipped_no_roi"] += 1
            continue

        all_scores.extend(scores)
        stats["processed_sentences"] += 1

    stats["roi_count"] = len(all_scores)
    save_cache(cache_file, cache)
    return all_scores, stats


def summarize(scores: Sequence[RoiScore]) -> Dict[str, float]:
    if not scores:
        return {"mean_noun_mass": 0.0, "mean_verb_mass": 0.0, "mean_noun_minus_verb": 0.0, "roi_count": 0}
    n = len(scores)
    return {
        "mean_noun_mass": sum(s.noun_mass for s in scores) / n,
        "mean_verb_mass": sum(s.verb_mass for s in scores) / n,
        "mean_noun_minus_verb": sum(s.noun_minus_verb for s in scores) / n,
        "roi_count": n,
    }


def maybe_sample(lines: List[str], sample_size: int) -> List[str]:
    return random.sample(lines, sample_size) if 0 < sample_size < len(lines) else lines


def collect_step_checkpoints(root: Path) -> List[Tuple[int, Path]]:
    entries: List[Tuple[int, Path]] = []
    for p in root.glob("step-*/lit_model.pth"):
        m = re.match(r"step-(\d+)", p.parent.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda x: x[0])
    return entries


def run_single_checkpoint(args: argparse.Namespace) -> None:
    if not args.eval_file:
        raise ValueError("--eval-file is required with --checkpoint")

    device = choose_device(args.device)
    tokenizer = Tokenizer(args.tokenizer_dir)
    model = load_checkpoint(args.checkpoint, device, args.max_seq_length)

    nouns, verbs = hardcoded_lexicons(args.vocab_file)
    noun_ids = torch.tensor(vocab_words_to_token_ids(tokenizer, nouns), dtype=torch.long, device=device)
    verb_ids = torch.tensor(vocab_words_to_token_ids(tokenizer, verbs), dtype=torch.long, device=device)

    sentences = maybe_sample(read_nonempty_lines(args.eval_file), args.sample_size)
    scores, stats = evaluate_dataset(model, tokenizer, sentences, verbs, noun_ids, verb_ids, device, args.cache_file)

    payload = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": {"checkpoint": str(resolve_checkpoint_file(args.checkpoint)), "max_seq_length": model.max_seq_length},
        "stats": stats,
        "summary": summarize(scores),
        "roi_scores": [asdict(s) for s in scores],
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved evaluation output to {args.output_file}")


# one function to run all this on several checkpoints in parallel, and gather results in a csv
# (parallel-ready API, currently run sequentially for deterministic GPU memory usage)
def run_checkpoint_scan(args: argparse.Namespace) -> None:
    if not args.eval_orc_file or not args.eval_wh_file:
        raise ValueError("--eval-orc-file and --eval-wh-file are required with --checkpoint-dir")

    device = choose_device(args.device)
    tokenizer = Tokenizer(args.tokenizer_dir)
    checkpoints = collect_step_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise ValueError(f"No step-*/lit_model.pth under {args.checkpoint_dir}")

    datasets = {
        "orc": (args.eval_orc_file, args.vocab_orc_file),
        "wh": (args.eval_wh_file, args.vocab_wh_file),
    }

    specs = {}
    for name, (eval_file, vocab_file) in datasets.items():
        nouns, verbs = hardcoded_lexicons(vocab_file)
        specs[name] = {
            "sentences": maybe_sample(read_nonempty_lines(eval_file), args.sample_size),
            "verbs": verbs,
            "noun_ids": torch.tensor(vocab_words_to_token_ids(tokenizer, nouns), dtype=torch.long, device=device),
            "verb_ids": torch.tensor(vocab_words_to_token_ids(tokenizer, verbs), dtype=torch.long, device=device),
        }

    rows = []
    for step, ckpt_file in checkpoints:
        model = load_checkpoint(ckpt_file, device, args.max_seq_length)
        for test_name, spec in specs.items():
            scores, _ = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                sentences=spec["sentences"],
                verbs_lexicon=spec["verbs"],
                noun_ids=spec["noun_ids"],
                verb_ids=spec["verb_ids"],
                device=device,
                cache_file=args.cache_file,
            )
            s = summarize(scores)
            rows.append(
                {
                    "step": step,
                    "test_dataset": test_name,
                    "noun_mass": s["mean_noun_mass"],
                    "verb_mass": s["mean_verb_mass"],
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.csv_output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "test_dataset", "noun_mass", "verb_mass"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved checkpoint scan CSV to {args.csv_output_file} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if args.checkpoint_dir:
        run_checkpoint_scan(args)
    else:
        run_single_checkpoint(args)


if __name__ == "__main__":
    main()
