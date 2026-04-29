#!/usr/bin/env python3
"""Evaluate noun-vs-verb next-token preference after verb ROIs."""

import argparse
import csv
import hashlib
import json
import pickle
import random
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

from litgpt import Tokenizer
from litgpt.config import Config
from litgpt.model import GPT


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
    noun_vs_verb_log_ratio: float


CacheValue = Tuple[List[int], List[Tuple[int, int]], List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate noun-vs-verb next-token mass after verb regions of interest."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--checkpoint", type=Path)
    mode_group.add_argument("--checkpoint-dir", type=Path)

    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--eval-orc-file", type=Path)
    parser.add_argument("--eval-wh-file", type=Path)

    parser.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    parser.add_argument("--vocab-file", type=Path, default=Path("data/english_data/vocab.txt"))
    parser.add_argument("--vocab-orc-file", type=Path, default=Path("data/english_data/orc.txt"))
    parser.add_argument("--vocab-wh-file", type=Path, default=Path("data/english_data/wh.txt"))

    parser.add_argument("--output-file", type=Path, default=Path("results/eval_roi_scores.json"))
    parser.add_argument("--csv-output-file", type=Path, default=Path("results/eval_checkpoint_scan.csv"))
    parser.add_argument("--cache-file", type=Path, default=Path("results/eval_tokenization_cache.pkl"))
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-seq-length", type=int, default=0)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_checkpoint_file(path: Path) -> Path:
    if path.is_file():
        return path
    file_path = path / "lit_model.pth"
    if file_path.is_file():
        return file_path
    raise FileNotFoundError(f"Could not find lit_model.pth in: {path}")


def load_model(checkpoint: Path, device: torch.device, max_seq_length: int) -> GPT:
    checkpoint_file = resolve_checkpoint_file(checkpoint)
    model = GPT(Config.from_checkpoint(checkpoint_file.parent))
    raw = torch.load(checkpoint_file, map_location="cpu")
    state = raw.get("model") if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else raw
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_file}")
    model.load_state_dict(state, strict=True)
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    model.to(device).eval()
    return model


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_words(path: Path) -> List[str]:
    words: List[str] = []
    strip_table = str.maketrans("", "", string.punctuation.replace("'", ""))
    for line in read_nonempty_lines(path):
        for token in line.split():
            cleaned = token.lower().translate(strip_table).strip()
            if cleaned:
                words.append(cleaned)
    return words


def build_lexicons(vocab_source: Path) -> Tuple[Set[str], Set[str], str]:
    words = sorted(set(extract_words(vocab_source)))
    try:
        import nltk

        try:
            nltk.pos_tag(["test"])
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)

        tagged = nltk.pos_tag(words)
        nouns = {w.lower() for w, tag in tagged if tag.startswith("NN")}
        verbs = {w.lower() for w, tag in tagged if tag.startswith("VB")}
        backend = "nltk"
    except Exception:
        verb_priors = {
            "be",
            "am",
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "have",
            "has",
            "had",
            "can",
            "could",
            "will",
            "would",
            "shall",
            "should",
            "may",
            "might",
            "must",
            "visit",
            "see",
            "like",
            "know",
            "chase",
            "admire",
            "help",
        }
        nouns = {
            w.lower()
            for w in words
            if w.lower().endswith(("tion", "ment", "ness", "ity", "er", "or", "ist", "ship", "age"))
        }
        verbs = {
            w.lower()
            for w in words
            if w.lower() in verb_priors or w.lower().endswith("ed") or w.lower().endswith("ing")
        }
        backend = "heuristic"

    if not nouns or not verbs:
        raise ValueError("Failed to infer noun/verb lexicons from vocabulary.")
    return nouns, verbs, backend


def build_candidate_token_ids(tokenizer: Tokenizer, words: Iterable[str]) -> List[int]:
    ids = {
        toks[0]
        for w in words
        for toks in [tokenizer.encode(" " + w, bos=False, eos=False).tolist()]
        if len(toks) == 1
    }
    if not ids:
        raise ValueError("No single-token candidates extracted from vocabulary.")
    return sorted(ids)


def tokenize_sentence_with_spans(tokenizer: Tokenizer, sentence: str) -> CacheValue:
    words = sentence.strip().split()
    token_ids: List[int] = []
    spans: List[Tuple[int, int]] = []
    aligned: List[str] = []
    for i, word in enumerate(words):
        piece = word if i == 0 else " " + word
        ids = tokenizer.encode(piece, bos=False, eos=False).tolist()
        if not ids:
            continue
        start = len(token_ids)
        token_ids.extend(ids)
        spans.append((start, len(token_ids)))
        aligned.append(word)
    return token_ids, spans, aligned


def cache_key(sentence: str) -> str:
    return hashlib.sha1(sentence.encode("utf-8")).hexdigest()


def load_disk_cache(path: Path) -> Dict[str, CacheValue]:
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def save_disk_cache(path: Path, cache: Dict[str, CacheValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(cache, f)


def evaluate_sentences(
    model: GPT,
    tokenizer: Tokenizer,
    sentences: Sequence[str],
    verbs_lexicon: Set[str],
    noun_token_ids: torch.Tensor,
    verb_token_ids: torch.Tensor,
    device: torch.device,
    cache_file: Path,
) -> Tuple[List[RoiScore], Dict[str, int]]:
    disk_cache = load_disk_cache(cache_file)
    stats = {
        "processed_sentences": 0,
        "total_sentences": len(sentences),
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_no_roi": 0,
        "roi_count": 0,
    }
    results: List[RoiScore] = []

    for sentence in sentences:
        key = cache_key(sentence)
        token_ids, spans, words = disk_cache.get(key, tokenize_sentence_with_spans(tokenizer, sentence))
        disk_cache.setdefault(key, (token_ids, spans, words))

        if len(token_ids) < 2 or len(spans) < 2:
            stats["skipped_too_short"] += 1
            continue
        if len(token_ids) > model.max_seq_length:
            stats["skipped_too_long"] += 1
            continue

        roi_indices = [i for i, w in enumerate(words[:-1]) if w.lower() in verbs_lexicon and i + 1 < len(spans)]
        if not roi_indices:
            stats["skipped_no_roi"] += 1
            continue

        with torch.no_grad():
            logits = model(torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0))[0]

        for i in roi_indices:
            pred_idx = spans[i][1] - 1
            next_tok_start = spans[i + 1][0]
            if pred_idx < 0 or pred_idx >= logits.shape[0] or next_tok_start >= len(token_ids):
                continue

            distribution = logits[pred_idx]
            probs = torch.softmax(distribution.float(), dim=-1)
            noun_mass = probs.index_select(0, noun_token_ids).sum().item()
            verb_mass = probs.index_select(0, verb_token_ids).sum().item()

            noun_logsumexp = torch.logsumexp(distribution.index_select(0, noun_token_ids), dim=0)
            verb_logsumexp = torch.logsumexp(distribution.index_select(0, verb_token_ids), dim=0)

            actual_next_token_id = token_ids[next_tok_start]
            results.append(
                RoiScore(
                    sentence=sentence,
                    verb_word=words[i],
                    verb_word_index=i,
                    predicted_token_index=pred_idx,
                    actual_next_word=words[i + 1],
                    actual_next_token_id=actual_next_token_id,
                    actual_next_token_prob=probs[actual_next_token_id].item(),
                    noun_mass=noun_mass,
                    verb_mass=verb_mass,
                    noun_minus_verb=noun_mass - verb_mass,
                    noun_vs_verb_log_ratio=(noun_logsumexp - verb_logsumexp).item(),
                )
            )

        stats["processed_sentences"] += 1

    stats["roi_count"] = len(results)
    save_disk_cache(cache_file, disk_cache)
    return results, stats


def summarize(results: Sequence[RoiScore]) -> Dict[str, float]:
    if not results:
        return {
            "mean_noun_mass": 0.0,
            "mean_verb_mass": 0.0,
            "mean_noun_minus_verb": 0.0,
            "mean_noun_vs_verb_log_ratio": 0.0,
            "roi_count": 0,
            "noun_mass_gt_verb_mass_ratio": 0.0,
        }

    n = len(results)
    noun_sum = sum(r.noun_mass for r in results)
    verb_sum = sum(r.verb_mass for r in results)
    delta_sum = sum(r.noun_minus_verb for r in results)
    log_ratio_sum = sum(r.noun_vs_verb_log_ratio for r in results)
    wins = sum(1 for r in results if r.noun_mass > r.verb_mass)
    return {
        "mean_noun_mass": noun_sum / n,
        "mean_verb_mass": verb_sum / n,
        "mean_noun_minus_verb": delta_sum / n,
        "mean_noun_vs_verb_log_ratio": log_ratio_sum / n,
        "roi_count": n,
        "noun_mass_gt_verb_mass_ratio": wins / n,
    }


def collect_step_checkpoints(checkpoint_dir: Path) -> List[Tuple[int, Path]]:
    entries: List[Tuple[int, Path]] = []
    for ckpt in checkpoint_dir.glob("step-*/lit_model.pth"):
        match = re.match(r"step-(\d+)", ckpt.parent.name)
        if not match:
            continue
        entries.append((int(match.group(1)), ckpt))
    entries.sort(key=lambda x: x[0])
    return entries


def maybe_sample(sentences: List[str], sample_size: int) -> List[str]:
    if 0 < sample_size < len(sentences):
        return random.sample(sentences, sample_size)
    return sentences


def evaluate_single_checkpoint(args: argparse.Namespace) -> None:
    if args.eval_file is None:
        raise ValueError("--eval-file is required when using --checkpoint.")

    device = choose_device(args.device)
    model = load_model(args.checkpoint, device=device, max_seq_length=args.max_seq_length)
    tokenizer = Tokenizer(args.tokenizer_dir)

    nouns, verbs, lexicon_backend = build_lexicons(args.vocab_file)
    noun_token_ids = torch.tensor(build_candidate_token_ids(tokenizer, nouns), dtype=torch.long, device=device)
    verb_token_ids = torch.tensor(build_candidate_token_ids(tokenizer, verbs), dtype=torch.long, device=device)

    sentences = maybe_sample(read_nonempty_lines(args.eval_file), args.sample_size)
    roi_scores, stats = evaluate_sentences(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        verbs_lexicon=verbs,
        noun_token_ids=noun_token_ids,
        verb_token_ids=verb_token_ids,
        device=device,
        cache_file=args.cache_file,
    )
    summary = summarize(roi_scores)

    output = {
        "args": {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device_resolved": str(device),
        },
        "model": {
            "checkpoint": str(resolve_checkpoint_file(args.checkpoint)),
            "max_seq_length": model.max_seq_length,
            "vocab_size": model.config.vocab_size,
            "padded_vocab_size": model.config.padded_vocab_size,
        },
        "lexicons": {
            "backend": lexicon_backend,
            "nouns_count": len(nouns),
            "verbs_count": len(verbs),
            "noun_token_ids_count": int(noun_token_ids.numel()),
            "verb_token_ids_count": int(verb_token_ids.numel()),
        },
        "stats": stats,
        "summary": summary,
        "roi_scores": [asdict(score) for score in roi_scores],
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved evaluation output to {args.output_file}")
    print(json.dumps({"stats": stats, "summary": summary}, indent=2))


def evaluate_checkpoint_dir(args: argparse.Namespace) -> None:
    if args.eval_orc_file is None or args.eval_wh_file is None:
        raise ValueError("--eval-orc-file and --eval-wh-file are required with --checkpoint-dir.")

    device = choose_device(args.device)
    tokenizer = Tokenizer(args.tokenizer_dir)

    datasets = {
        "orc": {
            "eval_file": args.eval_orc_file,
            "vocab_file": args.vocab_orc_file,
        },
        "wh": {
            "eval_file": args.eval_wh_file,
            "vocab_file": args.vocab_wh_file,
        },
    }

    dataset_specs = {}
    for name, spec in datasets.items():
        nouns, verbs, _ = build_lexicons(spec["vocab_file"])
        dataset_specs[name] = {
            "sentences": maybe_sample(read_nonempty_lines(spec["eval_file"]), args.sample_size),
            "verbs": verbs,
            "noun_token_ids": torch.tensor(
                build_candidate_token_ids(tokenizer, nouns), dtype=torch.long, device=device
            ),
            "verb_token_ids": torch.tensor(
                build_candidate_token_ids(tokenizer, verbs), dtype=torch.long, device=device
            ),
        }

    checkpoints = collect_step_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        raise ValueError(f"No step checkpoints found in {args.checkpoint_dir}.")

    rows = []
    for step, checkpoint_file in checkpoints:
        model = load_model(checkpoint_file, device=device, max_seq_length=args.max_seq_length)
        for dataset_name, spec in dataset_specs.items():
            _, stats = evaluate_sentences(
                model=model,
                tokenizer=tokenizer,
                sentences=spec["sentences"],
                verbs_lexicon=spec["verbs"],
                noun_token_ids=spec["noun_token_ids"],
                verb_token_ids=spec["verb_token_ids"],
                device=device,
                cache_file=args.cache_file,
            )
            summary = summarize(_)
            rows.append(
                {
                    "step": step,
                    "test_dataset": dataset_name,
                    "noun_mass": summary["mean_noun_mass"],
                    "verb_mass": summary["mean_verb_mass"],
                    "roi_count": summary["roi_count"],
                    "checkpoint": str(checkpoint_file),
                    "processed_sentences": stats["processed_sentences"],
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.csv_output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "test_dataset",
                "noun_mass",
                "verb_mass",
                "roi_count",
                "processed_sentences",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved checkpoint scan CSV to {args.csv_output_file}")
    print(f"Rows written: {len(rows)}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if args.checkpoint_dir is not None:
        evaluate_checkpoint_dir(args)
    else:
        evaluate_single_checkpoint(args)


if __name__ == "__main__":
    main()
