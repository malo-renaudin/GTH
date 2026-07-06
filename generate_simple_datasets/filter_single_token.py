#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from transformers import AutoTokenizer


def is_single_token(word, tokenizer):
    return len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1


def filter_transitivity(input_path, output_path, tokenizer):
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for row in rows:
        n1   = row["moved_np"].split()[-1]
        verb = row["verb"]
        n2   = row["pre_gap_text"].split()[-2]
        if all(is_single_token(w, tokenizer) for w in (n1, n2, verb)):
            kept.append(row)
    _write(output_path, kept, list(rows[0].keys()) if rows else [])
    print(f"orc_transitivity:         {len(rows):4d} -> {len(kept):4d} rows  ({len(rows)-len(kept)} dropped)")


def filter_semantic_distractors(input_path, output_path, tokenizer):
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for row in rows:
        n1   = row["moved_np"].split()[-1]
        verb = row["pre_gap_text"].split()[-1]
        n2   = row["pre_gap_text"].split()[-2]
        distractor_nouns = [obj.strip().split()[-1] for obj in row["plausible_objects"].split("|") if obj.strip()]
        words = [n1, n2, verb] + distractor_nouns
        if all(is_single_token(w, tokenizer) for w in words):
            kept.append(row)
    _write(output_path, kept, list(rows[0].keys()) if rows else [])
    print(f"orc_semantic_distractors: {len(rows):4d} -> {len(kept):4d} rows  ({len(rows)-len(kept)} dropped)")


def _write(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description="Filter ORC eval CSVs to single-token vocabulary words.")
    p.add_argument("--tokenizer",  default="gpt2")
    p.add_argument("--data-dir",   type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    out_dir = args.output_dir or args.data_dir
    print(f"Tokenizer: {args.tokenizer}")
    filter_transitivity(args.data_dir / "orc_transitivity.csv", out_dir / "orc_transitivity.csv", tokenizer)
    filter_semantic_distractors(args.data_dir / "orc_semantic_distractors.csv", out_dir / "orc_semantic_distractors.csv", tokenizer)
    print("Done.")


if __name__ == "__main__":
    main()
