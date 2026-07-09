"""
Turn a filler/gap quadruplet CSV into the (base_text, source_text, yb, ys,
base_position, source_position) format expected by the DAS training script.

Per quadruplet_id:
  base_row   = row with filler == --base-filler,   gap == --base-gap
  source_row = row with filler == --source-filler, gap == --source-gap
  yb_row     = row with filler == --base-filler,   gap == 1 - --base-gap
               (the grammatical continuation for the base's filler value)
  ys_row     = source_row itself (its own post_gap_text is the correct
               continuation for the source condition)

yb / ys = first word of yb_row / ys_row's post_gap_text.

Intervention positions: the last len(--position-words) whitespace-separated
words of pre_gap_text (default: relativizer, "the", noun, embedded verb).
Token positions are resolved separately for base_text and source_text with
--checkpoint's tokenizer (their lengths can differ), producing one output
row per (quadruplet_id, position word).

Output: --output-dir/train.csv and --output-dir/eval.csv, split by
quadruplet_id so all positions of a quadruplet stay on the same side.
"""

import argparse
import os
import re

import pandas as pd
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--checkpoint", required=True, help="tokenizer used to resolve positions")
    p.add_argument("--base-filler", type=int, default=0)
    p.add_argument("--base-gap", type=int, default=1)
    p.add_argument("--source-filler", type=int, default=1)
    p.add_argument("--source-gap", type=int, default=1)
    p.add_argument("--position-words", nargs="+",
                    default=["relativizer", "the", "noun", "verb"],
                    help="labels for the last N words of pre_gap_text, left to right")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def first_word(text):
    return text.strip().split()[0]


def last_n_word_spans(text, n):
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    return spans[-n:]


def token_position(tokenizer, text, char_start, char_end):
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    idx = None
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if s < char_end and e > char_start:
            idx = i
    return idx


def build_rows(df, tokenizer, args):
    rows = []
    for qid, group in df.groupby("quadruplet_id"):
        def find(filler, gap):
            match = group[(group["filler"] == filler) & (group["gap"] == gap)]
            return match.iloc[0] if len(match) else None

        base_row = find(args.base_filler, args.base_gap)
        source_row = find(args.source_filler, args.source_gap)
        yb_row = find(args.base_filler, 1 - args.base_gap)
        if base_row is None or source_row is None or yb_row is None:
            continue

        base_text, source_text = base_row["sentence"], source_row["sentence"]
        yb, ys = first_word(yb_row["post_gap_text"]), first_word(source_row["post_gap_text"])

        n = len(args.position_words)
        base_spans = last_n_word_spans(base_row["pre_gap_text"], n)
        source_spans = last_n_word_spans(source_row["pre_gap_text"], n)

        for label, (bs, be), (ss, se) in zip(args.position_words, base_spans, source_spans):
            base_pos = token_position(tokenizer, base_text, bs, be)
            source_pos = token_position(tokenizer, source_text, ss, se)
            if base_pos is None or source_pos is None:
                continue
            rows.append({
                "quadruplet_id": qid,
                "position_label": label,
                "base_text": base_text,
                "source_text": source_text,
                "yb": yb,
                "ys": ys,
                "base_position": base_pos,
                "source_position": source_pos,
            })
    return pd.DataFrame(rows)


def split_by_quadruplet(df, train_frac, seed):
    qids = df["quadruplet_id"].drop_duplicates().sample(frac=1, random_state=seed)
    n_train = int(len(qids) * train_frac)
    train_ids, eval_ids = set(qids[:n_train]), set(qids[n_train:])
    return df[df["quadruplet_id"].isin(train_ids)], df[df["quadruplet_id"].isin(eval_ids)]


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    df = pd.read_csv(args.input_csv)
    out = build_rows(df, tokenizer, args)

    train_df, eval_df = split_by_quadruplet(out, args.train_frac, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    eval_df.to_csv(os.path.join(args.output_dir, "eval.csv"), index=False)
    print(f"train: {len(train_df)} rows, eval: {len(eval_df)} rows")


if __name__ == "__main__":
    main()