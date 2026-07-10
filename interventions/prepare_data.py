"""
Turn a filler/gap quadruplet CSV into the anchor-based format expected by das.py.

Each example is a direct (base row, source row) pair from the same quadruplet_id.

  base_row   = row with filler == --base-filler,  gap == --base-gap
  source_row = row with filler == --source-filler, gap == --source-gap
  yb         = first word of base_row.post_gap_text
  ys         = first word of source_row.post_gap_text

When base and source share the same gap value, yb == ys and ODDS reduces to
log(p(ys|intervened)/p(ys|base)). For cross-gap pairings yb != ys and the
full ODDS formula applies.

Anchor words: the last len(--position-words) whitespace tokens of pre_gap_text
(default: relativizer, "the", noun, embedded verb). Stored as literal strings;
das.py resolves token indices at runtime.

Output columns:
    quadruplet_id, position_label,
    base_text, source_text, yb, ys,
    base_anchor, source_anchor

Output: --output-dir/train.csv and --output-dir/eval.csv, split by quadruplet_id.
"""

import argparse
import os
import re

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--base-filler", type=int, default=0)
    p.add_argument("--base-gap", type=int, default=1)
    p.add_argument("--source-filler", type=int, default=1)
    p.add_argument("--source-gap", type=int, default=1)
    p.add_argument("--position-words", nargs="+",
                    default=["relativizer", "the", "noun", "verb"],
                    help="labels for the last N words of pre_gap_text, left to right")
    p.add_argument("--outer-verb", action="store_true",
                    help="Use one anchor labeled 'outer_verb' at the last token of pre_gap_text")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def first_word(text):
    return text.strip().split()[0]


def last_n_words(text, n):
    """Return the last n whitespace-separated tokens of `text` as a list."""
    return re.findall(r"\S+", text)[-n:]


def build_rows(df, args):
    rows = []
    position_words = ["outer_verb"] if args.outer_verb else args.position_words
    for qid, group in df.groupby("quadruplet_id"):
        def find(filler, gap):
            match = group[(group["filler"] == filler) & (group["gap"] == gap)]
            return match.iloc[0] if len(match) else None

        base_row = find(args.base_filler, args.base_gap)
        source_row = find(args.source_filler, args.source_gap)
        if base_row is None or source_row is None:
            continue

        base_text, source_text = base_row["sentence"], source_row["sentence"]
        yb = first_word(base_row["post_gap_text"])
        ys = first_word(source_row["post_gap_text"])

        n = len(position_words)
        base_anchors = last_n_words(base_row["pre_gap_text"], n)
        source_anchors = last_n_words(source_row["pre_gap_text"], n)

        for label, b_anchor, s_anchor in zip(position_words, base_anchors, source_anchors):
            rows.append({
                "quadruplet_id": qid,
                "position_label": label,
                "base_text": base_text,
                "source_text": source_text,
                "yb": yb,
                "ys": ys,
                "base_anchor": b_anchor,
                "source_anchor": s_anchor,
            })
    return pd.DataFrame(rows)


def split_by_quadruplet(df, train_frac, seed):
    qids = df["quadruplet_id"].drop_duplicates().sample(frac=1, random_state=seed)
    n_train = int(len(qids) * train_frac)
    train_ids, eval_ids = set(qids[:n_train]), set(qids[n_train:])
    return df[df["quadruplet_id"].isin(train_ids)], df[df["quadruplet_id"].isin(eval_ids)]


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    out = build_rows(df, args)

    train_df, eval_df = split_by_quadruplet(out, args.train_frac, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    eval_df.to_csv(os.path.join(args.output_dir, "eval.csv"), index=False)
    print(f"train: {len(train_df)} rows, eval: {len(eval_df)} rows")


if __name__ == "__main__":
    main()