#!/usr/bin/env python3
"""
Check that all vocabulary words in a generation script are single GPT2 tokens.
Results are reported by vocabulary category (variable name).

Usage
-----
python generate_simple_datasets/check_vocab_tokens.py generate_simple_datasets/filler_gap_orc.py
python generate_simple_datasets/check_vocab_tokens.py generate_simple_datasets/orc_transitivity.py --tokenizer gpt2
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

from transformers import AutoTokenizer


def extract_word(item: str) -> str:
    """Strip article prefix ('a ', 'an ', 'the ') to get the bare content word."""
    for prefix in ("the ", "an ", "a "):
        if item.lower().startswith(prefix):
            return item[len(prefix):]
    return item


def check_category(label: str, words: list, tokenizer) -> tuple:
    ok, bad = [], []
    for w in words:
        bare = extract_word(w)
        ids  = tokenizer.encode(" " + bare, add_special_tokens=False)
        if len(ids) == 1:
            ok.append(w)
        else:
            bad.append((w, [tokenizer.decode([i]) for i in ids]))

    status = "✓" if not bad else "✗"
    print(f"  [{status}] {label:<30s}  {len(ok)}/{len(words)} single-token")
    for w, tokens in bad:
        print(f"         FAIL  '{w}' → {tokens}")
    return ok, bad


def main():
    p = argparse.ArgumentParser(description="Check vocab words are single GPT2 tokens.")
    p.add_argument("script",      type=Path, help="Path to a generation script")
    p.add_argument("--tokenizer", type=str,  default="gpt2")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)

    # Load the generation script; patch argv so argparse uses its defaults,
    # and mock open() so file-writing at module level is silently ignored.
    sys.argv = [str(args.script)]
    spec = importlib.util.spec_from_file_location("_vocab_mod", args.script)
    mod  = importlib.util.module_from_spec(spec)
    with patch("builtins.open", mock_open()):
        spec.loader.exec_module(mod)

    print(f"\nTokenizer : {args.tokenizer}")
    print(f"Script    : {args.script.name}\n")

    total_ok = total_bad = 0

    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        val = getattr(mod, name)

        if isinstance(val, list) and val and isinstance(val[0], str):
            ok, bad = check_category(name, val, tokenizer)
            total_ok += len(ok)
            total_bad += len(bad)

        elif isinstance(val, dict) and val:
            first_v = next(iter(val.values()))
            if isinstance(first_v, list) and first_v and isinstance(first_v[0], str):
                # dict of lists: check keys (verbs) and each value list (objects)
                ok, bad = check_category(f"{name} [keys]", list(val.keys()), tokenizer)
                total_ok += len(ok); total_bad += len(bad)
                for k, items in val.items():
                    ok, bad = check_category(f"{name}['{k}']", items, tokenizer)
                    total_ok += len(ok); total_bad += len(bad)

    print(f"\n{'─'*50}")
    print(f"  Total : {total_ok} single-token,  {total_bad} multi-token")
    if total_bad == 0:
        print("  All vocabulary words are single tokens. ✓")


if __name__ == "__main__":
    main()
