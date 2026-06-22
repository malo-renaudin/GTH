#!/usr/bin/env python3
"""Compute token-level perplexity for a small sample of lines using a lit checkpoint."""
import argparse
from pathlib import Path
import math
import random

import torch
import torch.nn.functional as F

from litgpt import Tokenizer
from utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--max-sents", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-seq-length", type=int, default=0)
    return p.parse_args()


def compute_ppl(checkpoint: Path, tokenizer_dir: Path, data_file: Path, max_sents: int, device: str, max_seq_length: int):
    device = torch.device(device)
    tokenizer = Tokenizer(tokenizer_dir)
    model = load_checkpoint(checkpoint, device=device, max_seq_length=max_seq_length)

    lines = [l.strip() for l in data_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return float("nan")
    random.seed(0)
    sample = lines if max_sents <= 0 or len(lines) <= max_sents else random.sample(lines, max_sents)

    total_nll = 0.0
    total_toks = 0

    for sent in sample:
        ids = tokenizer.encode(sent, bos=True, eos=True).tolist()
        if len(ids) < 2:
            continue
        if max_seq_length > 0 and len(ids) > max_seq_length:
            # truncate to last max_seq_length tokens (preserve suffix as typical LM context)
            ids = ids[-max_seq_length:]

        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)  # [1, L, V]
        # predict ids[1:] using logits[:, :-1, :]
        lps = logits[0, :-1, :]
        targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
        loss = F.cross_entropy(lps, targets, reduction="sum")
        total_nll += loss.item()
        total_toks += targets.numel()

    if total_toks == 0:
        return float("nan")
    avg_nll = total_nll / total_toks
    ppl = math.exp(avg_nll)
    return ppl


def main():
    args = parse_args()
    ppl = compute_ppl(args.checkpoint, args.tokenizer_dir, args.data_file, args.max_sents, args.device, args.max_seq_length)
    print(ppl)


if __name__ == "__main__":
    main()
