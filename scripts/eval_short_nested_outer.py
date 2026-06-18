#!/usr/bin/env python3
"""Evaluate models on short_nested_outer_english.json.

Usage examples:
  python eval_short_nested_outer.py --data short_nested_outer_english.json --hf-model gpt2

The script expects each example to have fields: `input`, `target_scores` (map token->0/1), and `comment`.
It supports a HuggingFace model via `--hf-model` or a custom Python callable via `--model-module module:callable`.
If neither is provided, uses a random baseline.
"""
import argparse
import json
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import torch

from litgpt import Tokenizer
from utils import (
    load_checkpoint,
    np_chain_logprob,
    resolve_checkpoint_file,
    step_from_checkpoint,
)


def load_examples(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    return j.get('examples', [])


def compute_one_checkpoint(
    ckpt_file: Path,
    tokenizer_dir: Path,
    examples: List[Dict],
    max_seq_length: int = 0,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(tokenizer_dir)
    model = load_checkpoint(ckpt_file, device, max_seq_length)

    # Pre-tokenize
    items = []
    for i, ex in enumerate(examples):
        pre_ids = tokenizer.encode(ex['input'], bos=True, eos=False).tolist()
        cand_lists = [tokenizer.encode(' ' + c, bos=False, eos=False).tolist() for c in ex['target_scores'].keys()]
        if not pre_ids or not cand_lists:
            continue
        items.append((pre_ids, list(ex['target_scores'].keys()), cand_lists, ex.get('comment', 'NA')))

    total = 0
    correct = 0
    by_cat = defaultdict(lambda: {'total': 0, 'correct': 0})

    # evaluate sequentially (small dataset)
    for pre_ids, cand_keys, cand_lists, comment in items:
        ctx = torch.tensor([pre_ids], device=device)
        with torch.no_grad():
            logits = model(ctx)
        # defensive: ensure logits shape is as expected before indexing
        try:
            sp = torch.softmax(logits[0, -1, :], dim=-1)
        except Exception as e:
            print(f"Debug: skipping example due to logits indexing error; logits_shape={None if logits is None else tuple(logits.shape)}; pre_ids_len={len(pre_ids)}; error={e}", file=sys.stderr)
            continue

        best_c = None
        best_score = -float('inf')
        for key, ids in zip(cand_keys, cand_lists):
            # compute chain prob using first_step_probs optimization
            prob = np_chain_logprob(model, ctx, ids, device, first_step_probs=sp)
            # geomean
            score = prob ** (1.0 / len(ids)) if ids else 0.0
            if score > best_score:
                best_score = score
                best_c = key

        golds = {k for k, v in dict(zip(cand_keys, [1 if True else 0 for _ in cand_keys])).items()}
        # The JSON provides target_scores mapping key->0/1; recompute correctly
        # Find which candidate(s) are marked 1
        # NOTE: reconstruct from examples list
        total += 1
        # find gold set for this example
        # (we need to map back to original example to get target_scores)
        # Simpler: pass along original target_scores via closure
        # Instead, adapt items to include gold set

    # Rebuild items with gold sets and re-evaluate properly
    total = 0
    correct = 0
    by_cat = defaultdict(lambda: {'total': 0, 'correct': 0})
    for ex in examples:
        pre_ids = tokenizer.encode(ex['input'], bos=True, eos=False).tolist()
        cand_keys = list(ex['target_scores'].keys())
        cand_lists = [tokenizer.encode(' ' + c, bos=False, eos=False).tolist() for c in cand_keys]
        if not pre_ids or not cand_lists:
            continue
        ctx = torch.tensor([pre_ids], device=device)
        with torch.no_grad():
            logits = model(ctx)
        sp = torch.softmax(logits[0, -1, :], dim=-1)

        best_c = None
        best_score = -float('inf')
        for key, ids in zip(cand_keys, cand_lists):
            prob = np_chain_logprob(model, ctx, ids, device, first_step_probs=sp)
        print(f"DEBUG example={i} input_len={len(pre_ids)} num_cands={len(cand_lists)} device={device}", file=sys.stderr)
        if cand_lists:
            print(f"DEBUG first_cand_len={len(cand_lists[0])} first_cand_ids={cand_lists[0][:8]}", file=sys.stderr)
            score = prob ** (1.0 / len(ids)) if ids else 0.0
            if score > best_score:
                print(f"DEBUG calling model with ctx_shape={(ctx.shape)}", file=sys.stderr)
                print(f"DEBUG logits_shape={None if logits is None else tuple(logits.shape)}", file=sys.stderr)
                best_score = score
                best_c = key

        gold = {k for k, v in ex['target_scores'].items() if v}
        is_correct = best_c in gold
        total += 1
        correct += int(is_correct)
        cat = ex.get('comment', 'NA')
        by_cat[cat]['total'] += 1
        by_cat[cat]['correct'] += int(is_correct)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    step = step_from_checkpoint(ckpt_file)

    overall = correct / total if total else 0.0
    per_category = {c: (v['correct'] / v['total'] if v['total'] else 0.0, v['correct'], v['total']) for c, v in by_cat.items()}
    return step, overall, per_category


def write_results(step: int, ckpt: Path, overall: float, per_cat: Dict[str, tuple], result_name: Path):
    result_name.parent.mkdir(parents=True, exist_ok=True)
    # write a compact CSV with one row per category plus overall
    with open(result_name, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'checkpoint', 'category', 'accuracy', 'correct', 'total'])
        writer.writerow([step, str(ckpt), 'ALL', f'{overall:.6f}', sum(v[1] for v in per_cat.values()), sum(v[2] for v in per_cat.values())])
        for c, (acc, correct, total) in sorted(per_cat.items()):
            writer.writerow([step, str(ckpt), c, f'{acc:.6f}', correct, total])


def main():
    p = argparse.ArgumentParser(description='Evaluate short_nested_outer dataset against lit checkpoints')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint', type=Path)
    g.add_argument('--checkpoint-dir', type=Path)
    p.add_argument('--tokenizer-dir', type=Path, default=Path('checkpoints/gpt2'))
    p.add_argument('--data', default='short_nested_inner_english.json')
    p.add_argument('--max-seq-length', type=int, default=0)
    p.add_argument('--result-name', type=Path, default=Path('results/eval_short_nested_inner.csv'))
    args = p.parse_args()

    examples = load_examples(args.data)

    if args.checkpoint is not None:
        ckpts = [resolve_checkpoint_file(args.checkpoint)]
    else:
        ckpts = sorted(
            args.checkpoint_dir.glob('step-*/lit_model.pth'),
            key=lambda x: int(x.parent.name.split('-')[1]),
        )
        if not ckpts:
            raise FileNotFoundError(f'No step checkpoints found under {args.checkpoint_dir}')

    all_rows = []
    for ckpt in ckpts:
        step, overall, per_cat = compute_one_checkpoint(ckpt, args.tokenizer_dir, examples, args.max_seq_length)
        write_results(step, ckpt, overall, per_cat, args.result_name)
        print(f'step {step} | overall acc: {overall:.4f}')


if __name__ == '__main__':
    main()
