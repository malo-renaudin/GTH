#!/usr/bin/env python3
"""
eval_locality.py

At the ORC gap site (right after the embedded verb), compare:
  P(N1 | ctx)  vs  P(N2 | ctx)

where N1 is the moved NP (syntactically correct filler, further from the gap)
and N2 is the embedded subject (locality attractor, adjacent to the gap).

Accuracy > 0.5  → model respects syntactic dependency over locality
Accuracy ≈ 0.5  → chance
Accuracy < 0.5  → model is distracted by the closer noun (locality effect)

Input JSON: generate_simple_datasets/generated_simple_datasets/locality_filler_gap.json

Usage
-----
python scripts/eval/eval_locality.py \\
    --ckpt checkpoints/my_model/checkpoint-1000 \\
    --eval-dataset generate_simple_datasets/generated_simple_datasets/locality_filler_gap.json \\
    --out-metrics results/locality_1000.json
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(ckpt_dir: str, eval_dataset_path: str, out_metrics_path: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, local_files_only=True
    ).to(device)

    with open(eval_dataset_path, encoding="utf-8") as f:
        examples = json.load(f)["examples"]

    correct = 0
    total   = 0
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for ex in examples:
            inputs = tokenizer(ex["input"], return_tensors="pt").to(device)
            sp     = model(input_ids=inputs["input_ids"]).logits[0, -1].log_softmax(dim=-1)

            # All target words are single tokens — direct lookup
            scores = {}
            for word in ex["target_scores"]:
                ids = tokenizer.encode(" " + word, add_special_tokens=False)
                if len(ids) != 1:
                    break           # skip if multi-token (shouldn't happen with this dataset)
                scores[word] = sp[ids[0]].item()
            else:
                pred = max(scores, key=scores.get)
                gold = max(ex["target_scores"], key=ex["target_scores"].get)
                ok   = int(pred == gold)

                correct += ok
                total   += 1
                cat      = ex.get("comment", "unknown")
                cat_correct[cat] += ok
                cat_total[cat]   += 1

    print(f"Overall accuracy: {correct / total:.4f}  ({correct}/{total})")
    print("\nAccuracy by category:")
    for cat in sorted(cat_total):
        print(f"  {cat}: {cat_correct[cat] / cat_total[cat]:.4f}  ({cat_correct[cat]}/{cat_total[cat]})")

    metrics = {
        "accuracy":     correct / total if total else float("nan"),
        "n_total":      total,
        "per_category": {c: cat_correct[c] / cat_total[c] for c in cat_total},
    }

    if out_metrics_path is not None:
        os.makedirs(os.path.dirname(out_metrics_path), exist_ok=True)
        with open(out_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Locality filler-gap evaluation.")
    p.add_argument("--ckpt",          type=str, required=True)
    p.add_argument("--eval-dataset",  type=str, required=True)
    p.add_argument("--out-metrics",   type=str, default=None)
    args = p.parse_args()
    run(args.ckpt, args.eval_dataset, args.out_metrics)
