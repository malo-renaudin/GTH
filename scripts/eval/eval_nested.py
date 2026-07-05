from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch 
import json
import argparse 
import os

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--ckpt", type=str)
argument_parser.add_argument("--eval-dataset", type=str)
argument_parser.add_argument("--out-metrics", type=str, default=None)
args = argument_parser.parse_args()

ckpt_dir = args.ckpt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(
    ckpt_dir,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    ckpt_dir,
    local_files_only=True,
).to(device)

with open(args.eval_dataset) as f:
    test_data = json.load(f)
    
test = test_data["examples"]

import torch
import math
from collections import defaultdict

device = next(model.parameters()).device
model.eval()

correct = 0
total = 0

cat_correct = defaultdict(int)
cat_total = defaultdict(int)

with torch.no_grad():
    for ex in test:
        print(ex["input"])

        inputs = tokenizer(ex["input"], return_tensors="pt").to(device)

        scores = {}

        # skip examples where any candidate is not a single token
        # Prefer the space-prefixed form (" eat") over the bare form ("eat")
        # since GPT-2 BPE encodes mid-sentence words with a leading space.
        def _best_ids(word):
            spaced = tokenizer.encode(" " + word, add_special_tokens=False)
            bare   = tokenizer.encode(word, add_special_tokens=False)
            if len(spaced) == 1:
                return spaced   # prefer " eat" over "eat" when both exist
            return bare

        all_word_ids = {word: _best_ids(word) for word in ex["target_scores"]}
        if any(len(ids) != 1 for ids in all_word_ids.values()):
            for word, ids in all_word_ids.items():
                if len(ids) != 1:
                    print(f"Skipping example: '{word}' splits into multiple tokens: {ids}")
            continue

        for word, word_ids in all_word_ids.items():
            # build context progressively
            cur_input = inputs["input_ids"]

            log_probs = 0.0

            for tid in word_ids:
                out = model(input_ids=cur_input)
                logits = out.logits[0, -1]
                lp = logits.log_softmax(dim=-1)

                log_probs += lp[tid].item()

                # append token for next step
                cur_input = torch.cat(
                    [cur_input, torch.tensor([[tid]], device=device)],
                    dim=1
                )

            scores[word] = log_probs

        pred = max(scores, key=scores.get)
        gold = max(ex["target_scores"], key=ex["target_scores"].get)

        is_correct = (pred == gold)

        correct += is_correct
        total += 1

        category = ex["comment"]
        cat_correct[category] += is_correct
        cat_total[category] += 1

print(f"Overall accuracy: {correct/total:.4f}")

print("\nAccuracy by category:")
for cat in sorted(cat_total):
    acc = cat_correct[cat] / cat_total[cat]
    print(f"{cat}: {acc:.4f} ({cat_correct[cat]}/{cat_total[cat]})")
    
eval_metrics = {
    "accuracy": correct / total,
    "per_category": {
        cat: cat_correct[cat] / cat_total[cat]
        for cat in cat_total
    }
}

if args.out_metrics is not None:
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    with open(args.out_metrics, "w") as f:
        json.dump(eval_metrics, f, indent=2)