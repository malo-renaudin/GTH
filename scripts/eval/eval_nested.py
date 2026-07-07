import argparse
import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(ckpt_dir: str, eval_dataset_path: str, out_metrics_path: str, model=None, tokenizer=None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_here = False
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, local_files_only=True
        ).to(device)
        loaded_here = True

    with open(eval_dataset_path) as f:
        test = json.load(f)["examples"]

    def _best_ids(word):
        spaced = tokenizer.encode(" " + word, add_special_tokens=False)
        bare   = tokenizer.encode(word, add_special_tokens=False)
        return spaced if len(spaced) == 1 else bare

    correct = 0
    total   = 0
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)

    model.eval()
    with torch.no_grad():
        for ex in test:
            inputs       = tokenizer(ex["input"], return_tensors="pt").to(device)
            all_word_ids = {word: _best_ids(word) for word in ex["target_scores"]}

            if any(len(ids) != 1 for ids in all_word_ids.values()):
                for word, ids in all_word_ids.items():
                    if len(ids) != 1:
                        print(f"Skipping example: '{word}' splits into multiple tokens: {ids}")
                continue

            scores = {}
            for word, word_ids in all_word_ids.items():
                cur_input = inputs["input_ids"]
                log_probs = 0.0
                for tid in word_ids:
                    logits     = model(input_ids=cur_input).logits[0, -1]
                    log_probs += logits.log_softmax(dim=-1)[tid].item()
                    cur_input  = torch.cat(
                        [cur_input, torch.tensor([[tid]], device=device)], dim=1
                    )
                scores[word] = log_probs

            pred = max(scores, key=scores.get)
            gold = max(ex["target_scores"], key=ex["target_scores"].get)
            is_correct = int(pred == gold)

            correct += is_correct
            total   += 1
            cat_correct[ex["comment"]] += is_correct
            cat_total[ex["comment"]]   += 1

    print(f"Overall accuracy: {correct / total:.4f}")
    print("\nAccuracy by category:")
    for cat in sorted(cat_total):
        print(f"{cat}: {cat_correct[cat] / cat_total[cat]:.4f} ({cat_correct[cat]}/{cat_total[cat]})")

    eval_metrics = {
        "accuracy": correct / total,
        "per_category": {c: cat_correct[c] / cat_total[c] for c in cat_total},
    }

    if out_metrics_path is not None:
        os.makedirs(os.path.dirname(out_metrics_path), exist_ok=True)
        with open(out_metrics_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)

    if loaded_here:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return eval_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          type=str, required=True)
    p.add_argument("--eval-dataset",  type=str, required=True)
    p.add_argument("--out-metrics",   type=str, default=None)
    args = p.parse_args()
    run(args.ckpt, args.eval_dataset, args.out_metrics)