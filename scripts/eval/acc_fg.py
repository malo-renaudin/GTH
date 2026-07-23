import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import os
import glob

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory containing subset CSV files"
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="Directory containing HF checkpoints"
)

parser.add_argument(
    "--output_file",
    type=str,
    default="embedded_wh_minimal_pair_accuracy.csv",
    help="Output CSV file"
)

args = parser.parse_args()

data_files = sorted(
    glob.glob(os.path.join(args.data_dir, "*.csv"))
)
checkpoint_dir = args.checkpoint_dir
output_file = args.output_file

checkpoint_paths = sorted(
    glob.glob(os.path.join(checkpoint_dir, "*"))
)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load data
# -------------------------
# df = pd.read_csv(data_file)


# -------------------------
# Surprisal function
# -------------------------
def batch_token_surprisal(model, tokenizer, contexts, continuations, batch_size=64):
    """
    contexts: list of strings
    continuations: list of one-token continuations (with or without leading space)

    Returns: list of surprisal values in bits
    """

    surprisals = []

    # prepare all token ids
    all_context_ids = []
    all_target_ids = []

    for context, continuation in zip(contexts, continuations):

        context = context.rstrip()
        continuation = " " + continuation.strip()

        context_ids = tokenizer(
            context,
            add_special_tokens=False
        ).input_ids

        full_ids = tokenizer(
            context + continuation,
            add_special_tokens=False
        ).input_ids

        assert len(full_ids) == len(context_ids) + 1, (
            f"{repr(continuation)} is not one token: "
            f"{tokenizer.tokenize(continuation)}"
        )

        all_context_ids.append(context_ids)
        all_target_ids.append(full_ids[-1])


    # batch by padding
    for start in range(0, len(all_context_ids), batch_size):

        batch_contexts = all_context_ids[start:start+batch_size]
        batch_targets = all_target_ids[start:start+batch_size]

        max_len = max(len(x) for x in batch_contexts)

        input_ids = []
        attention_mask = []

        for ids in batch_contexts:
            pad_len = max_len - len(ids)
            input_ids.append(ids + [tokenizer.eos_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(
            input_ids,
            device=model.device
        )

        attention_mask = torch.tensor(
            attention_mask,
            device=model.device
        )


        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits


        # position of last real token for each sequence
        last_positions = attention_mask.sum(dim=1) - 1

        next_logits = logits[
            torch.arange(len(batch_targets), device=model.device),
            last_positions
        ]

        log_probs = torch.log_softmax(
            next_logits,
            dim=-1
        )

        batch_surprisal = (
            -log_probs[
                torch.arange(len(batch_targets), device=model.device),
                torch.tensor(batch_targets, device=model.device)
            ]
            / torch.log(torch.tensor(2., device=model.device))
        )

        surprisals.extend(
            batch_surprisal.cpu().tolist()
        )

    return surprisals
# -------------------------
# Evaluate checkpoints
# -------------------------
results = []

for ckpt in checkpoint_paths:

    print(f"\nEvaluating {ckpt}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model.eval()

    subset_accuracies = []
    subset_fg_accuracies = []
    subset_f_no_g_accuracies = []
    subset_no_f_g_accuracies = []
    subset_no_f_no_g_accuracies = []
    subset_scores = []

    for data_file in data_files:

        print(f"  Dataset: {os.path.basename(data_file)}")

        df = pd.read_csv(data_file)

        contexts = []
        continuations = []
        quad_ids = []

        for qid, group in df.groupby("quadruplet_id"):

            filler_context = group[group.filler == 1].iloc[0]["pre_gap_text"]
            no_filler_context = group[group.filler == 0].iloc[0]["pre_gap_text"]

            filler_gap = group[(group.filler == 1) & (group.gap == 1)].iloc[0]["post_gap_text"]
            filler_no_gap = group[(group.filler == 1) & (group.gap == 0)].iloc[0]["post_gap_text"]

            no_filler_gap = group[(group.filler == 0) & (group.gap == 1)].iloc[0]["post_gap_text"]
            no_filler_no_gap = group[(group.filler == 0) & (group.gap == 0)].iloc[0]["post_gap_text"]

            contexts.extend([
                filler_context,
                filler_context,
                no_filler_context,
                no_filler_context,
            ])

            continuations.extend([
                filler_gap,
                filler_no_gap,
                no_filler_gap,
                no_filler_no_gap,
            ])

            quad_ids.append(qid)


        surprisals = batch_token_surprisal(
            model,
            tokenizer,
            contexts,
            continuations,
            batch_size=128
        )


        scores = []
        fg = []
        f_no_g = []
        no_f_g = []
        no_f_no_g = []

        for i in range(len(quad_ids)):

            S_filler_gap = surprisals[4*i]
            S_filler_no_gap = surprisals[4*i+1]
            S_no_filler_gap = surprisals[4*i+2]
            S_no_filler_no_gap = surprisals[4*i+3]

            scores.append(
                (S_filler_no_gap - S_filler_gap)
                -
                (S_no_filler_no_gap - S_no_filler_gap)
            )

            fg.append(S_filler_gap)
            f_no_g.append(S_filler_no_gap)
            no_f_g.append(S_no_filler_gap)
            no_f_no_g.append(S_no_filler_no_gap)


        scores = pd.Series(scores)


        results.append({
            "checkpoint": os.path.basename(ckpt),
            "dataset": os.path.basename(data_file),

            "accuracy": (scores > 0).mean(),
            "mean_score": scores.mean(),
            "std_score": scores.std(),

            "mean_fg": pd.Series(fg).mean(),
            "mean_f_no_g": pd.Series(f_no_g).mean(),
            "mean_no_f_g": pd.Series(no_f_g).mean(),
            "mean_no_f_no_g": pd.Series(no_f_no_g).mean(),

            "n_items": len(scores)
        })

        print(
            f"    {os.path.basename(data_file)} "
            f"accuracy={results[-1]['accuracy']:.4f}"
        )

results_df = pd.DataFrame(results)

print("\nFinal results:")
print(results_df)

results_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved results to {output_file}")