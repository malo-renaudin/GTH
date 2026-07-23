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
    "--data_file",
    type=str,
    required=True,
    help="Path to the WH embedded CSV file"
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

data_file = args.data_file
checkpoint_dir = args.checkpoint_dir
output_file = args.output_file

checkpoint_paths = sorted(
    glob.glob(os.path.join(checkpoint_dir, "*"))
)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load data
# -------------------------
df = pd.read_csv(data_file)


# -------------------------
# Surprisal function
# -------------------------
def token_surprisal(model, tokenizer, context, continuation):

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

    target_id = full_ids[-1]

    inputs = torch.tensor(
        [context_ids],
        device=model.device
    )

    with torch.no_grad():
        logits = model(inputs).logits

    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

    return (
        -log_probs[0, target_id].item()
        / torch.log(torch.tensor(2.)).item()
    )


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

    scores = []


    for qid, group in tqdm(
        df.groupby("quadruplet_id"),
        desc=os.path.basename(ckpt)
    ):

        filler_context = group[
            group.filler == 1
        ].iloc[0]["pre_gap_text"]

        no_filler_context = group[
            group.filler == 0
        ].iloc[0]["pre_gap_text"]


        filler_gap = group[
            (group.filler == 1) &
            (group.gap == 1)
        ].iloc[0]["post_gap_text"]

        filler_no_gap = group[
            (group.filler == 1) &
            (group.gap == 0)
        ].iloc[0]["post_gap_text"]

        no_filler_gap = group[
            (group.filler == 0) &
            (group.gap == 1)
        ].iloc[0]["post_gap_text"]

        no_filler_no_gap = group[
            (group.filler == 0) &
            (group.gap == 0)
        ].iloc[0]["post_gap_text"]


        S_filler_gap = token_surprisal(
            model, tokenizer,
            filler_context,
            filler_gap
        )

        S_filler_no_gap = token_surprisal(
            model, tokenizer,
            filler_context,
            filler_no_gap
        )

        S_no_filler_gap = token_surprisal(
            model, tokenizer,
            no_filler_context,
            no_filler_gap
        )

        S_no_filler_no_gap = token_surprisal(
            model, tokenizer,
            no_filler_context,
            no_filler_no_gap
        )


        # Positive = model prefers gap after filler
        score = (
            (S_filler_no_gap - S_filler_gap)
            -
            (S_no_filler_no_gap - S_no_filler_gap)
        )

        scores.append(score)


    scores = pd.Series(scores)

    results.append({
        "checkpoint": os.path.basename(ckpt),
        "accuracy": (scores > 0).mean(),
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "n_items": len(scores)
    })


    del model
    torch.cuda.empty_cache()


# -------------------------
# Save results
# -------------------------
results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    output_file,
    index=False
)

print(f"\nSaved to {output_file}")