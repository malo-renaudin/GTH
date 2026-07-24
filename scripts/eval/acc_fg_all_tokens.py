import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os
import glob
import numpy as np
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
def batch_sequence_surprisal(model, tokenizer, sentences, batch_size=64):
    """
    Returns:
        surprisals: list of np.ndarray
        token_ids:  list of list[int]
    """

    import numpy as np

    encoded = [
        tokenizer(s, add_special_tokens=False).input_ids
        for s in sentences
    ]

    all_surprisals = []

    for start in range(0, len(encoded), batch_size):

        batch = encoded[start:start + batch_size]

        max_len = max(len(x) for x in batch)

        input_ids = []
        attention_mask = []

        for ids in batch:
            pad = max_len - len(ids)
            input_ids.append(ids + [tokenizer.eos_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        input_ids = torch.tensor(input_ids, device=model.device)
        attention_mask = torch.tensor(attention_mask, device=model.device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits

        log_probs = torch.log_softmax(logits, dim=-1)

        for i, ids in enumerate(batch):

            targets = torch.tensor(ids[1:], device=model.device)

            lp = log_probs[i, :len(ids)-1]

            surprisal = (
                -lp[
                    torch.arange(len(targets), device=model.device),
                    targets
                ]
                / torch.log(torch.tensor(2., device=model.device))
            )

            all_surprisals.append(surprisal.cpu().numpy())

    return all_surprisals, encoded
# -------------------------
# Evaluate checkpoints
# -------------------------
# results = []
token_results = []

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
        

        sentences = df["sentence"].tolist()

        surprisals, encoded = batch_sequence_surprisal(
            model,
            tokenizer,
            sentences,
            batch_size=128,
        )

        condition_surps = {
            "fg": {},
            "f_no_g": {},
            "no_f_g": {},
            "no_f_no_g": {},
        }

        for (_, row), s, ids in zip(df.iterrows(), surprisals, encoded):

            if row.filler == 1 and row.gap == 1:
                cond = "fg"
            elif row.filler == 1:
                cond = "f_no_g"
            elif row.gap == 1:
                cond = "no_f_g"
            else:
                cond = "no_f_no_g"

            L = len(ids)

            if L not in condition_surps[cond]:
                condition_surps[cond][L] = {
                    "surprisals": [],
                    "tokens": tokenizer.convert_ids_to_tokens(ids),
                }

            condition_surps[cond][L]["surprisals"].append(s)


        for cond, lengths in condition_surps.items():

            for L, info in lengths.items():

                arr = np.stack(info["surprisals"])

                mean = arr.mean(axis=0)
                std = arr.std(axis=0)

                tokens = info["tokens"][1:]   # first token has no surprisal

                for pos, (tok, m, sd) in enumerate(zip(tokens, mean, std)):

                    token_results.append({
                        "checkpoint": os.path.basename(ckpt),
                        "dataset": os.path.basename(data_file),
                        "condition": cond,
                        "token_length": L,
                        "position": pos,
                        "token": tokenizer.convert_tokens_to_string([tok]),
                        "mean_surprisal": m,
                        "std_surprisal": sd,
                        "n_sentences": arr.shape[0],
                        "sentence_length": len(tokens),
                    })

# results_df = pd.DataFrame(results)
# results_df.to_csv(output_file, index=False)

token_df = pd.DataFrame(token_results)
token_df.to_csv(
    output_file.replace(".csv", "_tokens.csv"),
    index=False,
)

print(f"Saved summary to {output_file}")
print(f"Saved token surprisals to {output_file.replace('.csv', '_tokens.csv')}")