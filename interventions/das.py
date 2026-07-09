"""
Thin experiment wrapper around CausalGym / pyvene for a DAS sweep over
(layer, position) pairs.

DAS itself (the rotated low-rank subspace intervention, its optimization,
and the intervenable forward pass) is entirely provided by pyvene via
`LowRankRotatedSpaceIntervention` + `IntervenableModel`. This script only
adds: dataframe loading, tokenization, a selected-position CE training
loop, ODDS evaluation, and result/checkpoint saving.

Assumptions made explicit here (adjust if your CausalGym fork differs):

  * `--positions` is the outer sweep grid (paired with `--layers`), so for
    each sweep cell the same position is applied to every example, i.e. it
    overrides the per-row `base_position` / `source_position` columns in
    the dataframes for that (layer, position) run. If you'd rather keep
    per-row positions and only sweep layers, drop the position-override
    lines below and iterate `args.positions` as a no-op label instead.
  * `yb` / `ys` are single-token targets: we take the first BPE token of
    `" " + str(target)` under the tokenizer.
  * ODDS is computed at the same token positions as `--loss-positions`
    and averaged over them (the task spec does not define a separate
    eval position), in addition to being averaged over examples.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import pyvene as pv
from pyvene import (
    IntervenableConfig,
    IntervenableModel,
    RepresentationConfig,
    LowRankRotatedSpaceIntervention,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-checkpoint", required=True)
    p.add_argument("--eval-checkpoint", required=True)
    p.add_argument("--train-df", required=True)
    p.add_argument("--eval-df", required=True)
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--positions", nargs="+", type=int, required=True)
    p.add_argument("--loss-positions", nargs="+", type=int, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def load_model(checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def target_token_id(tokenizer, target):
    ids = tokenizer.encode(" " + str(target).strip(), add_special_tokens=False)
    return ids[0]


def make_intervenable(model, layer, device):
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component="block_output",
                unit="pos",
                max_number_of_units=1,
                low_rank_dimension=1,
            )
        ],
        intervention_types=LowRankRotatedSpaceIntervention,
    )
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable


def build_unit_locations(base_pos, source_pos):
    # single intervention (one representation) -> outer list of length 1
    return {
        "sources->base": (
            [[[int(p)] for p in source_pos]],
            [[[int(p)] for p in base_pos]],
        )
    }


def tokenize_batch(tokenizer, texts, device):
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


def train_intervention(intervenable, tokenizer, train_df, loss_positions,
                        epochs, batch_size, lr, device):
    optimizer = torch.optim.Adam(intervenable.get_trainable_parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        df = train_df.sample(frac=1).reset_index(drop=True)
        for start in tqdm(range(0, len(df), batch_size), desc=f"train epoch {epoch}"):
            batch = df.iloc[start:start + batch_size]
            base = tokenize_batch(tokenizer, batch["base_text"].tolist(), device)
            source = tokenize_batch(tokenizer, batch["source_text"].tolist(), device)
            unit_locations = build_unit_locations(
                batch["base_position"].tolist(), batch["source_position"].tolist()
            )
            target_ids = torch.tensor(
                [target_token_id(tokenizer, y) for y in batch["ys"]], device=device
            )

            _, cf_outputs = intervenable(base, [source], unit_locations=unit_locations)
            logits = cf_outputs.logits

            loss = sum(
                F.cross_entropy(logits[:, pos, :], target_ids) for pos in loss_positions
            ) / len(loss_positions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    tail = losses[-max(1, len(losses) // 10):]
    return float(np.mean(tail))


@torch.no_grad()
def evaluate(intervenable, base_model, tokenizer, eval_df, loss_positions,
             batch_size, device):
    odds_all = []
    for start in tqdm(range(0, len(eval_df), batch_size), desc="eval"):
        batch = eval_df.iloc[start:start + batch_size]
        base = tokenize_batch(tokenizer, batch["base_text"].tolist(), device)
        source = tokenize_batch(tokenizer, batch["source_text"].tolist(), device)
        unit_locations = build_unit_locations(
            batch["base_position"].tolist(), batch["source_position"].tolist()
        )

        yb_ids = torch.tensor(
            [target_token_id(tokenizer, y) for y in batch["yb"]], device=device
        )
        ys_ids = torch.tensor(
            [target_token_id(tokenizer, y) for y in batch["ys"]], device=device
        )

        base_logits = base_model(**base).logits
        _, cf_outputs = intervenable(base, [source], unit_locations=unit_locations)
        int_logits = cf_outputs.logits

        for pos in loss_positions:
            base_logp = F.log_softmax(base_logits[:, pos, :], dim=-1)
            int_logp = F.log_softmax(int_logits[:, pos, :], dim=-1)

            log_p_yb_base = base_logp.gather(1, yb_ids[:, None]).squeeze(1)
            log_p_ys_base = base_logp.gather(1, ys_ids[:, None]).squeeze(1)
            log_p_yb_int = int_logp.gather(1, yb_ids[:, None]).squeeze(1)
            log_p_ys_int = int_logp.gather(1, ys_ids[:, None]).squeeze(1)

            odds = (log_p_yb_base - log_p_ys_base) + (log_p_ys_int - log_p_yb_int)
            odds_all.extend(odds.cpu().tolist())

    return float(np.mean(odds_all)), float(np.std(odds_all))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_model, train_tokenizer = load_model(args.train_checkpoint, device)
    eval_model, eval_tokenizer = load_model(args.eval_checkpoint, device)

    train_df_src = pd.read_csv(args.train_df)
    eval_df_src = pd.read_csv(args.eval_df)

    interventions_dir = os.path.join(args.output_dir, "interventions")
    os.makedirs(interventions_dir, exist_ok=True)

    results = []
    for layer in args.layers:
        for pos in args.positions:
            print(f"=== layer={layer} position={pos} ===")

            train_df = train_df_src.copy()
            train_df["base_position"] = pos
            train_df["source_position"] = pos

            eval_df = eval_df_src.copy()
            eval_df["base_position"] = pos
            eval_df["source_position"] = pos

            intervenable = make_intervenable(train_model, layer, device)
            train_loss = train_intervention(
                intervenable, train_tokenizer, train_df, args.loss_positions,
                args.epochs, args.batch_size, args.lr, device,
            )

            state_dict = intervenable.state_dict()
            cell_dir = os.path.join(interventions_dir, f"layer{layer}_pos{pos}")
            os.makedirs(cell_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(cell_dir, "state_dict.pt"))

            eval_intervenable = make_intervenable(eval_model, layer, device)
            eval_intervenable.load_state_dict(state_dict)
            odds_mean, odds_std = evaluate(
                eval_intervenable, eval_model, eval_tokenizer, eval_df,
                args.loss_positions, args.batch_size, device,
            )

            results.append({
                "layer": layer,
                "position": pos,
                "odds_mean": odds_mean,
                "odds_std": odds_std,
                "train_loss": train_loss,
            })

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    main()