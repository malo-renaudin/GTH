"""
DAS experiment: thin wrapper around pyvene's LowRankRotatedSpaceIntervention.
Sweeps over (layer, position_label) pairs.

CSV columns expected:
    quadruplet_id, position_label, base_text, source_text,
    yb, ys, base_anchor, source_anchor

Token positions are resolved dynamically from the anchor strings using offset
mappings, so the CSV is tokenizer-agnostic.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pyvene import (
    IntervenableConfig,
    IntervenableModel,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-checkpoint", required=True)
    p.add_argument("--eval-checkpoint", required=True)
    p.add_argument("--train-df", required=True)
    p.add_argument("--eval-df", required=True)
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--loss-positions", nargs="+", type=int, required=True,
                   help="token indices (in the base sentence) where CE / ODDS are computed")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint, device):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Token position resolution (anchor-based, robust)
# ---------------------------------------------------------------------------

def find_token_position(tokenizer, text, anchor):
    """Return the token index whose offset overlaps the first unambiguous
    whole-word occurrence of `anchor` in `text`.
    Raises ValueError if ambiguous after deduplication.
    Prints a warning and returns None if not found.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    pattern = r"(?<!\w)" + re.escape(str(anchor)) + r"(?!\w)"
    char_spans = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    if not char_spans:
        print(f"WARNING: anchor '{anchor}' not found in: {text!r}")
        return None

    candidates = []
    for char_start, char_end in char_spans:
        for i, (s, e) in enumerate(offsets):
            if s < char_end and e > char_start:
                candidates.append(i)
                break

    unique = list(dict.fromkeys(candidates))
    if len(unique) > 1:
        raise ValueError(
            f"Anchor '{anchor}' is ambiguous in {text!r} — "
            f"maps to token positions {unique}"
        )
    if not unique:
        print(f"WARNING: anchor '{anchor}' found in text but maps to no token: {text!r}")
        return None
    return unique[0]


# ---------------------------------------------------------------------------
# Target tokenization (multi-token)
# ---------------------------------------------------------------------------

def tokenize_targets(tokenizer, target, device):
    """Return a 1D tensor of token IDs for `target` (prefixed with a space)."""
    ids = tokenizer.encode(" " + str(target).strip(), add_special_tokens=False)
    return torch.tensor(ids, device=device)


def sequence_log_prob(logits, target_ids, loss_positions):
    """Sum of log-probs for target_ids at the given loss_positions.

    logits         : (1, seq_len, vocab)
    target_ids     : (n,)  — tokens to score (first n used)
    loss_positions : list[int] — positions p such that logits[:,p,:] predicts
                     the token at p+1; we use the first min(n, len(positions))
    Returns a scalar tensor.
    """
    n = min(len(target_ids), len(loss_positions))
    log_prob = torch.tensor(0.0, device=logits.device)
    for k in range(n):
        pos = loss_positions[k]
        log_prob = log_prob + F.log_softmax(logits[0, pos, :], dim=-1)[target_ids[k]]
    return log_prob


# ---------------------------------------------------------------------------
# pyvene helpers
# ---------------------------------------------------------------------------

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


def build_unit_locations(base_positions, source_positions):
    return {
        "sources->base": (
            [[[int(p)] for p in source_positions]],
            [[[int(p)] for p in base_positions]],
        )
    }


def tokenize_batch(tokenizer, texts, device):
    return tokenizer(texts, return_tensors="pt", padding=True).to(device)


def resolve_positions(tokenizer, texts, anchors):
    return [find_token_position(tokenizer, t, a) for t, a in zip(texts, anchors)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_intervention(intervenable, tokenizer, train_df, loss_positions,
                       epochs, batch_size, lr, device):
    optimizer = torch.optim.Adam(intervenable.get_trainable_parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        df = train_df.sample(frac=1).reset_index(drop=True)
        for start in tqdm(range(0, len(df), batch_size), desc=f"train epoch {epoch}"):
            batch = df.iloc[start:start + batch_size]

            base_pos = resolve_positions(tokenizer,
                                         batch["base_text"].tolist(),
                                         batch["base_anchor"].tolist())
            src_pos = resolve_positions(tokenizer,
                                        batch["source_text"].tolist(),
                                        batch["source_anchor"].tolist())
            valid = [i for i, (b, s) in enumerate(zip(base_pos, src_pos))
                     if b is not None and s is not None]
            if not valid:
                continue

            base_enc = tokenize_batch(tokenizer, [batch.iloc[i]["base_text"] for i in valid], device)
            src_enc = tokenize_batch(tokenizer, [batch.iloc[i]["source_text"] for i in valid], device)
            unit_locations = build_unit_locations(
                [base_pos[i] for i in valid], [src_pos[i] for i in valid]
            )

            _, cf_outputs = intervenable(base_enc, [src_enc], unit_locations=unit_locations)
            logits = cf_outputs.logits  # (B, seq, vocab)

            # CE at each loss_position, first token of ys as target.
            ys_first = torch.tensor(
                [tokenize_targets(tokenizer, batch.iloc[i]["ys"], device)[0].item()
                 for i in valid],
                device=device,
            )
            loss = sum(
                F.cross_entropy(logits[:, pos, :], ys_first)
                for pos in loss_positions
            ) / len(loss_positions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    tail = losses[-max(1, len(losses) // 10):]
    return float(np.mean(tail)) if tail else float("nan")


# ---------------------------------------------------------------------------
# Evaluation (ODDS over full target strings)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_odds(intervenable, base_model, tokenizer, eval_df,
                 loss_positions, batch_size, device):
    odds_all = []
    for start in tqdm(range(0, len(eval_df), batch_size), desc="eval"):
        batch = eval_df.iloc[start:start + batch_size]

        base_pos = resolve_positions(tokenizer,
                                     batch["base_text"].tolist(),
                                     batch["base_anchor"].tolist())
        src_pos = resolve_positions(tokenizer,
                                    batch["source_text"].tolist(),
                                    batch["source_anchor"].tolist())
        valid = [i for i, (b, s) in enumerate(zip(base_pos, src_pos))
                 if b is not None and s is not None]
        if not valid:
            continue

        base_enc = tokenize_batch(tokenizer, [batch.iloc[i]["base_text"] for i in valid], device)
        src_enc = tokenize_batch(tokenizer, [batch.iloc[i]["source_text"] for i in valid], device)
        unit_locations = build_unit_locations(
            [base_pos[i] for i in valid], [src_pos[i] for i in valid]
        )

        base_logits = base_model(**base_enc).logits
        _, cf_outputs = intervenable(base_enc, [src_enc], unit_locations=unit_locations)
        int_logits = cf_outputs.logits

        for j, idx in enumerate(valid):
            row = batch.iloc[idx]
            yb_ids = tokenize_targets(tokenizer, row["yb"], device)
            ys_ids = tokenize_targets(tokenizer, row["ys"], device)

            log_p_yb_base = sequence_log_prob(base_logits[j:j+1], yb_ids, loss_positions).item()
            log_p_ys_base = sequence_log_prob(base_logits[j:j+1], ys_ids, loss_positions).item()
            log_p_ys_int = sequence_log_prob(int_logits[j:j+1], ys_ids, loss_positions).item()
            log_p_yb_int = sequence_log_prob(int_logits[j:j+1], yb_ids, loss_positions).item()

            odds_all.append((log_p_yb_base - log_p_ys_base) + (log_p_ys_int - log_p_yb_int))

    if not odds_all:
        return float("nan"), float("nan")
    return float(np.mean(odds_all)), float(np.std(odds_all))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_model, train_tokenizer = load_model(args.train_checkpoint, device)
    eval_model, eval_tokenizer = load_model(args.eval_checkpoint, device)

    train_df_full = pd.read_csv(args.train_df)
    eval_df_full = pd.read_csv(args.eval_df)

    interventions_dir = os.path.join(args.output_dir, "interventions")
    os.makedirs(interventions_dir, exist_ok=True)

    results = []
    position_labels = train_df_full["position_label"].unique()

    for layer in args.layers:
        for label in position_labels:
            print(f"=== layer={layer}  position_label={label} ===")

            tr = train_df_full[train_df_full["position_label"] == label].reset_index(drop=True)
            ev = eval_df_full[eval_df_full["position_label"] == label].reset_index(drop=True)

            intervenable = make_intervenable(train_model, layer, device)
            train_loss = train_intervention(
                intervenable, train_tokenizer, tr, args.loss_positions,
                args.epochs, args.batch_size, args.lr, device,
            )

            state_dict = intervenable.state_dict()
            cell_dir = os.path.join(interventions_dir, f"layer{layer}_{label}")
            os.makedirs(cell_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(cell_dir, "state_dict.pt"))

            eval_intervenable = make_intervenable(eval_model, layer, device)
            eval_intervenable.load_state_dict(state_dict)
            odds_mean, odds_std = evaluate_odds(
                eval_intervenable, eval_model, eval_tokenizer, ev,
                args.loss_positions, args.batch_size, device,
            )

            results.append({
                "layer": layer,
                "position_label": label,
                "odds_mean": odds_mean,
                "odds_std": odds_std,
                "train_loss": train_loss,
            })
            print(f"  odds_mean={odds_mean:.4f}  odds_std={odds_std:.4f}  train_loss={train_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print("Done. Results saved to", os.path.join(args.output_dir, "results.csv"))


if __name__ == "__main__":
    main()