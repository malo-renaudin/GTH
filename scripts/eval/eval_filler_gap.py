import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_gap_surprisals_batch(
    items: List[Tuple[str, str]],
    model, tokenizer, device, batch_size: int = 32,
) -> List[Tuple[float, float]]:
    """
    Batched version: given a list of (pre_gap, post_gap) pairs, return
    [(surprisal_first, surprisal_mean), ...] for each pair.
    """
    results = []

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]

        encoded = []
        for pre_gap, post_gap in batch:
            context_ids = tokenizer.encode(pre_gap, add_special_tokens=False)
            suffix_ids  = tokenizer.encode(post_gap, add_special_tokens=False)
            encoded.append((context_ids, suffix_ids))

        max_len = max(len(c) + len(s) for c, s in encoded)
        pad_id  = 0

        padded = [
            c + s + [pad_id] * (max_len - len(c) - len(s))
            for c, s in encoded
        ]
        x = torch.tensor(padded, device=device)  # (B, max_len)

        with torch.no_grad():
            logits = model(input_ids=x).logits  # (B, max_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)  # (B, max_len, vocab)

        for i, (context_ids, suffix_ids) in enumerate(encoded):
            if not suffix_ids:
                results.append((float("nan"), float("nan")))
                continue
            context_len = len(context_ids)
            surprisals = [
                -log_probs[i, context_len - 1 + j, tok_id].item()
                for j, tok_id in enumerate(suffix_ids)
            ]
            results.append((surprisals[0], sum(surprisals) / len(surprisals)))

    return results


def _mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else float("nan")


def print_metrics(results: List[dict], surprisal_col: str, label: str) -> None:
    by_condition = defaultdict(list)
    for r in results:
        key = (int(r["filler"]), int(r["gap"]))
        v = r[surprisal_col]
        if v == v:  # skip NaN
            by_condition[key].append(v)

    s = {k: _mean(v) for k, v in by_condition.items()}
    s_f0g1 = s.get((0, 1), float("nan"))
    s_f1g1 = s.get((1, 1), float("nan"))
    s_f0g0 = s.get((0, 0), float("nan"))
    s_f1g0 = s.get((1, 0), float("nan"))

    diff_gap    = s_f0g1 - s_f1g1
    diff_nogap  = s_f1g0 - s_f0g0 
    interaction = diff_gap + diff_nogap

    print(f"\n  [{label}]")
    print(f"  surp(filler=0,gap=1) - surp(filler=1,gap=1) = {diff_gap:+.4f}  "
          f"({s_f0g1:.4f} - {s_f1g1:.4f})")
    print(f"  surp(filler=1,gap=0) - surp(filler=0,gap=0) = {diff_nogap:+.4f}  "
          f"({s_f1g0:.4f} - {s_f0g0:.4f})")
    print(f"  Interaction (gap_diff + nogap_diff)          = {interaction:+.4f}")

def _run_lme_one_step(step, df, surprisal_col="surprisal_first"):
    """Worker: fit LME for one step. Returns a dict of results."""
    # step, df_step, surprisal_col = args
    data = df.copy()
    data["filler_c"] = data["filler"].map({0: -0.5, 1: 0.5})
    data["gap_c"]    = data["gap"].map({0: -0.5, 1: 0.5})

    # Raw surprisal as DV; random intercepts by item capture item-level variance
    formula = f"{surprisal_col} ~ filler_c * gap_c"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = smf.mixedlm(formula, data, groups=data["quadruplet_id"]).fit(reml=True)

    params = result.params
    pvalues = result.pvalues
    conf = result.conf_int()

    # --- Masson & Loftus (2003) CIs ---
    # Subtract by-item means to remove item-level variance, then compute
    # condition means and CIs from the within-item variance only.
    item_means = data.groupby("quadruplet_id")[surprisal_col].transform("mean")
    data["ml_centered"] = data[surprisal_col] - item_means

    data["condition"] = (
        data["filler"].map({0: "no_wh", 1: "wh"})
        + "_"
        + data["gap"].map({0: "no_gap", 1: "gap"})
    )
    wide = data.pivot_table(
        index="quadruplet_id", columns="condition", values="ml_centered"
    )
    n_items = len(wide)
    t_crit = stats.t.ppf(0.975, df=n_items - 1)

    def _se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    # Per-item licensing interaction: (wh_no_gap - no_wh_no_gap) - (wh_gap - no_wh_gap)
    per_item_int = (
        (wide["wh_no_gap"] - wide["no_wh_no_gap"])
        - (wide["wh_gap"]  - wide["no_wh_gap"])
    )
    ml_int_mean = per_item_int.mean()
    ml_int_se   = _se(per_item_int)

    return {
        "step": step,
        "n_obs": len(data),
        "n_items": n_items,
        # fixed effects
        "intercept":          params["Intercept"],
        "filler_c":           params["filler_c"],
        "gap_c":              params["gap_c"],
        "filler_c:gap_c":     params["filler_c:gap_c"],
        # licensing interaction = -beta(filler_c:gap_c) with ±0.5 coding
        "licensing_interaction": -params["filler_c:gap_c"],
        # p-values
        "p_filler_c":         pvalues["filler_c"],
        "p_gap_c":            pvalues["gap_c"],
        "p_interaction":      pvalues["filler_c:gap_c"],
        # 95% CI on the interaction from LME (flipped for licensing direction)
        "interaction_ci_low":  -conf.loc["filler_c:gap_c", 1],
        "interaction_ci_high": -conf.loc["filler_c:gap_c", 0],
        "converged": result.converged,
        # Masson & Loftus (2003): condition means and SEs (from within-item variance)
        "ml_mean_wh_gap":       wide["wh_gap"].mean(),
        "ml_mean_wh_no_gap":    wide["wh_no_gap"].mean(),
        "ml_mean_no_wh_gap":    wide["no_wh_gap"].mean(),
        "ml_mean_no_wh_no_gap": wide["no_wh_no_gap"].mean(),
        "ml_se_wh_gap":         _se(wide["wh_gap"]),
        "ml_se_wh_no_gap":      _se(wide["wh_no_gap"]),
        "ml_se_no_wh_gap":      _se(wide["no_wh_gap"]),
        "ml_se_no_wh_no_gap":   _se(wide["no_wh_no_gap"]),
        # Masson & Loftus CI on the licensing interaction contrast
        "ml_interaction":          ml_int_mean,
        "ml_interaction_ci_low":   ml_int_mean - t_crit * ml_int_se,
        "ml_interaction_ci_high":  ml_int_mean + t_crit * ml_int_se,
    }
    
def run_checkpoint(ckpt_dir: Path, tokenizer, rows: List[dict],
                   device: torch.device,
                   batch_size: int = 32,
                   model=None) -> Tuple[List[dict], List[dict]]:
    """Returns (surprisal_rows, lme_rows)."""
    name = ckpt_dir.name
    step = int(name.split("-")[-1]) if "-" in name else 0
    print(f"\n=== {name} (step {step}) ===")
    loaded_here = False
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, local_files_only=True
        ).to(device)
        model.eval()
        loaded_here = True

    # Resolve post_gap for rows where it's empty (wh +gap: derive from sentence)
    items = []
    for row in rows:
        pre_gap  = row["pre_gap_text"]
        post_gap = row["post_gap_text"]
        if not post_gap:
            post_gap = row["sentence"][len(pre_gap):].strip()
        items.append((pre_gap, post_gap))

    surprisals = []
    for start in tqdm(range(0, len(items), batch_size), desc="  batches", leave=False):
        batch_items = items[start : start + batch_size]
        surprisals.extend(compute_gap_surprisals_batch(batch_items, model, tokenizer, device, batch_size))

    results = [
        {**row, "step": step,
         "surprisal_first": round(s_first, 6),
         "surprisal_mean":  round(s_mean,  6)}
        for row, (s_first, s_mean) in zip(rows, surprisals)
    ]

    if loaded_here:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_metrics(results, "surprisal_first", "surprisal_first")
    print_metrics(results, "surprisal_mean",  "surprisal_mean")

    df = pd.DataFrame(results)
    df["filler"] = df["filler"].astype(int)
    df["gap"]    = df["gap"].astype(int)
    lme_rows = []
    for col in ("surprisal_first", "surprisal_mean"):
        lme = _run_lme_one_step(step, df, surprisal_col=col)
        lme["surprisal_col"] = col
        lme_rows.append(lme)
        print(f"\n[LME {col}] step={step}  n_obs={lme['n_obs']}  n_items={lme['n_items']}  converged={lme['converged']}")
        print(f"  licensing_interaction = {lme['licensing_interaction']:+.4f}  "
              f"[{lme['interaction_ci_low']:+.4f}, {lme['interaction_ci_high']:+.4f}]  "
              f"p={lme['p_interaction']:.4f}")
        print(f"  ML interaction        = {lme['ml_interaction']:+.4f}  "
              f"[{lme['ml_interaction_ci_low']:+.4f}, {lme['ml_interaction_ci_high']:+.4f}]")

    return results, lme_rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute gap-site surprisal for filler-gap / wh-movement factorial CSVs."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--tokenizer-dir", type=Path, default=None,
                   help="HF tokenizer directory. Defaults to --checkpoint or --checkpoint-dir.")
    p.add_argument("--input-csv", type=Path, required=True,
                   help="filler_gap_factorial.csv or wh_movement_factorial.csv")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    with open(args.input_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    input_fields = list(rows[0].keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = args.tokenizer_dir or args.checkpoint or args.checkpoint_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)

    if args.checkpoint is not None:
        ckpts = [args.checkpoint]
    else:
        ckpts = sorted(
            [d for d in args.checkpoint_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint-* directories found in {args.checkpoint_dir}")

    all_results = []
    all_lme_rows = []
    for ckpt_file in ckpts:
        surprisal_rows, lme_rows = run_checkpoint(ckpt_file, tokenizer, rows, device, args.batch_size)
        all_results.extend(surprisal_rows)
        all_lme_rows.extend(lme_rows)

    out_fields = input_fields + ["step", "surprisal_first", "surprisal_mean"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to: {args.output_csv}")

    lme_csv = args.output_csv.with_stem(args.output_csv.stem + "_lme")
    lme_fields = ["step", "surprisal_col", "n_obs", "n_items", "converged",
                  "intercept", "filler_c", "gap_c", "filler_c:gap_c",
                  "licensing_interaction", "p_filler_c", "p_gap_c", "p_interaction",
                  "interaction_ci_low", "interaction_ci_high",
                  "ml_mean_wh_gap", "ml_mean_wh_no_gap", "ml_mean_no_wh_gap", "ml_mean_no_wh_no_gap",
                  "ml_se_wh_gap", "ml_se_wh_no_gap", "ml_se_no_wh_gap", "ml_se_no_wh_no_gap",
                  "ml_interaction", "ml_interaction_ci_low", "ml_interaction_ci_high"]
    with open(lme_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lme_fields)
        writer.writeheader()
        writer.writerows(all_lme_rows)
    print(f"LME results saved to: {lme_csv}")


if __name__ == "__main__":
    main()


def run(ckpt_dir: str, csv_path: str, out_csv_path: str, batch_size: int = 32, model=None, tokenizer=None) -> None:
    """Callable entry-point used by the training callback."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    input_fields = list(rows[0].keys())
    surprisal_rows, lme_rows = run_checkpoint(Path(ckpt_dir), tokenizer, rows, device, batch_size, model=model)

    out = Path(out_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_fields = input_fields + ["step", "surprisal_first", "surprisal_mean"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(surprisal_rows)

    lme_fields = ["step", "surprisal_col", "n_obs", "n_items", "converged",
                  "licensing_interaction", "p_interaction",
                  "ml_interaction", "ml_interaction_ci_low", "ml_interaction_ci_high"]
    with open(out.with_stem(out.stem + "_lme"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lme_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lme_rows)
