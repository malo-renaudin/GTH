"""
Evaluate any HuggingFace checkpoint on the filler-gap paradigms.

Usage
-----
    # local checkpoint
    python scripts/eval/run_filler_gap_eval.py \
        --model path/to/checkpoint \
        --output-dir results/my_model

    # HuggingFace Hub model
    python scripts/eval/run_filler_gap_eval.py \
        --model gpt2 \
        --output-dir results/gpt2

    # skip a specific eval
    python scripts/eval/run_filler_gap_eval.py \
        --model gpt2 \
        --filler-gap-orc-oov none \
        --filler-gap-wh-oov none \
        --output-dir results/gpt2_iv_only
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Import eval_filler_gap from the same directory as this script.
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).resolve().parent


def _import_eval(name: str):
    path = _EVAL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_filler_gap = _import_eval("eval_filler_gap")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _none_or_path(s: str) -> Path | None:
    """Allow passing 'none' on the CLI to skip an eval."""
    if s.lower() == "none":
        return None
    return Path(s)


p = argparse.ArgumentParser(
    description="Run filler-gap evaluations on any HuggingFace checkpoint."
)
p.add_argument("--model", required=True,
               help="HF Hub model name (e.g. 'gpt2') or path to a local checkpoint directory.")
p.add_argument("--tokenizer", default=None,
               help="HF tokenizer name or path. Defaults to --model.")
p.add_argument("--output-dir", required=True, type=Path,
               help="Directory where result CSVs will be written.")
p.add_argument("--batch-size", type=int, default=32)

# Filler-gap data files — pass 'none' to skip any single eval.
p.add_argument("--filler-gap-orc-iv",  type=_none_or_path, default=Path("eval_data/orc_factorial_iv.csv"))
p.add_argument("--filler-gap-orc-oov", type=_none_or_path, default=Path("eval_data/orc_factorial_oov.csv"))
p.add_argument("--filler-gap-wh-iv",   type=_none_or_path, default=Path("eval_data/wh_factorial_iv.csv"))
p.add_argument("--filler-gap-wh-oov",  type=_none_or_path, default=Path("eval_data/wh_factorial_oov.csv"))

args = p.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tok_name = args.tokenizer or args.model
print(f"Loading tokenizer from: {tok_name}")
tokenizer = AutoTokenizer.from_pretrained(tok_name, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from: {args.model}")
model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).to(device)
model.eval()

args.output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: run one filler-gap CSV and write output
# ---------------------------------------------------------------------------

LME_FIELDS = [
    "surprisal_col", "n_obs", "n_items", "converged",
    "intercept", "filler_c", "gap_c", "filler_c:gap_c",
    "licensing_interaction", "p_filler_c", "p_gap_c", "p_interaction",
    "interaction_ci_low", "interaction_ci_high",
    "ml_mean_wh_gap", "ml_mean_wh_no_gap", "ml_mean_no_wh_gap", "ml_mean_no_wh_no_gap",
    "ml_se_wh_gap", "ml_se_wh_no_gap", "ml_se_no_wh_gap", "ml_se_no_wh_no_gap",
    "ml_interaction", "ml_interaction_ci_low", "ml_interaction_ci_high",
]


def run_eval(label: str, csv_path: Path) -> None:
    """Run filler-gap eval for a single CSV file and write results."""
    if csv_path is None:
        print(f"\n[{label}] skipped (path is None)")
        return
    if not csv_path.exists():
        print(f"\n[{label}] skipped — file not found: {csv_path}")
        return

    print(f"\n[{label}] reading {csv_path} ...")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    input_fields = list(rows[0].keys())

    # Use step=0 for a single non-checkpoint model; run_checkpoint extracts the
    # step from the directory name (returns 0 if the name has no "-" in it).
    ckpt_path = Path(args.model)
    surprisal_rows, lme_rows = eval_filler_gap.run_checkpoint(
        ckpt_path, tokenizer, rows, device,
        batch_size=args.batch_size,
        model=model,   # pass model in so it is not re-loaded (and no local_files_only check)
    )

    # Surprisal CSV
    out_csv = args.output_dir / f"filler_gap_{label}.csv"
    out_fields = input_fields + ["step", "surprisal_first", "surprisal_mean"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(surprisal_rows)
    print(f"  Surprisals → {out_csv}")

    # LME CSV
    lme_csv = args.output_dir / f"filler_gap_{label}_lme.csv"
    with open(lme_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LME_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(lme_rows)
    print(f"  LME        → {lme_csv}")


# ---------------------------------------------------------------------------
# Run all four evals
# ---------------------------------------------------------------------------

run_eval("orc_iv",  args.filler_gap_orc_iv)
run_eval("orc_oov", args.filler_gap_orc_oov)
run_eval("wh_iv",   args.filler_gap_wh_iv)
run_eval("wh_oov",  args.filler_gap_wh_oov)

# Cleanup
del model
if device.type == "cuda":
    torch.cuda.empty_cache()

print(f"\nAll done. Results in: {args.output_dir}")
