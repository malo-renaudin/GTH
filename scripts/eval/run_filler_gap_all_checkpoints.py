"""
Run filler-gap evaluations across **all downloaded checkpoints** of a model
and write combined CSVs — one surprisal file and one LME file per eval type.

Layout expected on disk
-----------------------
    <model-dir>/
        main/            <- or any branch / revision name
        checkpoint-10000/
        checkpoint-20000/
        ...

Usage
-----
    python scripts/eval/run_filler_gap_all_checkpoints.py \
        --model-dir babylm-baseline-100m-gpt2 \
        --output-dir results/babylm_all_ckpts

    # use a specific tokenizer dir (defaults to first checkpoint found)
    python scripts/eval/run_filler_gap_all_checkpoints.py \
        --model-dir babylm-baseline-100m-gpt2 \
        --tokenizer babylm-baseline-100m-gpt2/main \
        --output-dir results/babylm_all_ckpts

    # skip OOV evals
    python scripts/eval/run_filler_gap_all_checkpoints.py \
        --model-dir babylm-baseline-100m-gpt2 \
        --filler-gap-orc-oov none \
        --filler-gap-wh-oov none \
        --output-dir results/babylm_all_ckpts_iv_only
"""

import argparse
import csv
import importlib.util
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
    if s.lower() == "none":
        return None
    return Path(s)


def _step_key(ckpt_dir: Path) -> int:
    """Sort key: extract numeric step from dir name, or -1 for non-numeric."""
    name = ckpt_dir.name
    try:
        return int(name.split("-")[-1]) if "-" in name else -1
    except ValueError:
        return -1


p = argparse.ArgumentParser(
    description="Batch filler-gap eval over all checkpoints in a directory."
)
p.add_argument("--model-dir", required=True, type=Path,
               help="Parent directory whose sub-directories are individual checkpoints.")
p.add_argument("--tokenizer", default=None, type=Path,
               help="Path to a tokenizer directory. Defaults to the first checkpoint found.")
p.add_argument("--output-dir", required=True, type=Path,
               help="Where to write combined CSVs.")
p.add_argument("--batch-size", type=int, default=32)

p.add_argument("--filler-gap-orc-iv",  type=_none_or_path, default=Path("eval_data/orc_factorial_iv.csv"))
p.add_argument("--filler-gap-orc-oov", type=_none_or_path, default=Path("eval_data/orc_factorial_oov.csv"))
p.add_argument("--filler-gap-wh-iv",   type=_none_or_path, default=Path("eval_data/wh_factorial_iv.csv"))
p.add_argument("--filler-gap-wh-oov",  type=_none_or_path, default=Path("eval_data/embedded_wh_fg_test.csv"))

args = p.parse_args()

# ---------------------------------------------------------------------------
# Discover and sort checkpoints
# ---------------------------------------------------------------------------

all_ckpts = sorted(
    [d for d in args.model_dir.iterdir() if d.is_dir()],
    key=_step_key,
)

if not all_ckpts:
    raise SystemExit(f"No subdirectories found in {args.model_dir}")

print(f"Found {len(all_ckpts)} checkpoint(s):")
for d in all_ckpts:
    print(f"  {d.name}  (step {_step_key(d)})")

# ---------------------------------------------------------------------------
# Load tokenizer once
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

tok_dir = args.tokenizer or all_ckpts[0]
print(f"Loading tokenizer from: {tok_dir}")
tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# Pre-load eval CSV rows (once per label)
# ---------------------------------------------------------------------------

EVALS = [
    ("orc_iv",  args.filler_gap_orc_iv),
    ("orc_oov", args.filler_gap_orc_oov),
    ("wh_iv",   args.filler_gap_wh_iv),
    ("wh_oov",  args.filler_gap_wh_oov),
]

eval_rows: dict[str, list[dict]] = {}
for label, csv_path in EVALS:
    if csv_path is None:
        print(f"[{label}] skipped (path is None)")
        continue
    if not csv_path.exists():
        print(f"[{label}] skipped — file not found: {csv_path}")
        continue
    with open(csv_path, newline="") as f:
        eval_rows[label] = list(csv.DictReader(f))
    print(f"[{label}] loaded {len(eval_rows[label])} rows from {csv_path}")

if not eval_rows:
    raise SystemExit("No eval CSV files could be loaded. Exiting.")

# Collect output field names per label (input fields + step + surprisal cols)
input_fields: dict[str, list[str]] = {
    label: list(rows[0].keys()) for label, rows in eval_rows.items()
}

LME_FIELDS = [
    "revision", "step", "surprisal_col", "n_obs", "n_items", "converged",
    "intercept", "filler_c", "gap_c", "filler_c:gap_c",
    "licensing_interaction", "p_filler_c", "p_gap_c", "p_interaction",
    "interaction_ci_low", "interaction_ci_high",
    "ml_mean_wh_gap", "ml_mean_wh_no_gap", "ml_mean_no_wh_gap", "ml_mean_no_wh_no_gap",
    "ml_se_wh_gap", "ml_se_wh_no_gap", "ml_se_no_wh_gap", "ml_se_no_wh_no_gap",
    "ml_interaction", "ml_interaction_ci_low", "ml_interaction_ci_high",
]

# ---------------------------------------------------------------------------
# Accumulators: label -> list of rows
# ---------------------------------------------------------------------------

all_surprisal: dict[str, list[dict]] = {label: [] for label in eval_rows}
all_lme:       dict[str, list[dict]] = {label: [] for label in eval_rows}

args.output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Main loop over checkpoints
# ---------------------------------------------------------------------------

for ckpt_dir in all_ckpts:
    revision = ckpt_dir.name
    print(f"\n{'='*60}")
    print(f"Checkpoint: {revision}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir, local_files_only=True
    ).to(device)
    model.eval()

    for label, rows in eval_rows.items():
        surprisal_rows, lme_rows = eval_filler_gap.run_checkpoint(
            ckpt_dir, tokenizer, rows, device,
            batch_size=args.batch_size,
            model=model,
        )
        # Tag every row with the revision name
        for r in surprisal_rows:
            r["revision"] = revision
        for r in lme_rows:
            r["revision"] = revision

        all_surprisal[label].extend(surprisal_rows)
        all_lme[label].extend(lme_rows)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Write combined CSVs
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("Writing combined CSVs ...")

for label in eval_rows:
    # Surprisal CSV
    surp_fields = ["revision"] + input_fields[label] + ["step", "surprisal_first", "surprisal_mean"]
    surp_path = args.output_dir / f"filler_gap_{label}_surprisals.csv"
    with open(surp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=surp_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_surprisal[label])
    print(f"  [{label}] surprisals → {surp_path}  ({len(all_surprisal[label])} rows)")

    # LME CSV
    lme_path = args.output_dir / f"filler_gap_{label}_lme.csv"
    with open(lme_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LME_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_lme[label])
    print(f"  [{label}] LME       → {lme_path}  ({len(all_lme[label])} rows)")

print(f"\nAll done. Results in: {args.output_dir}")
