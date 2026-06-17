#!/usr/bin/env python3
"""Collect grid results into a CSV.

For each experiment directory under `out/pretrain/grid/*` it reads
`final/hyperparameters.yaml`, attempts to extract train/val loss from
`logs/tensorboard` events (if `tensorboard` is installed), and runs the
short-nested-outer evaluator to get overall and per-condition accuracy.

Usage (dry-run lists experiments):
  python scripts/collect_grid_results.py --dry-run

To actually evaluate (will load models and may need a GPU):
  python scripts/collect_grid_results.py --run-eval

Outputs: `results/grid_summary.csv`
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys

import yaml

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    EventAccumulator = None


def read_hyperparams(p: Path) -> dict:
    f = p / "final" / "hyperparameters.yaml"
    if not f.exists():
        return {}
    try:
        with f.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        # fallback: return raw text parsed as YAML if possible
        try:
            return yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:
            return {}


def find_event_files(p: Path):
    tb = p / "logs" / "tensorboard"
    if not tb.exists():
        return []
    return list(tb.glob("*event*"))


def extract_losses_from_events(event_files):
    if EventAccumulator is None or not event_files:
        return None, None
    # merge by reading the last scalar for tags containing 'loss'
    train_loss = None
    val_loss = None
    for ef in event_files:
        try:
            ea = EventAccumulator(str(ef))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for t in tags:
                if "train" in t and "loss" in t:
                    vals = ea.Scalars(t)
                    if vals:
                        train_loss = float(vals[-1].value)
                if ("val" in t or "valid" in t or "validation" in t) and "loss" in t:
                    vals = ea.Scalars(t)
                    if vals:
                        val_loss = float(vals[-1].value)
            # fallback: any tag with 'loss'
            if train_loss is None or val_loss is None:
                for t in tags:
                    if "loss" in t and all(x not in t for x in ("val","valid","validation")): 
                        vals = ea.Scalars(t)
                        if vals:
                            train_loss = train_loss or float(vals[-1].value)
                    if "loss" in t and any(x in t for x in ("val","valid","validation")):
                        vals = ea.Scalars(t)
                        if vals:
                            val_loss = val_loss or float(vals[-1].value)
        except Exception:
            continue
    return train_loss, val_loss


def fallback_parse_logs(p: Path):
    # look for any .txt under out dir and regex for 'loss' floats
    train_loss = None
    val_loss = None
    for txt in p.rglob("*.txt"):
        try:
            s = txt.read_text(errors="ignore")
        except Exception:
            continue
        # find patterns like 'train loss: 1.2345' or 'val loss: 1.23'
        m = re.search(r"train[^0-9\n]{0,10}loss[^0-9\n]{0,10}([0-9]+\.?[0-9]*)", s, re.I)
        if m:
            train_loss = float(m.group(1))
        m2 = re.search(r"(val|valid|validation)[^0-9\n]{0,10}loss[^0-9\n]{0,10}([0-9]+\.?[0-9]*)", s, re.I)
        if m2:
            val_loss = float(m2.group(2))
        if train_loss and val_loss:
            break
    return train_loss, val_loss


def collect(grid_root: Path, results_csv: Path, dry_run: bool, run_eval: bool, tokenizer_dir: Path, eval_data: Path, max_seq_length: int):
    rows = []
    dirs = sorted([d for d in grid_root.iterdir() if d.is_dir()])
    if dry_run:
        print(f"Found {len(dirs)} experiment dirs under {grid_root}")
    for d in dirs:
        name = d.name
        hp = read_hyperparams(d)
        event_files = find_event_files(d)
        train_loss = val_loss = None
        if event_files:
            train_loss, val_loss = extract_losses_from_events(event_files)
        if train_loss is None and val_loss is None:
            t1, v1 = fallback_parse_logs(d)
            train_loss = train_loss or t1
            val_loss = val_loss or v1

        step = ""
        overall = ""
        per_cat = {}
        if run_eval:
            # import eval functions lazily to avoid heavy imports in dry-run
            try:
                from eval_short_nested_outer import load_examples, compute_one_checkpoint
            except Exception as e:
                # fallback: try loading the file by path (useful when cwd/import path differs)
                try:
                    import importlib.util
                    repo_root = Path(__file__).resolve().parent.parent
                    alt_path = repo_root / "eval_short_nested_outer.py"
                    if alt_path.exists():
                        spec = importlib.util.spec_from_file_location("eval_short_nested_outer", str(alt_path))
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        load_examples = module.load_examples
                        compute_one_checkpoint = module.compute_one_checkpoint
                    else:
                        raise FileNotFoundError(f"Fallback file not found: {alt_path}")
                except Exception as e2:
                    print("Could not import eval_short_nested_outer:", e, file=sys.stderr)
                    print("Fallback import failed:", e2, file=sys.stderr)
                    run_eval = False
            else:
                examples = load_examples(str(eval_data))
                ckpt = d / "final" / "lit_model.pth"
                if ckpt.exists():
                    try:
                        step, overall, per_cat = compute_one_checkpoint(ckpt, tokenizer_dir, examples, max_seq_length)
                    except Exception as e:
                        print(f"Evaluation failed for {d}: {e}", file=sys.stderr)
                else:
                    print(f"Checkpoint not found for {d}: {ckpt}", file=sys.stderr)

        # Extract only grid hyperparameters (lr, warmup, batch, weight_decay, max_norm, precision)
        try:
            opt_init = hp.get("optimizer", {}).get("init_args", {}) if isinstance(hp, dict) else {}
        except Exception:
            opt_init = {}
        try:
            train_cfg = hp.get("train", {}) if isinstance(hp, dict) else {}
        except Exception:
            train_cfg = {}

        lr = opt_init.get("lr")
        weight_decay = opt_init.get("weight_decay")
        lr_warmup = train_cfg.get("lr_warmup_steps")
        global_batch = train_cfg.get("global_batch_size")
        micro_batch = train_cfg.get("micro_batch_size")
        max_norm = train_cfg.get("max_norm")
        precision = hp.get("precision") if isinstance(hp, dict) else None

        row = {
            "exp": name,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_warmup_steps": lr_warmup,
            "global_batch_size": global_batch,
            "micro_batch_size": micro_batch,
            "max_norm": max_norm,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "step": step,
            "overall_acc": overall,
            "per_category": json.dumps(per_cat, ensure_ascii=False),
        }
        rows.append(row)
        if dry_run:
            print(json.dumps(row, indent=2, default=str))

    if not dry_run:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "exp",
                "out_dir",
                "lr",
                "weight_decay",
                "lr_warmup_steps",
                "global_batch_size",
                "micro_batch_size",
                "max_norm",
                "precision",
                "train_loss",
                "val_loss",
                "step",
                "overall_acc",
                "per_category",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote summary to {results_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid-root", type=Path, default=Path("out/pretrain/grid"))
    p.add_argument("--output", type=Path, default=Path("results/grid_summary.csv"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-eval", action="store_true", help="Run eval_short_nested_outer on each final checkpoint (may need GPU)")
    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--eval-data", type=Path, default=Path("short_nested_outer_english.json"))
    p.add_argument("--max-seq-length", type=int, default=0)
    args = p.parse_args()

    collect(args.grid_root, args.output, args.dry_run, args.run_eval, args.tokenizer_dir, args.eval_data, args.max_seq_length)


if __name__ == "__main__":
    main()
