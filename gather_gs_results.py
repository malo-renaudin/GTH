#!/usr/bin/env python3

import json
import re
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results/grid_search")

rows = []

for run_dir in RESULTS_DIR.iterdir():

    if not run_dir.is_dir():
        continue

    metrics_file = run_dir / "train_metrics.json"

    if not metrics_file.exists():
        continue

    with open(metrics_file) as f:
        metrics = json.load(f)

    row = {
        "run_dir": run_dir.name,
        "train_loss": metrics.get("train_loss"),
        "val_loss": metrics.get("val_loss"),
    }

    # ---------------------------------------
    # Extract hyperparameters from directory name
    # ---------------------------------------
    name = run_dir.name

    # Examples:
    # lr5e-5_wd0.01_warm0.1_123456_0
    # lr0.0001_wd0.1_epochs5_123456_1

    patterns = {
        "learning_rate": r"lr([0-9eE\.-]+)",
        "weight_decay": r"wd([0-9eE\.-]+)",
        "warmup_ratio": r"warm([0-9eE\.-]+)",
        "epochs": r"epochs([0-9]+)",
        "batch_size": r"bs([0-9]+)",
    }

    for col, pattern in patterns.items():
        m = re.search(pattern, name)
        if m:
            value = m.group(1)

            try:
                value = float(value)
            except ValueError:
                pass

            row[col] = value

    rows.append(row)

df = pd.DataFrame(rows)

if "val_loss" in df.columns:
    df = df.sort_values("val_loss")

df.to_csv("grid_search_results.csv", index=False)

print(df.head(20))
print(f"\nSaved {len(df)} runs.")