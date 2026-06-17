#!/usr/bin/env python3
"""Generate a grid of training YAML configs from a base YAML.

Writes configs to `configs/grid/exp-{idx}.yaml` and a manifest `configs/grid/manifest.tsv`
with lines: <config_path>\t<out_dir>. Indexing starts at 1 to match SLURM_ARRAY_TASK_ID.

Usage:
  python scripts/make_grid_configs.py --base babylm_gpt2_100m.yaml
"""
import argparse
import itertools
import os
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("babylm_gpt2_100m.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("configs/grid"))
    p.add_argument("--max-tokens", type=int, default=100_000_000)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    base = args.base
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    with base.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # grid
    lrs = [1e-5, 2e-5, 5e-5]
    warmups = [500, 2116]
    global_batch_sizes = [8, 16]
    weight_decays = [0.0, 0.01]
    clip_norms = [0.5, 1.0]
    precisions = [cfg.get("precision", "bf16-mixed")]

    combos = list(itertools.product(lrs, warmups, global_batch_sizes, weight_decays, clip_norms, precisions))

    manifest_lines = []
    for idx, (lr, warmup, gbs, wd, clip, prec) in enumerate(combos, start=1):
        new = dict(cfg)  # shallow copy; we'll mutate nested dicts inplace

        # Mutate optimizer lr & weight_decay
        opt = new.setdefault("optimizer", {})
        init_args = opt.setdefault("init_args", {})
        init_args["lr"] = float(lr)
        init_args["weight_decay"] = float(wd)

        # train params
        train = new.setdefault("train", {})
        train["lr_warmup_steps"] = int(warmup)
        train["global_batch_size"] = int(gbs)
        train["micro_batch_size"] = int(gbs)
        train["max_tokens"] = int(args.max_tokens)
        train["max_norm"] = float(clip)
        # set huge save interval to avoid intermediate saves; job will cleanup anyway
        train["save_interval"] = 10**9

        new["precision"] = prec
        new["resume"] = False

        out_dir = Path(new.get("out_dir", "out/pretrain/grid"))
        # unique out_dir per experiment
        exp_out = out_dir.parent / f"grid/{out_dir.name}_exp{idx:03d}"
        new["out_dir"] = str(exp_out)

        cfg_path = out_root / f"exp-{idx:03d}.yaml"
        if args.dry_run:
            print(f"[{idx}] lr={lr} warmup={warmup} gbs={gbs} wd={wd} clip={clip} prec={prec} -> {cfg_path} -> {exp_out}")
        else:
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(new, f, default_flow_style=False, sort_keys=False)
        manifest_lines.append(f"{cfg_path}\t{exp_out}")

    manifest_file = out_root / "manifest.tsv"
    if args.dry_run:
        print(f"Would write manifest to {manifest_file} with {len(manifest_lines)} entries")
    else:
        with manifest_file.open("w", encoding="utf-8") as f:
            f.write("#config\tout_dir\n")
            for line in manifest_lines:
                f.write(line + "\n")
        print(f"Wrote {len(manifest_lines)} configs to {out_root} and manifest {manifest_file}")


if __name__ == "__main__":
    main()
