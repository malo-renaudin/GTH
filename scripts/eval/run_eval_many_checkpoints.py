#!/usr/bin/env python3
"""
Run `eval_probability_masses.py` on multiple checkpoints and save outputs.

Usage:
  python scripts/eval/run_eval_many_checkpoints.py --checkpoints ckpt1 ckpt2 ... \
    --orc_test data/orc_transitivity.csv --wh_test data/orc_semantic_distractors.csv \
    --out-dir results/eval_masses --device cpu

The runner will invoke the eval script for each checkpoint and save a per-checkpoint
CSV named <checkpoint_name>_prob_masses.csv under --out-dir. If a checkpoint is a
path, its basename is used for filenames.
"""

import argparse
import subprocess
import os
from pathlib import Path


def run_eval(checkpoint: str, orc_test: str, wh_test: str, out_dir: str, device: str = None, extra_args=None):
    extra_args = extra_args or []
    ckpt_name = Path(checkpoint).name.replace("/", "_")
    out_path = Path(out_dir) / f"{ckpt_name}_prob_masses.csv"
    cmd = [
        "python",
        os.path.join("scripts", "eval", "eval_probability_masses.py"),
        "--checkpoint",
        checkpoint,
        "--orc_test",
        orc_test,
        "--wh_test",
        wh_test,
        "--out",
        str(out_path),
    ]
    if device:
        cmd += ["--device", device]
    cmd += extra_args

    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Eval failed for checkpoint {checkpoint}: {e}")
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=False, help="List of HF checkpoint ids or local paths")
    p.add_argument("--checkpoints-dir", required=False, help="Directory containing multiple HF checkpoint folders")
    p.add_argument("--orc_test", required=True)
    p.add_argument("--wh_test", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=None, help="Extra args passed to eval script")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # build the list of checkpoints from either --checkpoints or --checkpoints-dir
    ckpt_list = []
    if args.checkpoints:
        ckpt_list.extend(args.checkpoints)
    if args.checkpoints_dir:
        d = Path(args.checkpoints_dir)
        if not d.exists():
            print(f"Checkpoints directory does not exist: {d}")
            return
        # include immediate subdirectories and files (useful for HF-style folders or tarballs)
        for child in sorted(d.iterdir()):
            # skip hidden files
            if child.name.startswith("."):
                continue
            ckpt_list.append(str(child))

    if not ckpt_list:
        print("No checkpoints found. Provide --checkpoints or --checkpoints-dir")
        return

    for ckpt in ckpt_list:
        ok = run_eval(ckpt, args.orc_test, args.wh_test, args.out_dir, device=args.device, extra_args=args.extra)
        if not ok:
            print(f"Skipping remaining checkpoints due to failure on {ckpt}")
            break

    print("Done.")


if __name__ == "__main__":
    main()
