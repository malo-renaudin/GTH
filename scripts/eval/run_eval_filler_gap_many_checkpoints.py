#!/usr/bin/env python3
"""
Run eval_filler_gap.py on many checkpoints.

Example:
  python scripts/eval/run_eval_filler_gap_many_checkpoints.py \
    --checkpoints results/baseline_wh_augmented \
    --input-csv generate_simple_datasets/generated_simple_datasets/filler_gap_wh.csv \
    --out-dir results/baseline_wh_augmented/eval_filler_gap \
    --batch-size 32
"""
import argparse
import subprocess
import os
from pathlib import Path

def _looks_like_hf_checkpoint(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_tokenizer = any((path / name).exists() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "spiece.model"))
    has_model = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists() or any(str(p).endswith(".bin") for p in path.glob("pytorch_model-*.bin"))
    has_config = (path / "config.json").exists()
    return has_model and (has_tokenizer or has_config)

def expand_candidate(candidate: str):
    p = Path(candidate)
    if p.exists() and p.is_dir():
        if _looks_like_hf_checkpoint(p):
            return [str(p)]
        # otherwise look for checkpoint-like subdirs
        found = [str(child) for child in sorted(p.iterdir()) if child.is_dir() and _looks_like_hf_checkpoint(child)]
        return found
    # treat as HF hub id or file path / single checkpoint
    return [candidate]

def run_eval(checkpoint: str, input_csv: str, out_dir: str, tokenizer_dir: str = None, batch_size: int = None, extra_args=None) -> bool:
    ckpt_name = Path(checkpoint).name
    out_path = Path(out_dir) / f"{ckpt_name}_filler_gap.csv"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", os.path.join("scripts", "eval", "eval_filler_gap.py"),
        "--checkpoint", checkpoint,
        "--input-csv", input_csv,
        "--output-csv", str(out_path),
    ]
    if tokenizer_dir:
        cmd += ["--tokenizer-dir", tokenizer_dir]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if extra_args:
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
    p.add_argument("--checkpoints", nargs="+", help="List of checkpoint dirs or hub ids")
    p.add_argument("--checkpoints-dir", help="Directory containing checkpoint subfolders")
    p.add_argument("--input-csv", required=True, help="Input factorial CSV (filler_gap_wh.csv)")
    p.add_argument("--out-dir", required=True, help="Directory to save per-checkpoint outputs")
    p.add_argument("--tokenizer-dir", default=None, help="Optional tokenizer dir to pass to eval")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=None, help="Extra args passed to eval_filler_gap.py")
    args = p.parse_args()

    candidates = []
    if args.checkpoints:
        for c in args.checkpoints:
            candidates.extend(expand_candidate(c))
    if args.checkpoints_dir:
        d = Path(args.checkpoints_dir)
        if d.exists():
            for child in sorted(d.iterdir()):
                if child.is_dir():
                    candidates.extend(expand_candidate(str(child)))
    # de-duplicate while preserving order
    seen = set()
    ckpts = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ckpts.append(c)
    if not ckpts:
        print("No checkpoints found. Provide --checkpoints or --checkpoints-dir")
        return

    for ckpt in ckpts:
        ok = run_eval(ckpt, args.input_csv, args.out_dir, tokenizer_dir=args.tokenizer_dir, batch_size=args.batch_size, extra_args=args.extra)
        if not ok:
            print(f"Stopping due to failure on {ckpt}")
            break

if __name__ == "__main__":
    main()