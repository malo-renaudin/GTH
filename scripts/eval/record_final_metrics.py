#!/usr/bin/env python3
"""Extract hyperparameters from a config, compute train/valid perplexity on final checkpoint, and append CSV."""
import argparse
from pathlib import Path
import csv
import yaml
import subprocess
import shlex
import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--results-csv", type=Path, default=Path("configs/grid/results.csv"))
    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--train-file", type=Path, default=Path("data/english_data/train.txt"))
    p.add_argument("--valid-file", type=Path, default=Path("data/english_data/valid.txt"))
    p.add_argument("--max-sents", type=int, default=500)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def find_last_checkpoint(out_dir: Path):
    if not out_dir.exists():
        raise FileNotFoundError(f"Out dir not found: {out_dir}")
    steps = sorted([p for p in out_dir.glob("step-*") if p.is_dir()], key=lambda p: int(p.name.split("-")[1]))
    if not steps:
        # maybe lit_model.pth directly in out_dir
        ck = out_dir / "lit_model.pth"
        if ck.exists():
            return ck
        raise FileNotFoundError(f"No step-* dirs or lit_model.pth in {out_dir}")
    last = steps[-1] / "lit_model.pth"
    if not last.exists():
        raise FileNotFoundError(f"No lit_model.pth in {last.parent}")
    return last


def read_hparams(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train = cfg.get("train", {})
    opt = cfg.get("optimizer", {}).get("init_args", {})
    return {
        "lr": opt.get("lr"),
        "weight_decay": opt.get("weight_decay"),
        "global_batch_size": train.get("global_batch_size"),
        "micro_batch_size": train.get("micro_batch_size"),
        "lr_warmup_steps": train.get("lr_warmup_steps"),
        "max_norm": train.get("max_norm"),
        "max_tokens": train.get("max_tokens"),
        "precision": cfg.get("precision"),
    }


def run_compute_ppl(ckpt, tokenizer_dir, data_file, max_sents, device):
    cmd = f"python3 scripts/compute_ppl.py --checkpoint {shlex.quote(str(ckpt))} --tokenizer-dir {shlex.quote(str(tokenizer_dir))} --data-file {shlex.quote(str(data_file))} --max-sents {max_sents} --device {shlex.quote(device)}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"compute_ppl failed: {res.stderr}\ncmd={cmd}")
    return float(res.stdout.strip())


def main():
    args = parse_args()
    hparams = read_hparams(args.config)
    ckpt = find_last_checkpoint(args.out_dir)

    train_ppl = run_compute_ppl(ckpt, args.tokenizer_dir, args.train_file, args.max_sents, args.device)
    valid_ppl = run_compute_ppl(ckpt, args.tokenizer_dir, args.valid_file, args.max_sents, args.device)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp", "config", "out_dir", "lr", "weight_decay", "global_batch_size", "micro_batch_size", "lr_warmup_steps", "max_norm", "max_tokens", "precision", "train_ppl", "valid_ppl"]
    write_header = not args.results_csv.exists()
    with args.results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            datetime.datetime.utcnow().isoformat(),
            str(args.config),
            str(args.out_dir),
            hparams.get("lr"),
            hparams.get("weight_decay"),
            hparams.get("global_batch_size"),
            hparams.get("micro_batch_size"),
            hparams.get("lr_warmup_steps"),
            hparams.get("max_norm"),
            hparams.get("max_tokens"),
            hparams.get("precision"),
            train_ppl,
            valid_ppl,
        ])
    print(f"Appended results to {args.results_csv}: train_ppl={train_ppl:.3f}, valid_ppl={valid_ppl:.3f}")


if __name__ == "__main__":
    main()
