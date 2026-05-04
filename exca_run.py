#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pydantic
import exca as xk


REPO_ROOT = Path(__file__).resolve().parent
EVAL_FIELDS = [
    "step",
    "structure",
    "target_label",
    "comparator_label",
    "target_mass",
    "target_head_mass",
    "comparator_mass",
    "target_minus_comparator",
    "roi_count",
    "checkpoint",
]


def list_step_checkpoints(out_dir: Path) -> List[Path]:
    ckpts = []
    for p in out_dir.glob("step-*/lit_model.pth"):
        m = re.match(r"step-(\d+)", p.parent.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda t: t[0])
    return [p for _, p in ckpts]


def step_from_checkpoint(ckpt: Path) -> int:
    return int(ckpt.parent.name.split("-")[1])


def out_dir_from_config(config_path: Path) -> Path:
    """Read litgpt out_dir from a YAML config without extra dependencies."""
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^out_dir:\s*(.+?)\s*$", text)
    if match is None:
        raise ValueError(f"Could not find 'out_dir:' in config file {config_path}")
    raw = match.group(1).strip().strip("\"'")
    out_dir = Path(raw)
    if not out_dir.is_absolute():
        out_dir = config_path.parent / out_dir
    return out_dir


class EvalCheckpointTask(pydantic.BaseModel):
    checkpoint: Path
    structure: str  # "orc" or "wh"
    sentences_file: Path
    tokenizer_dir: Path = Path("checkpoints/gpt2")
    max_seq_length: int = 0
    infra: xk.TaskInfra = xk.TaskInfra()

    @infra.apply
    def process(self) -> Dict:
        """Evaluate a single checkpoint using eval_test.py in single-checkpoint mode."""
        checkpoint = self.checkpoint.resolve()
        sentences_file = self.sentences_file.resolve()
        tokenizer_dir = self.tokenizer_dir.resolve()
        step = step_from_checkpoint(checkpoint)
        temp_csv = (REPO_ROOT / f"results/eval_tmp_step_{step}_{self.structure}.csv").resolve()
        temp_csv.parent.mkdir(parents=True, exist_ok=True)

        # Call eval_test.py with single-checkpoint mode
        cmd = [
            "python",
            str((REPO_ROOT / "eval_test.py").resolve()),
            "--checkpoint",
            str(checkpoint),
            "--sentences-file",
            str(sentences_file),
            "--structure",
            self.structure,
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--max-seq-length",
            str(self.max_seq_length),
            "--result-name",
            str(temp_csv),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(
                "eval_test.py failed for checkpoint "
                f"{checkpoint}\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        print(f"[eval_test] step {step}:\n{result.stdout}")

        # Parse the CSV output
        row = None
        if temp_csv.exists():
            with temp_csv.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_dict in reader:
                    row = row_dict
                    break

        if row is None:
            raise RuntimeError(f"eval_test.py produced no output for checkpoint {checkpoint}")

        return {
            "step": step,
            "structure": row["structure"],
            "target_label": row["target_label"],
            "comparator_label": row["comparator_label"],
            "target_mass": float(row["target_mass"]),
            "target_head_mass": float(row["target_head_mass"]),
            "comparator_mass": float(row["comparator_mass"]),
            "target_minus_comparator": float(row["target_minus_comparator"]),
            "roi_count": int(row["roi_count"]),
            "checkpoint": str(self.checkpoint),
        }


class TrainTask(pydantic.BaseModel):
    config: Path = Path("gpt.yaml")
    out_dir: Path = Path("out/pretrain/gpt")
    infra: xk.TaskInfra = xk.TaskInfra()

    @infra.apply
    def process(self) -> Dict:
        cmd = [
            "litgpt",
            "pretrain",
            "--config",
            str(self.config),
        ]
        proc = subprocess.Popen(cmd)
        return {"pid": proc.pid, "out_dir": str(self.out_dir)}


class TrainAndEvalTask(pydantic.BaseModel):
    config: Path = Path("gpt.yaml")
    test_file_orc: Path
    test_file_wh: Path
    csv_out: Path = Path("results/live_eval.csv")
    poll_seconds: int = 30
    infra: xk.TaskInfra = xk.TaskInfra()

    @infra.apply
    def process(self) -> Dict:
        out_dir = out_dir_from_config(self.config)
        print(f"[train] checkpoint directory resolved from config: {out_dir}")
        self.csv_out.parent.mkdir(parents=True, exist_ok=True)

        # Header CSV
        if not self.csv_out.exists():
            with self.csv_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=EVAL_FIELDS,
                )
                w.writeheader()

        # Start training
        cmd = [
            "litgpt",
            "pretrain",
            "--config",
            str(self.config),
        ]
        proc = subprocess.Popen(cmd)

        evaluated_steps = set()
        rows_written = 0

        while True:
            ckpts = list_step_checkpoints(out_dir)
            new_ckpts = [c for c in ckpts if step_from_checkpoint(c) not in evaluated_steps]

            for ckpt in new_ckpts:
                step = step_from_checkpoint(ckpt)

                # Evaluate both ORC and WH structures for this checkpoint
                for structure, test_file in [("orc", self.test_file_orc), ("wh", self.test_file_wh)]:
                    row = EvalCheckpointTask(
                        checkpoint=ckpt,
                        structure=structure,
                        sentences_file=test_file,
                        infra={
                            "folder": self.infra.folder / "eval_ckpt",
                            "cluster": None,
                            "mode": "force",
                        },
                    ).process()

                    with self.csv_out.open("a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(
                            f,
                            fieldnames=EVAL_FIELDS,
                        )
                        w.writerow(row)

                    rows_written += 1
                    print(
                        f"[eval] step={step} structure={structure} "
                        f"{row['target_label']}={row['target_mass']:.6f} "
                        f"head={row['target_head_mass']:.6f} "
                        f"{row['comparator_label']}={row['comparator_mass']:.6f} "
                        f"roi_count={row['roi_count']}"
                    )

                # Mark this step as evaluated after both structures are done
                evaluated_steps.add(step)

            # break condition: training ended and no new ckpt left
            ret = proc.poll()
            if ret is not None:
                ckpts_after = list_step_checkpoints(out_dir)
                remaining = [c for c in ckpts_after if step_from_checkpoint(c) not in evaluated_steps]
                if not remaining:
                    break

            time.sleep(self.poll_seconds)

        return {
            "csv": str(self.csv_out),
            "out_dir": str(out_dir),
            "rows_written": rows_written,
            "evaluated_steps": sorted(evaluated_steps),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train + evaluate checkpoints (ORC and WH) with EXCA.")
    parser.add_argument("--config", type=Path, default=Path("gpt_baseline.yaml"))
    parser.add_argument("--test-file-orc", type=Path, default=Path("data/valid_orc7_not_in_train_orc_6_72.txt"))
    parser.add_argument("--test-file-wh", type=Path, default=Path("data/valid_wh5_not_in_wh_7_20.txt"))
    parser.add_argument("--csv-out", type=Path, default=Path("results/eval_baseline.csv"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--cache-folder", type=Path, default=Path("results/exca_cache/train_eval_baseline"))

    # EXCA/submitit infrastructure controls
    parser.add_argument("--cluster", choices=["slurm", "local", "auto", "debug"], default="slurm")
    parser.add_argument("--infra-mode", choices=["cached", "retry", "force", "read-only"], default="retry")
    parser.add_argument("--job-name", type=str, default="gpt_baseline")
    parser.add_argument("--logs", type=str, default="outputs/gpt/%j")
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--timeout-min", type=int, default=120)
    parser.add_argument("--slurm-account", type=str, default="ywa@h100")
    parser.add_argument("--slurm-partition", type=str, default="gpu_p6")
    parser.add_argument("--slurm-qos", type=str, default="qos_gpu_h100-dev")
    parser.add_argument("--slurm-constraint", type=str, default="h100")
    parser.add_argument("--mail-user", type=str, default="malorenaudin1@gmail.com")
    return parser.parse_args()


def build_main_infra(args: argparse.Namespace) -> Dict:
    infra: Dict = {
        "folder": args.cache_folder,
        "cluster": args.cluster,
        "mode": args.infra_mode,
        "job_name": args.job_name,
        "logs": args.logs,
        "gpus_per_node": args.gpus_per_node,
        "cpus_per_task": args.cpus_per_task,
        "timeout_min": args.timeout_min,
        "slurm_account": args.slurm_account,
        "slurm_additional_parameters": {
            "hint": "nomultithread",
            "signal": "SIGUSR1@90",
            "mail-type": "ALL",
            "mail-user": args.mail_user,
        },
    }
    if args.slurm_partition:
        infra["slurm_partition"] = args.slurm_partition
    if args.slurm_qos:
        infra["slurm_qos"] = args.slurm_qos
    if args.slurm_constraint:
        infra["slurm_constraint"] = args.slurm_constraint
    return infra


if __name__ == "__main__":
    args = parse_args()

    main_infra = build_main_infra(args)
    print(f"[infra] main task infra: {main_infra}")

    task = TrainAndEvalTask(
        config=args.config,
        test_file_orc=args.test_file_orc,
        test_file_wh=args.test_file_wh,
        csv_out=args.csv_out,
        poll_seconds=args.poll_seconds,
        infra=main_infra,
    )
    out = task.process()
    print(out)