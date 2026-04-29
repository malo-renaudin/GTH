#!/usr/bin/env python3
import csv
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List

import pydantic
import exca as xk


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
        step = step_from_checkpoint(self.checkpoint)
        temp_csv = Path(f"results/eval_tmp_step_{step}.csv")
        temp_csv.parent.mkdir(parents=True, exist_ok=True)

        # Call eval_test.py with single-checkpoint mode
        cmd = [
            "python",
            "eval_test.py",
            "--checkpoint",
            str(self.checkpoint),
            "--sentences-file",
            str(self.sentences_file),
            "--structure",
            self.structure,
            "--tokenizer-dir",
            str(self.tokenizer_dir),
            "--max-seq-length",
            str(self.max_seq_length),
            "--result-name",
            str(temp_csv),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
            raise RuntimeError(f"eval_test.py produced no output for checkpoint {self.checkpoint}")

        noun_mass = float(row["noun_mass"])
        verb_mass = float(row["verb_mass"])
        roi_count = int(row["roi_count"])

        return {
            "step": step,
            "test_dataset": self.structure,
            "noun_mass": noun_mass,
            "verb_mass": verb_mass,
            "roi_count": roi_count,
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
    out_dir: Path = Path("out/pretrain/gpt")
    test_file_orc: Path
    test_file_wh: Path
    csv_out: Path = Path("results/live_eval.csv")
    poll_seconds: int = 30
    infra: xk.TaskInfra = xk.TaskInfra()

    @infra.apply
    def process(self) -> Dict:
        self.csv_out.parent.mkdir(parents=True, exist_ok=True)

        # Header CSV
        if not self.csv_out.exists():
            with self.csv_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["step", "test_dataset", "noun_mass", "verb_mass", "roi_count", "checkpoint"],
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
            ckpts = list_step_checkpoints(self.out_dir)
            new_ckpts = [c for c in ckpts if step_from_checkpoint(c) not in evaluated_steps]

            for ckpt in new_ckpts:
                step = step_from_checkpoint(ckpt)

                # Evaluate both ORC and WH structures for this checkpoint
                for structure, test_file in [("orc", self.test_file_orc), ("wh", self.test_file_wh)]:
                    row = EvalCheckpointTask(
                        checkpoint=ckpt,
                        structure=structure,
                        sentences_file=test_file,
                        infra={"folder": self.infra.folder / "eval_ckpt", "cluster": "auto"},
                    ).process()

                    with self.csv_out.open("a", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(
                            f,
                            fieldnames=["step", "test_dataset", "noun_mass", "verb_mass", "roi_count", "checkpoint"],
                        )
                        w.writerow(row)

                    rows_written += 1
                    print(f"[eval] step={step} structure={structure} noun={row['noun_mass']:.6f} verb={row['verb_mass']:.6f} roi_count={row.get('roi_count', 0)}")

                # Mark this step as evaluated after both structures are done
                evaluated_steps.add(step)

            # break condition: training ended and no new ckpt left
            ret = proc.poll()
            if ret is not None:
                ckpts_after = list_step_checkpoints(self.out_dir)
                remaining = [c for c in ckpts_after if step_from_checkpoint(c) not in evaluated_steps]
                if not remaining:
                    break

            time.sleep(self.poll_seconds)

        return {
            "csv": str(self.csv_out),
            "rows_written": rows_written,
            "evaluated_steps": sorted(evaluated_steps),
        }


if __name__ == "__main__":
    task = TrainAndEvalTask(
        config=Path("gpt_baseline.yaml"),
        test_file_orc=Path("data/test_orc7_not_in_train_orc_6_72.txt"),
        test_file_wh=Path("data/valid_wh5_not_in_wh_7_20.txt"),
        infra={
            "folder": Path("results/exca_cache/live_train_eval"),
            "cluster": "auto",  # slurm si dispo
        },
    )
    out = task.process()
    print(out)