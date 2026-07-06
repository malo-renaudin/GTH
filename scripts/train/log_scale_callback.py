import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

def _import_eval(name: str):
    path = _EVAL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

eval_blimp  = _import_eval("eval_blimp")
eval_nested = _import_eval("eval_nested")


# ---------------------------------------------------------------------------

def log_scale_steps(max_steps: int, n_points: int, start_step: int):
    steps = {int(round(x)) for x in np.geomspace(start_step, max_steps, n_points)}
    steps.add(max_steps)
    return steps


def evaluate(ckpt_dir: Path, tokenizer, paradigms, nested_dataset, results_dir: Path):
    """Run all evaluations for a checkpoint. Add new eval calls here."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)

    if paradigms is not None:
        _, paradigm_rows = eval_blimp.evaluate_checkpoint(
            ckpt_dir, tokenizer, paradigms, device
        )
        fields = ["step", "uid", "field", "linguistics_term", "n_pairs", "n_correct", "accuracy"]
        with open(results_dir / "blimp.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(paradigm_rows)
        total_n  = sum(r["n_pairs"]   for r in paradigm_rows)
        total_ok = sum(r["n_correct"] for r in paradigm_rows)
        print(f"  BLiMP accuracy: {total_ok / total_n:.4f}" if total_n else "  BLiMP: no pairs")

    if nested_dataset is not None:
        eval_nested.run(str(ckpt_dir), str(nested_dataset), str(results_dir / "nested.json"))


# ---------------------------------------------------------------------------

class LogScaleCallback(TrainerCallback):

    def __init__(self, max_steps, n_points, start_step, output_dir,
                 tokenizer, blimp_dir=None, nested_eval_dataset=None, eval_results_dir=None):
        self.eval_steps   = log_scale_steps(max_steps, n_points, start_step)
        self.output_dir   = Path(output_dir)
        self.tokenizer    = tokenizer
        self.nested_path  = Path(nested_eval_dataset) if nested_eval_dataset else None
        self.results_dir  = (
            Path(eval_results_dir) if eval_results_dir
            else self.output_dir / "eval_results"
        )
        self.paradigms = (
            eval_blimp.load_paradigms(Path(blimp_dir), tokenizer)
            if blimp_dir else None
        )
        print(f"[LogScaleCallback] {len(self.eval_steps)} steps: {sorted(self.eval_steps)}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.eval_steps:
            control.should_evaluate = True   # HF runs val loss / perplexity
            control.should_save     = True   # writes checkpoint → on_save fires
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        print(f"\n[LogScaleCallback] running evals at step {step}")
        evaluate(
            ckpt_dir       = Path(args.output_dir) / f"checkpoint-{step}",
            tokenizer      = self.tokenizer,
            paradigms      = self.paradigms,
            nested_dataset = self.nested_path,
            results_dir    = self.results_dir / f"step-{step}",
        )
        return control