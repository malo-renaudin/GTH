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

eval_blimp                    = _import_eval("eval_blimp")
eval_nested                   = _import_eval("eval_nested")
eval_locality                 = _import_eval("eval_locality")
eval_filler_gap               = _import_eval("eval_filler_gap")
eval_transitivity_orc         = _import_eval("eval_transitivity_orc")
eval_semantic_distractors_orc = _import_eval("eval_semantic_distractors_orc")


# ---------------------------------------------------------------------------

def log_scale_steps(max_steps: int, n_points: int, start_step: int):
    steps = {int(round(x)) for x in np.geomspace(start_step, max_steps, n_points)}
    steps.add(max_steps)
    return steps


def evaluate(ckpt_dir: Path, tokenizer, paradigms, results_dir: Path,
             nested_inner=None, nested_outer=None, locality=None,
             filler_gap_orc=None, filler_gap_wh=None,
             transitivity_orc=None, semantic_distractor=None):
    """Run all evaluations for a checkpoint. Add new eval calls here."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)

    if paradigms is not None:
        _, paradigm_rows = eval_blimp.evaluate_checkpoint(ckpt_dir, tokenizer, paradigms, device)
        fields = ["step", "uid", "field", "linguistics_term", "n_pairs", "n_correct", "accuracy"]
        with open(results_dir / "blimp.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(paradigm_rows)
        total_n  = sum(r["n_pairs"]   for r in paradigm_rows)
        total_ok = sum(r["n_correct"] for r in paradigm_rows)
        print(f"  BLiMP accuracy: {total_ok / total_n:.4f}" if total_n else "  BLiMP: no pairs")

    if nested_inner is not None:
        eval_nested.run(str(ckpt_dir), str(nested_inner),  str(results_dir / "nested_inner.json"))

    if nested_outer is not None:
        eval_nested.run(str(ckpt_dir), str(nested_outer),  str(results_dir / "nested_outer.json"))

    if locality is not None:
        eval_locality.run(str(ckpt_dir), str(locality),    str(results_dir / "locality.json"))

    if filler_gap_orc is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_orc), str(results_dir / "filler_gap_orc.csv"))

    if filler_gap_wh is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_wh),  str(results_dir / "filler_gap_wh.csv"))

    if transitivity_orc is not None:
        eval_transitivity_orc.run(str(ckpt_dir), str(transitivity_orc),
                                  str(results_dir / "transitivity_orc.csv"))

    if semantic_distractor is not None:
        eval_semantic_distractors_orc.run(str(ckpt_dir), str(semantic_distractor),
                                          str(results_dir / "semantic_distractor.csv"))


# ---------------------------------------------------------------------------

class LogScaleCallback(TrainerCallback):

    def __init__(self, max_steps, n_points, start_step, output_dir, tokenizer,
                 blimp_dir=None,
                 nested_inner=None, nested_outer=None, locality=None,
                 filler_gap_orc=None, filler_gap_wh=None,
                 transitivity_orc=None, semantic_distractor=None,
                 eval_results_dir=None):
        self.eval_steps        = log_scale_steps(max_steps, n_points, start_step)
        self.output_dir        = Path(output_dir)
        self.tokenizer         = tokenizer
        self.results_dir       = (
            Path(eval_results_dir) if eval_results_dir
            else self.output_dir / "eval_results"
        )
        self.paradigms         = (
            eval_blimp.load_paradigms(Path(blimp_dir), tokenizer) if blimp_dir else None
        )
        self.nested_inner      = Path(nested_inner)      if nested_inner      else None
        self.nested_outer      = Path(nested_outer)      if nested_outer      else None
        self.locality          = Path(locality)          if locality          else None
        self.filler_gap_orc    = Path(filler_gap_orc)    if filler_gap_orc    else None
        self.filler_gap_wh     = Path(filler_gap_wh)     if filler_gap_wh     else None
        self.transitivity_orc  = Path(transitivity_orc)  if transitivity_orc  else None
        self.semantic_distractor = Path(semantic_distractor) if semantic_distractor else None
        print(f"[LogScaleCallback] {len(self.eval_steps)} steps: {sorted(self.eval_steps)}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.eval_steps:
            control.should_evaluate = True
            control.should_save     = True
        return control

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        print(f"\n[LogScaleCallback] running evals at step {step}")
        evaluate(
            ckpt_dir          = Path(args.output_dir) / f"checkpoint-{step}",
            tokenizer         = self.tokenizer,
            paradigms         = self.paradigms,
            results_dir       = self.results_dir / f"step-{step}",
            nested_inner      = self.nested_inner,
            nested_outer      = self.nested_outer,
            locality          = self.locality,
            filler_gap_orc    = self.filler_gap_orc,
            filler_gap_wh     = self.filler_gap_wh,
            transitivity_orc  = self.transitivity_orc,
            semantic_distractor = self.semantic_distractor,
        )
        return control