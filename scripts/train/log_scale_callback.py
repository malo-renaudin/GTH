import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback, AutoModelForCausalLM

_EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

def _import_eval(name: str):
    path = _EVAL_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

eval_blimp              = _import_eval("eval_blimp")
eval_nested             = _import_eval("eval_nested")
eval_filler_gap         = _import_eval("eval_filler_gap")
eval_probability_masses = _import_eval("eval_probability_masses")


# ---------------------------------------------------------------------------

def log_scale_steps(max_steps: int, n_points: int, start_step: int):
    steps = {int(round(x)) for x in np.geomspace(start_step, max_steps, n_points)}
    steps.add(max_steps)
    return steps


def _safe(fn):
    """Run fn(); return None on any error (missing file, bad format, etc.)."""
    try:
        return fn()
    except Exception:
        return None


def collect_metrics(step: int, ckpt_dir: Path, results_dir: Path) -> dict:
    """Read each eval's output and extract the headline scalar metric."""
    row = {"step": step, "checkpoint": str(ckpt_dir)}

    # BLiMP — overall accuracy from blimp.csv
    def _blimp():
        with open(results_dir / "blimp.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        n = sum(int(r["n_pairs"]) for r in rows)
        ok = sum(int(r["n_correct"]) for r in rows)
        return round(ok / n, 6) if n else None
    row["blimp_accuracy"] = _safe(_blimp)

    # nested — accuracy from JSON (inner / outer)
    for key, fname in [("nested_inner_accuracy", "nested_inner.json"),
                       ("nested_outer_accuracy", "nested_outer.json")]:
        def _json(fname=fname):
            with open(results_dir / fname) as f:
                return round(json.load(f)["accuracy"], 6)
        row[key] = _safe(_json)

    # filler-gap — ML interaction from LME CSV (surprisal_first row)
    for key, stem in [("filler_gap_orc_interaction", "filler_gap_orc_lme"),
                      ("filler_gap_wh_interaction",  "filler_gap_wh_lme")]:
        def _fg(stem=stem):
            with open(results_dir / f"{stem}.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            r = next(r for r in rows if r.get("surprisal_col") == "surprisal_first")
            return round(float(r["ml_interaction"]), 6)
        row[key] = _safe(_fg)

    # probability masses — read JSON files written by the probability-masses eval
    def _pm(fname, key):
        with open(results_dir / fname) as f:
            data = json.load(f)
        return round(float(data.get(key)), 6)

    row["prob_mass_orc_NP"] = _safe(lambda: _pm("probability_masses_orc.json", "NP"))
    row["prob_mass_orc_VP"] = _safe(lambda: _pm("probability_masses_orc.json", "VP"))
    row["prob_mass_orc_qm"] = _safe(lambda: _pm("probability_masses_orc.json", "?"))
    row["prob_mass_wh_NP"]  = _safe(lambda: _pm("probability_masses_wh.json",  "NP"))
    row["prob_mass_wh_VP"]  = _safe(lambda: _pm("probability_masses_wh.json",  "VP"))
    row["prob_mass_wh_qm"]  = _safe(lambda: _pm("probability_masses_wh.json",  "?"))

    return row


SUMMARY_FIELDS = [
    "step", "checkpoint",
    "blimp_accuracy",
    "nested_inner_accuracy", "nested_outer_accuracy",
    "filler_gap_orc_interaction", "filler_gap_wh_interaction",
    "prob_mass_orc_NP", "prob_mass_orc_VP", "prob_mass_orc_qm",
    "prob_mass_wh_NP",  "prob_mass_wh_VP",  "prob_mass_wh_qm",
]


def append_summary(summary_csv: Path, row: dict) -> None:
    write_header = not summary_csv.exists()
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------

def evaluate(ckpt_dir: Path, tokenizer, paradigms, results_dir: Path,
             nested_inner=None, nested_outer=None, locality=None,
             filler_gap_orc=None, filler_gap_wh=None,
             probability_masses_orc=None, probability_masses_wh=None,
             transitivity_orc=None, semantic_distractor=None):
    """Run all evaluations for a checkpoint. Add new eval calls here."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Load model once if any eval needs it
    need_model = any([
        paradigms is not None,
        nested_inner is not None,
        nested_outer is not None,
        filler_gap_orc is not None,
        filler_gap_wh is not None,
        probability_masses_orc is not None,
        probability_masses_wh is not None,
    ])

    model = None
    loaded_model_here = False
    if need_model:
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, local_files_only=True).to(device)
        model.config.use_cache = True
        model.eval()
        loaded_model_here = True

    if paradigms is not None:
        _, paradigm_rows = eval_blimp.evaluate_checkpoint(ckpt_dir, tokenizer, paradigms, device, model=model)
        fields = ["step", "uid", "field", "linguistics_term", "n_pairs", "n_correct", "accuracy"]
        with open(results_dir / "blimp.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(paradigm_rows)
        total_n  = sum(r["n_pairs"]   for r in paradigm_rows)
        total_ok = sum(r["n_correct"] for r in paradigm_rows)
        print(f"  BLiMP accuracy: {total_ok / total_n:.4f}" if total_n else "  BLiMP: no pairs")

    if nested_inner is not None:
        eval_nested.run(str(ckpt_dir), str(nested_inner),  str(results_dir / "nested_inner.json"), model=model)

    if nested_outer is not None:
        eval_nested.run(str(ckpt_dir), str(nested_outer),  str(results_dir / "nested_outer.json"), model=model)

    if filler_gap_orc is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_orc), str(results_dir / "filler_gap_orc.csv"), batch_size=32, model=model)

    if filler_gap_wh is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_wh),  str(results_dir / "filler_gap_wh.csv"), batch_size=32, model=model)

    # Probability masses: evaluate using the same loaded model
    if probability_masses_orc is not None or probability_masses_wh is not None:
        print("  Running probability-masses evaluation...")
        vocab = eval_probability_masses.build_vocabulary()

        if probability_masses_orc is not None:
            pm_orc = eval_probability_masses.process_dataset(str(probability_masses_orc), model, tokenizer, vocab["orc"], eval_probability_masses.get_orc_context, 32)
            with open(results_dir / "probability_masses_orc.json", "w") as f:
                json.dump(pm_orc, f)
            print(f"  Probability masses (ORC): {pm_orc}")

        if probability_masses_wh is not None:
            pm_wh = eval_probability_masses.process_dataset(str(probability_masses_wh), model, tokenizer, vocab["wh"], eval_probability_masses.get_wh_context, 32)
            with open(results_dir / "probability_masses_wh.json", "w") as f:
                json.dump(pm_wh, f)
            print(f"  Probability masses (WH): {pm_wh}")

    if loaded_model_here:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # (transitivity / semantic-distractor evaluations omitted per configuration)

    # Consolidate all headline metrics into the shared summary CSV
    step = int(ckpt_dir.name.split("-")[-1]) if "-" in ckpt_dir.name else 0
    summary_row = collect_metrics(step, ckpt_dir, results_dir)
    append_summary(results_dir.parent / "summary.csv", summary_row)
    print(f"  Summary row written → {results_dir.parent / 'summary.csv'}")


# ---------------------------------------------------------------------------

class LogScaleCallback(TrainerCallback):

    def __init__(self, max_steps, n_points, start_step, output_dir, tokenizer,
                 blimp_dir=None,
                 nested_inner=None, nested_outer=None, locality=None,
                 filler_gap_orc=None, filler_gap_wh=None,
                 probability_masses_orc=None, probability_masses_wh=None,
                 transitivity_orc=None, semantic_distractor=None):
        self.eval_steps        = log_scale_steps(max_steps, n_points, start_step)
        # Add a dense evaluation window between 1000 and 2000 (every 50 steps)
        dense_window = set(range(1000, 2001, 50))
        self.eval_steps.update(dense_window)
        self.output_dir        = Path(output_dir)
        self.tokenizer         = tokenizer
        self.results_dir       = (
            self.output_dir / "eval_results"
        )
        self.paradigms         = (
            eval_blimp.load_paradigms(Path(blimp_dir), tokenizer) if blimp_dir else None
        )
        self.nested_inner      = Path(nested_inner)      if nested_inner      else None
        self.nested_outer      = Path(nested_outer)      if nested_outer      else None
        self.locality          = Path(locality)          if locality          else None
        self.filler_gap_orc    = Path(filler_gap_orc)    if filler_gap_orc    else None
        self.filler_gap_wh     = Path(filler_gap_wh)     if filler_gap_wh     else None
        self.probability_masses_orc = Path(probability_masses_orc) if probability_masses_orc else None
        self.probability_masses_wh  = Path(probability_masses_wh)  if probability_masses_wh  else None
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
            probability_masses_orc = self.probability_masses_orc,
            probability_masses_wh  = self.probability_masses_wh,
            transitivity_orc  = self.transitivity_orc,
            semantic_distractor = self.semantic_distractor,
        )
        return control