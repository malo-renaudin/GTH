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

_BLIMP_TERMS = [
    "anaphor_agreement", "argument_structure", "binding", "control_raising",
    "determiner_noun_agreement", "ellipsis", "filler_gap_dependency",
    "irregular_forms", "island_effects", "npi_licensing", "quantifiers",
    "s-selection", "subject_verb_agreement",
]
_NESTED_CATS = ["plural_plural", "plural_singular", "singular_plural", "singular_singular"]


# ---------------------------------------------------------------------------

def log_scale_steps(max_steps: int, n_points: int, start_step: int):
    del n_points, start_step
    steps = set(range(200, max_steps + 1, 200))
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

    # BLiMP — overall accuracy + per-linguistics_term
    def _blimp():
        from collections import defaultdict
        with open(results_dir / "blimp.csv", newline="") as f:
            brows = list(csv.DictReader(f))
        n  = sum(int(r["n_pairs"])   for r in brows)
        ok = sum(int(r["n_correct"]) for r in brows)
        term_n, term_ok = defaultdict(int), defaultdict(int)
        for r in brows:
            t = r["linguistics_term"]
            term_n[t]  += int(r["n_pairs"])
            term_ok[t] += int(r["n_correct"])
        per_term = {t: round(term_ok[t] / term_n[t], 6) for t in term_n if term_n[t] > 0}
        return (round(ok / n, 6) if n else None), per_term
    blimp_result = _safe(_blimp)
    row["blimp_accuracy"] = blimp_result[0] if blimp_result else None
    blimp_per_term = blimp_result[1] if blimp_result else {}
    for term in _BLIMP_TERMS:
        row[f"blimp_{term}"] = blimp_per_term.get(term)

    # nested — overall accuracy + per-category (inner / outer)
    for side in ("inner", "outer"):
        def _nested(side=side):
            with open(results_dir / f"nested_{side}.json") as f:
                return json.load(f)
        data = _safe(_nested)
        row[f"nested_{side}_accuracy"] = round(data["accuracy"], 6) if data else None
        per_cat = (data or {}).get("per_category", {})
        for cat in _NESTED_CATS:
            row[f"nested_{side}_{cat}"] = round(per_cat[cat], 6) if cat in per_cat else None

    # filler-gap — ML interaction from LME CSV (surprisal_first), IV and OOV
    for tag in ("iv", "oov"):
        for task in ("orc", "wh"):
            def _fg(task=task, tag=tag):
                with open(results_dir / f"filler_gap_{task}_{tag}_lme.csv", newline="") as f:
                    frows = list(csv.DictReader(f))
                r = next(r for r in frows if r.get("surprisal_col") == "surprisal_first")
                return round(float(r["ml_interaction"]), 6)
            row[f"filler_gap_{task}_{tag}_interaction"] = _safe(_fg)

    # probability masses — IV and OOV
    for tag in ("iv", "oov"):
        for task in ("orc", "wh"):
            def _pm(task=task, tag=tag):
                with open(results_dir / f"probability_masses_{task}_{tag}.json") as f:
                    return json.load(f)
            data = _safe(_pm)
            for k, k_name in (("NP", "NP"), ("VP", "VP"), ("?", "qm")):
                row[f"prob_mass_{task}_{tag}_{k_name}"] = (
                    round(float(data[k]), 6) if data and k in data else None
                )

    return row


SUMMARY_FIELDS = [
    "step", "checkpoint",
    "blimp_accuracy",
    *[f"blimp_{t}" for t in _BLIMP_TERMS],
    "nested_inner_accuracy", "nested_outer_accuracy",
    *[f"nested_inner_{c}" for c in _NESTED_CATS],
    *[f"nested_outer_{c}" for c in _NESTED_CATS],
    "filler_gap_orc_iv_interaction",  "filler_gap_orc_oov_interaction",
    "filler_gap_wh_iv_interaction",   "filler_gap_wh_oov_interaction",
    "prob_mass_orc_iv_NP",  "prob_mass_orc_iv_VP",  "prob_mass_orc_iv_qm",
    "prob_mass_orc_oov_NP", "prob_mass_orc_oov_VP", "prob_mass_orc_oov_qm",
    "prob_mass_wh_iv_NP",   "prob_mass_wh_iv_VP",   "prob_mass_wh_iv_qm",
    "prob_mass_wh_oov_NP",  "prob_mass_wh_oov_VP",  "prob_mass_wh_oov_qm",
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
             filler_gap_orc_iv=None, filler_gap_orc_oov=None,
             filler_gap_wh_iv=None,  filler_gap_wh_oov=None,
             probability_masses_orc_iv=None,  probability_masses_orc_oov=None,
             probability_masses_wh_iv=None,   probability_masses_wh_oov=None,
             transitivity_orc=None, semantic_distractor=None):
    """Run all evaluations for a checkpoint. Add new eval calls here."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Load model once if any eval needs it
    need_model = any([
        paradigms is not None,
        nested_inner is not None,
        nested_outer is not None,
        filler_gap_orc_iv is not None,
        filler_gap_orc_oov is not None,
        filler_gap_wh_iv is not None,
        filler_gap_wh_oov is not None,
        probability_masses_orc_iv is not None,
        probability_masses_orc_oov is not None,
        probability_masses_wh_iv is not None,
        probability_masses_wh_oov is not None,
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

    if filler_gap_orc_iv is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_orc_iv),  str(results_dir / "filler_gap_orc_iv.csv"),  batch_size=32, model=model, tokenizer=tokenizer)

    if filler_gap_orc_oov is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_orc_oov), str(results_dir / "filler_gap_orc_oov.csv"), batch_size=32, model=model, tokenizer=tokenizer)

    if filler_gap_wh_iv is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_wh_iv),   str(results_dir / "filler_gap_wh_iv.csv"),   batch_size=32, model=model, tokenizer=tokenizer)

    if filler_gap_wh_oov is not None:
        eval_filler_gap.run(str(ckpt_dir), str(filler_gap_wh_oov),  str(results_dir / "filler_gap_wh_oov.csv"),  batch_size=32, model=model, tokenizer=tokenizer)

    # Probability masses: IV and OOV, using the same loaded model
    pm_tasks = [
        ("iv",  probability_masses_orc_iv,  probability_masses_wh_iv),
        ("oov", probability_masses_orc_oov, probability_masses_wh_oov),
    ]
    for vocab_tag, orc_file, wh_file in pm_tasks:
        if orc_file is None and wh_file is None:
            continue
        print(f"  Running probability-masses evaluation ({vocab_tag})...")
        vocab_cands = eval_probability_masses.build_vocabulary(vocab_tag)
        orc_ctx_fn = (eval_probability_masses.get_orc_context if vocab_tag == "iv"
                      else lambda s, _v=vocab_tag: eval_probability_masses.find_orc_context(s, _v))

        if orc_file is not None:
            pm = eval_probability_masses.process_dataset(
                file_path=str(orc_file), model=model, tokenizer=tokenizer,
                vocab=vocab_cands["orc"], context_fn=orc_ctx_fn, batch_size=1024,
            )
            with open(results_dir / f"probability_masses_orc_{vocab_tag}.json", "w") as f:
                json.dump(pm, f)
            print(f"  PM ORC {vocab_tag}: {pm}")

        if wh_file is not None:
            pm = eval_probability_masses.process_dataset(
                file_path=str(wh_file), model=model, tokenizer=tokenizer,
                vocab=vocab_cands["wh"], context_fn=eval_probability_masses.get_wh_context,
                batch_size=1024,
            )
            with open(results_dir / f"probability_masses_wh_{vocab_tag}.json", "w") as f:
                json.dump(pm, f)
            print(f"  PM WH {vocab_tag}: {pm}")

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
                 filler_gap_orc_iv=None,  filler_gap_orc_oov=None,
                 filler_gap_wh_iv=None,   filler_gap_wh_oov=None,
                 probability_masses_orc_iv=None,  probability_masses_orc_oov=None,
                 probability_masses_wh_iv=None,   probability_masses_wh_oov=None,
                 transitivity_orc=None, semantic_distractor=None):
        self.eval_steps        = log_scale_steps(max_steps, n_points, start_step)
        # Add a dense evaluation window between 1000 and 5000 (every 100 steps)
        # dense_window = set(range(1000, 5001, 100))
        # self.eval_steps.update(dense_window)
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
        self.filler_gap_orc_iv   = Path(filler_gap_orc_iv)   if filler_gap_orc_iv   else None
        self.filler_gap_orc_oov  = Path(filler_gap_orc_oov)  if filler_gap_orc_oov  else None
        self.filler_gap_wh_iv    = Path(filler_gap_wh_iv)    if filler_gap_wh_iv    else None
        self.filler_gap_wh_oov   = Path(filler_gap_wh_oov)   if filler_gap_wh_oov   else None
        self.probability_masses_orc_iv  = Path(probability_masses_orc_iv)  if probability_masses_orc_iv  else None
        self.probability_masses_orc_oov = Path(probability_masses_orc_oov) if probability_masses_orc_oov else None
        self.probability_masses_wh_iv   = Path(probability_masses_wh_iv)   if probability_masses_wh_iv   else None
        self.probability_masses_wh_oov  = Path(probability_masses_wh_oov)  if probability_masses_wh_oov  else None
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
            ckpt_dir                   = Path(args.output_dir) / f"checkpoint-{step}",
            tokenizer                  = self.tokenizer,
            paradigms                  = self.paradigms,
            results_dir                = self.results_dir / f"step-{step}",
            nested_inner               = self.nested_inner,
            nested_outer               = self.nested_outer,
            locality                   = self.locality,
            filler_gap_orc_iv          = self.filler_gap_orc_iv,
            filler_gap_orc_oov         = self.filler_gap_orc_oov,
            filler_gap_wh_iv           = self.filler_gap_wh_iv,
            filler_gap_wh_oov          = self.filler_gap_wh_oov,
            probability_masses_orc_iv  = self.probability_masses_orc_iv,
            probability_masses_orc_oov = self.probability_masses_orc_oov,
            probability_masses_wh_iv   = self.probability_masses_wh_iv,
            probability_masses_wh_oov  = self.probability_masses_wh_oov,
            transitivity_orc           = self.transitivity_orc,
            semantic_distractor        = self.semantic_distractor,
        )
        return control