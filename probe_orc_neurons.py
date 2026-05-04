"""
probe_orc_neurons.py — Find neurons specialized for Object Relative Clauses.

For each SRC/ORC minimal pair, the model is run up to and including the
relativizer ("that"/"who"). The hidden state at that position is collected
from every transformer block, then a Welch t-test is run per neuron:

  positive t-value  →  neuron is more active for ORC than SRC

The top-1% neurons (by t-value) are then ablated (zeroed out at the
relativizer position) and the model is re-run; P(next word) is reported
for both conditions before and after ablation.

Usage:
    python probe_orc_neurons.py \\
        --src-file src_pairs.txt \\
        --orc-file orc_pairs.txt \\
        --checkpoint <path/to/checkpoint_dir_or_file> \\
        [--tokenizer-dir checkpoints/gpt2] \\
        [--output orc_neurons.json] \\
        [--top-pct 1.0]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from litgpt import Tokenizer
from litgpt.config import Config
from litgpt.model import GPT

from eval_test import (
    _word_token_spans, lexical_mass, extract_orc_moved_np,
    nouns_orc, verbs_orc, verbs_orc_continuation,
)
REL_MARKERS = {"that", "who"}


# ── Checkpoint helpers (same pattern as eval_test.py) ────────────────────────

def load_checkpoint(ckpt_file: Path, device: torch.device) -> GPT:
    model = GPT(Config.from_checkpoint(ckpt_file.parent))
    raw = torch.load(ckpt_file, map_location=device, weights_only=False)
    state = raw.get("model") if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else raw
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_file}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def resolve_checkpoint_file(p: Path) -> Path:
    if p.is_file():
        return p
    ckpt = p / "lit_model.pth"
    if ckpt.exists():
        return ckpt
    raise FileNotFoundError(f"No lit_model.pth found in {p}")


# ── Model structure ───────────────────────────────────────────────────────────

def get_transformer_blocks(model: GPT) -> List[torch.nn.Module]:
    """Return the ordered list of transformer Block modules (litgpt GPT-2 style)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise AttributeError(
        "Cannot locate transformer blocks. Expected model.transformer.h."
    )


# ── Tokenisation helpers ──────────────────────────────────────────────────────

def normalize_word(w: str) -> str:
    return w.strip(".,?!;:\"'()[]{}").lower()


def tokenize_sentence(
    sentence: str, tokenizer: Tokenizer
) -> Tuple[List[int], List[Tuple[str, int, int]]]:
    """
    Tokenise word-by-word (no BOS/EOS) and return:
      tokens : flat token-id list
      spans  : [(word_str, tok_start, tok_end_exclusive), …]
    """
    tokens: List[int] = []
    spans:  List[Tuple[str, int, int]] = []
    pos = 0
    for i, word in enumerate(sentence.split()):
        prefix = "" if i == 0 else " "
        ids = tokenizer.encode(prefix + word, bos=False, eos=False).tolist()
        spans.append((word, pos, pos + len(ids)))
        tokens.extend(ids)
        pos += len(ids)
    return tokens, spans


def find_rel_marker_span_idx(
    spans: List[Tuple[str, int, int]]
) -> Optional[int]:
    """Return the spans-list index of the first 'that'/'who', or None."""
    for i, (word, start, end) in enumerate(spans):
        if normalize_word(word) in REL_MARKERS:
            return i
    return None


# ── Hidden-state collection ───────────────────────────────────────────────────

def collect_paired_hidden_states(
    model: GPT,
    tokenizer: Tokenizer,
    src_sentences: List[str],
    orc_sentences: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect hidden states at the embedded verb position for aligned SRC/ORC pairs.
    The embedded verb is the same lexical item in both conditions (minimal pairs),
    so hidden-state differences reflect structural context (filled vs. unfilled gap),
    not token-identity differences.
    Only pairs where both sentences have a valid relativizer and an embedded verb
    are retained, preserving alignment for the paired t-test.

    Returns:
      src_acts : float32 array, shape (n_valid_pairs, n_layers * n_embd)
      orc_acts : float32 array, shape (n_valid_pairs, n_layers * n_embd)
    """
    blocks = get_transformer_blocks(model)
    n_layers = len(blocks)
    roi_verb_set = set(verbs_orc)

    layer_captures: List[Optional[torch.Tensor]] = [None] * n_layers
    target_pos = [0]

    def make_hook(layer_idx: int):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_captures[layer_idx] = h[0, target_pos[0], :].detach().cpu()
        return hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]

    def _run_one(tokens: List[int], spans: List[Tuple[str, int, int]]) -> Optional[np.ndarray]:
        # Require a relativizer
        if find_rel_marker_span_idx(spans) is None:
            return None
        # Probe position: last token of the embedded verb
        roi_idx = None
        for word, start, end in spans:
            if normalize_word(word) in roi_verb_set:
                roi_idx = end - 1
                break
        if roi_idx is None:
            return None
        target_pos[0] = roi_idx
        x = torch.tensor(tokens[: target_pos[0] + 1], device=device).unsqueeze(0)
        with torch.no_grad():
            _ = model(x)
        fallback_dim = next(
            (layer_captures[i].shape[0] for i in range(n_layers) if layer_captures[i] is not None), 1
        )
        return np.concatenate([
            layer_captures[i].numpy() if layer_captures[i] is not None
            else np.zeros(fallback_dim, dtype=np.float32)
            for i in range(n_layers)
        ])

    src_acts_list: List[np.ndarray] = []
    orc_acts_list: List[np.ndarray] = []
    n_skipped = 0

    try:
        for src_sent, orc_sent in tqdm(
            zip(src_sentences, orc_sentences), total=len(src_sentences),
            desc="  collecting paired activations", leave=False,
        ):
            src_tokens, src_spans = tokenize_sentence(src_sent, tokenizer)
            orc_tokens, orc_spans = tokenize_sentence(orc_sent, tokenizer)
            src_vec = _run_one(src_tokens, src_spans)
            orc_vec = _run_one(orc_tokens, orc_spans)
            if src_vec is None or orc_vec is None:
                n_skipped += 1
                continue
            src_acts_list.append(src_vec)
            orc_acts_list.append(orc_vec)
    finally:
        for h in handles:
            h.remove()

    if not src_acts_list:
        raise ValueError("No valid pairs found (no 'that'/'who' in any sentence).")
    if n_skipped:
        print(f"  Warning: {n_skipped} pairs skipped (missing relativizer).")
    return (
        np.stack(src_acts_list).astype(np.float32),
        np.stack(orc_acts_list).astype(np.float32),
    )


def eval_orc_task(
    model, tokenizer, sentences, device,
    ablate_dims=None,
):
    blocks = get_transformer_blocks(model)
    handles = []

    if ablate_dims:
        def make_ablation_hook(layer_idx, dims):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[0, :, dims] = 0.0  # zero at every token position
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook
        for layer_idx, dims in ablate_dims.items():
            if dims:
                handles.append(blocks[layer_idx].register_forward_hook(
                    make_ablation_hook(layer_idx, dims)))

    roi_verb_set = set(verbs_orc)
    orc_verb_cont_lists = [tokenizer.encode(" " + w, bos=False, eos=False).tolist()
                           for w in verbs_orc_continuation]
    noun_forms = set(nouns_orc)

    results = []
    try:
        for sent in tqdm(sentences, desc="  orc eval", leave=False):
            tok, spans = _word_token_spans(sent, tokenizer)

            if find_rel_marker_span_idx(spans) is None:
                results.append(None)
                continue

            # Find ROI: first embedded verb
            roi_idx = None
            for word, start, end in spans:
                if normalize_word(word) in roi_verb_set:
                    roi_idx = end - 1
                    break
            if roi_idx is None:
                results.append(None)
                continue

            x = torch.tensor(tok[: roi_idx + 1], device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            np_words, head = extract_orc_moved_np(sent, noun_forms)
            if not np_words or head is None:
                results.append(None)
                continue

            np_lists   = [tokenizer.encode(" " + w, bos=False, eos=False).tolist() for w in np_words]
            np_mass    = lexical_mass(probs, np_lists) / max(1, len(np_lists))
            verb_mass  = lexical_mass(probs, orc_verb_cont_lists) / max(1, len(orc_verb_cont_lists))
            results.append({"np_mass": np_mass, "verb_mass": verb_mass,
                             "np_minus_verb": np_mass - verb_mass})
    finally:
        for h in handles:
            h.remove()
    return results



def eval_src_task(
    model: GPT,
    tokenizer: Tokenizer,
    sentences: List[str],
    device: torch.device,
    ablate_dims: Optional[Dict[int, List[int]]] = None,
) -> List[Optional[dict]]:
    """
    SRC selectivity check at the embedded verb.
    In SRC the subject slot is filled by the filler; the verb still needs its
    direct object, so the model should predict 'the' (start of the object NP)
    rather than a main-clause verb continuation.
    Score: the_mass - verb_mass  (positive = model correctly expects an object)
    Neurons are zeroed at every token position throughout the full forward pass.
    """
    blocks = get_transformer_blocks(model)
    handles: List = []

    if ablate_dims:
        def make_ablation_hook(layer_idx: int, dims: List[int]):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[0, :, dims] = 0.0  # zero at every token position
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook
        for layer_idx, dims in ablate_dims.items():
            if dims:
                handles.append(blocks[layer_idx].register_forward_hook(
                    make_ablation_hook(layer_idx, dims)))

    roi_verb_set = set(verbs_orc)
    orc_verb_cont_lists = [tokenizer.encode(" " + w, bos=False, eos=False).tolist()
                           for w in verbs_orc_continuation]
    the_lists = [tokenizer.encode(" the", bos=False, eos=False).tolist()]

    results: List[Optional[dict]] = []
    try:
        for sent in tqdm(sentences, desc="  src eval", leave=False):
            tok, spans = _word_token_spans(sent, tokenizer)

            if find_rel_marker_span_idx(spans) is None:
                results.append(None)
                continue

            roi_idx = None
            for word, start, end in spans:
                if normalize_word(word) in roi_verb_set:
                    roi_idx = end - 1
                    break
            if roi_idx is None:
                results.append(None)
                continue

            x = torch.tensor(tok[: roi_idx + 1], device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            the_mass  = lexical_mass(probs, the_lists)
            verb_mass = lexical_mass(probs, orc_verb_cont_lists) / max(1, len(orc_verb_cont_lists))
            results.append({"the_mass": the_mass, "verb_mass": verb_mass,
                             "the_minus_verb": the_mass - verb_mass})
    finally:
        for h in handles:
            h.remove()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe for ORC-specialized neurons via Welch t-test and ablation."
    )
    ap.add_argument("--src-file", type=Path, required=True,
                    help="SRC sentences, one per line (aligned with --orc-file)")
    ap.add_argument("--orc-file", type=Path, required=True,
                    help="ORC sentences, one per line (aligned with --src-file)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint",     type=Path, help="Path to checkpoint file or directory")
    g.add_argument("--checkpoint-dir", type=Path, help="Dir with step-* subdirs; uses the latest")
    ap.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    ap.add_argument("--output",  type=Path, default=Path("orc_neurons.json"),
                    help="Output JSON with positive-t neuron list")
    ap.add_argument("--top-pct", type=float, default=1.0,
                    help="Top %% of positive-t neurons to ablate (default: 1.0)")
    ap.add_argument("--min-cohens-d", type=float, default=0.5,
                    help="Minimum |Cohen's d| for a neuron to be considered selective (default: 0.5)")
    ap.add_argument("--device",  type=str, default=None,
                    help="Force device, e.g. 'cpu' or 'cuda:1'")
    args = ap.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Resolve checkpoint
    if args.checkpoint is not None:
        ckpt_file = resolve_checkpoint_file(args.checkpoint)
    else:
        ckpts = sorted(
            args.checkpoint_dir.glob("step-*/lit_model.pth"),
            key=lambda p: int(p.parent.name.split("-")[1]),
        )
        if not ckpts:
            raise FileNotFoundError(f"No step-* checkpoints found under {args.checkpoint_dir}")
        ckpt_file = ckpts[-1]
        print(f"Using latest checkpoint: {ckpt_file}")

    # Load sentences
    src_sents = [l.strip() for l in args.src_file.read_text().splitlines() if l.strip()]
    orc_sents = [l.strip() for l in args.orc_file.read_text().splitlines() if l.strip()]
    if len(src_sents) != len(orc_sents):
        raise ValueError("--src-file and --orc-file must have the same number of lines.")
    print(f"Loaded {len(src_sents)} sentence pairs.")

    # Load model
    print(f"Loading model from {ckpt_file} …")
    model = load_checkpoint(ckpt_file, device)
    tokenizer = Tokenizer(args.tokenizer_dir)
    n_layers = len(get_transformer_blocks(model))
    print(f"Model: {n_layers} transformer layers")

    # ── 1. Collect hidden states ─────────────────────────────────────────────
    print("\n[1/3] Collecting hidden states at embedded verb position (paired) …")
    src_acts, orc_acts = collect_paired_hidden_states(
        model, tokenizer, src_sents, orc_sents, device
    )

    n_neurons = src_acts.shape[1]
    n_embd    = n_neurons // n_layers
    print(f"  hidden dim per layer: {n_embd}, total neurons: {n_neurons}")

    # ── 2. Paired t-test per neuron + Bonferroni + Cohen's d ────────────────
    print("\n[2/3] Running paired t-tests (ORC vs SRC) with Bonferroni + Cohen's d filter …")
    diffs      = (orc_acts - src_acts).astype(np.float64)   # shape (n_pairs, n_neurons)
    mean_delta = diffs.mean(axis=0)
    std_delta  = diffs.std(axis=0, ddof=1)
    cohens_d   = mean_delta / (std_delta + 1e-12)

    t_stats = np.empty(n_neurons, dtype=np.float64)
    p_vals  = np.empty(n_neurons, dtype=np.float64)
    for i in range(n_neurons):
        t, p = stats.ttest_rel(orc_acts[:, i], src_acts[:, i])
        t_stats[i] = t
        p_vals[i]  = p

    p_vals_bonf = np.clip(p_vals * n_neurons, 0.0, 1.0)

    selective = (t_stats > 0) & (p_vals_bonf < 0.05) & (cohens_d >= args.min_cohens_d)
    pos_idx        = np.where(selective)[0]
    pos_idx_sorted = pos_idx[np.argsort(t_stats[pos_idx])[::-1]]   # descending by t

    n_pos  = int(np.sum(t_stats > 0))
    n_bonf = int(np.sum((t_stats > 0) & (p_vals_bonf < 0.05)))
    print(f"  Neurons with t > 0 (ORC > SRC): {n_pos} / {n_neurons}")
    print(f"  Significant after Bonferroni (p_bonf < 0.05): {n_bonf}")
    print(f"  Also |Cohen's d| >= {args.min_cohens_d}: {len(pos_idx_sorted)}")

    neuron_records = [
        {
            "flat_idx":     int(idx),
            "layer":        int(idx // n_embd),
            "dim":          int(idx % n_embd),
            "t_stat":       float(t_stats[idx]),
            "mean_delta":   float(mean_delta[idx]),
            "cohens_d":     float(cohens_d[idx]),
            "p_value":      float(p_vals[idx]),
            "p_value_bonf": float(p_vals_bonf[idx]),
        }
        for idx in pos_idx_sorted
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(neuron_records, indent=2))
    print(f"  Saved neuron list → {args.output}")

    # ── 3. Ablation of top-N% neurons ────────────────────────────────────────
    n_selective = len(pos_idx_sorted)
    n_top   = max(1, int(np.ceil(args.top_pct / 100.0 * n_selective)))
    top_idx = pos_idx_sorted[:n_top]
    print(f"\n[3/3] Ablating top {args.top_pct}% of {n_selective} selective neurons = {n_top} neurons …")

    if neuron_records:
        top5 = [(r["layer"], r["dim"], round(r["t_stat"], 3)) for r in neuron_records[:5]]
        print(f"  Top-5 neurons (layer, dim, t): {top5}")

    # Group ablated dims by layer
    ablate_dims: Dict[int, List[int]] = {}
    for idx in top_idx:
        layer = int(idx // n_embd)
        dim   = int(idx % n_embd)
        ablate_dims.setdefault(layer, []).append(dim)

    def mean_nonnull_key(lst: List[Optional[dict]], key: str) -> float:
        v = [x[key] for x in lst if x is not None]
        return float(np.mean(v)) if v else float("nan")

    print("  Baseline (no ablation):")
    orc_base = eval_orc_task(model, tokenizer, orc_sents, device)
    src_base = eval_src_task(model, tokenizer, src_sents, device)

    print("  Ablated:")
    orc_ablated = eval_orc_task(model, tokenizer, orc_sents, device, ablate_dims)
    src_ablated = eval_src_task(model, tokenizer, src_sents, device, ablate_dims)

    orc_b = mean_nonnull_key(orc_base,    "np_minus_verb")
    orc_a = mean_nonnull_key(orc_ablated, "np_minus_verb")
    src_b = mean_nonnull_key(src_base,    "the_minus_verb")
    src_a = mean_nonnull_key(src_ablated, "the_minus_verb")

    print("\n=== Ablation results ===")
    print(f"{'Condition':42s} {'Baseline':>10} {'Ablated':>10} {'Delta':>10}")
    print(f"{'ORC  np_minus_verb   (neg = good)':42s} {orc_b:10.6f} {orc_a:10.6f} {orc_a - orc_b:+10.6f}")
    print(f"{'SRC  the_minus_verb  (pos = good)':42s} {src_b:10.6f} {src_a:10.6f} {src_a - src_b:+10.6f}")
    print("Selective ORC neurons: ORC delta >> 0, SRC delta ≈ 0")

    ablation_output = args.output.parent / (args.output.stem + "_ablation.json")
    ablation_results = {
        "n_ablated_neurons": n_top,
        "top_pct":           args.top_pct,
        "orc": {
            "baseline_mean_np_minus_verb": orc_b,
            "ablated_mean_np_minus_verb":  orc_a,
            "delta":                       orc_a - orc_b,
            "per_sentence": [
                {"baseline": b, "ablated": a}
                for b, a in zip(orc_base, orc_ablated)
            ],
        },
        "src": {
            "baseline_mean_the_minus_verb": src_b,
            "ablated_mean_the_minus_verb":  src_a,
            "delta":                        src_a - src_b,
            "per_sentence": [
                {"baseline": b, "ablated": a}
                for b, a in zip(src_base, src_ablated)
            ],
        },
    }
    ablation_output.write_text(json.dumps(ablation_results, indent=2))
    print(f"Ablation results saved → {ablation_output}")


if __name__ == "__main__":
    main()
