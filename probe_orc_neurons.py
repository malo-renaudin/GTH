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


def find_post_rel_token_pos(
    spans: List[Tuple[str, int, int]]
) -> Optional[int]:
    """Return the last-token index of the word immediately after 'that'/'who'."""
    rel_idx = find_rel_marker_span_idx(spans)
    if rel_idx is None or rel_idx + 1 >= len(spans):
        return None
    _, _, end = spans[rel_idx + 1]
    return end - 1


# ── Hidden-state collection ───────────────────────────────────────────────────

def collect_hidden_states(
    model: GPT,
    tokenizer: Tokenizer,
    sentences: List[str],
    device: torch.device,
) -> Tuple[np.ndarray, List[Optional[int]]]:
    """
    For each sentence:
      1. Find the token position of 'that'/'who'.
      2. Run the model on tokens[0 : rel_pos + 1].
      3. Capture every transformer block's output at rel_pos.

    Returns:
      activations  : float32 array, shape (n_valid, n_layers * n_embd)
      next_tok_ids : first token of the word immediately after the relativizer
    """
    blocks = get_transformer_blocks(model)
    n_layers = len(blocks)

    layer_captures: List[Optional[torch.Tensor]] = [None] * n_layers
    target_pos = [0]  # mutable cell updated before each forward pass

    def make_hook(layer_idx: int):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_captures[layer_idx] = h[0, target_pos[0], :].detach().cpu()
        return hook

    handles = [b.register_forward_hook(make_hook(i)) for i, b in enumerate(blocks)]

    all_activations: List[Optional[np.ndarray]] = []
    next_tok_ids:    List[Optional[int]]         = []

    try:
        for sent in tqdm(sentences, desc="  collecting activations", leave=False):
            tokens, spans = tokenize_sentence(sent, tokenizer)
            rel_idx = find_rel_marker_span_idx(spans)
            if rel_idx is None or rel_idx + 1 >= len(spans):
                all_activations.append(None)
                next_tok_ids.append(None)
                continue

            # Probe at the word immediately after the relativizer
            _, _, probe_end = spans[rel_idx + 1]
            probe_pos = probe_end - 1

            # Next token id: first token of the word two positions after the relativizer
            next_tok_id = tokens[spans[rel_idx + 2][1]] if rel_idx + 2 < len(spans) else None

            target_pos[0] = probe_pos
            x = torch.tensor(tokens[: probe_pos + 1], device=device).unsqueeze(0)
            with torch.no_grad():
                _ = model(x)

            # Fallback dim in case a layer wasn't triggered (shouldn't happen)
            fallback_dim = next(
                (layer_captures[i].shape[0] for i in range(n_layers) if layer_captures[i] is not None),
                1,
            )
            vec = np.concatenate([
                layer_captures[i].numpy()
                if layer_captures[i] is not None
                else np.zeros(fallback_dim, dtype=np.float32)
                for i in range(n_layers)
            ])
            all_activations.append(vec)
            next_tok_ids.append(next_tok_id)
    finally:
        for h in handles:
            h.remove()

    valid = [(a, t) for a, t in zip(all_activations, next_tok_ids) if a is not None]
    if not valid:
        raise ValueError("No 'that'/'who' found in any sentence.")
    return (
        np.stack([v[0] for v in valid]).astype(np.float32),
        [v[1] for v in valid],
    )


# ── Next-token probability (with optional ablation) ───────────────────────────

def collect_next_token_probs(
    model: GPT,
    tokenizer: Tokenizer,
    sentences: List[str],
    device: torch.device,
    ablate_dims: Optional[Dict[int, List[int]]] = None,
) -> List[Optional[float]]:
    """
    For each sentence, run the model up to and including the relativizer,
    optionally zero-out ablate_dims[layer] at that position in each layer's
    output, and return P(first token of the next word).
    """
    blocks = get_transformer_blocks(model)
    handles: List[torch.utils.hooks.RemovableHook] = []
    target_pos = [0]

    if ablate_dims:
        def make_ablation_hook(layer_idx: int, dims: List[int]):
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                h[0, target_pos[0], dims] = 0.0
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook

        for layer_idx, dims in ablate_dims.items():
            if dims:
                handles.append(
                    blocks[layer_idx].register_forward_hook(
                        make_ablation_hook(layer_idx, dims)
                    )
                )

    probs: List[Optional[float]] = []
    try:
        for sent in tqdm(sentences, desc="  next-token probs", leave=False):
            tokens, spans = tokenize_sentence(sent, tokenizer)
            rel_idx = find_rel_marker_span_idx(spans)
            if rel_idx is None or rel_idx + 1 >= len(spans):
                probs.append(None)
                continue

            _, _, probe_end = spans[rel_idx + 1]
            probe_pos = probe_end - 1

            if rel_idx + 2 >= len(spans):
                probs.append(None)
                continue
            next_tok_id = tokens[spans[rel_idx + 2][1]]

            target_pos[0] = probe_pos
            x = torch.tensor(tokens[: probe_pos + 1], device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)[0, -1, :]
            p = torch.softmax(logits, dim=-1)
            probs.append(p[next_tok_id].item())
    finally:
        for h in handles:
            h.remove()

    return probs


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
    print("\n[1/3] Collecting hidden states at relativizer position …")
    print("  SRC:")
    src_acts, _ = collect_hidden_states(model, tokenizer, src_sents, device)
    print("  ORC:")
    orc_acts, _ = collect_hidden_states(model, tokenizer, orc_sents, device)

    n_neurons = src_acts.shape[1]
    n_embd    = n_neurons // n_layers
    print(f"  hidden dim per layer: {n_embd}, total neurons: {n_neurons}")

    # ── 2. Welch t-test per neuron ────────────────────────────────────────────
    print("\n[2/3] Running Welch t-tests (ORC vs SRC) …")
    t_stats = np.empty(n_neurons, dtype=np.float64)
    p_vals  = np.empty(n_neurons, dtype=np.float64)
    for i in range(n_neurons):
        t, p = stats.ttest_ind(orc_acts[:, i], src_acts[:, i], equal_var=False)
        t_stats[i] = t
        p_vals[i]  = p

    pos_idx        = np.where(t_stats > 0)[0]
    pos_idx_sorted = pos_idx[np.argsort(t_stats[pos_idx])[::-1]]   # descending by t
    print(f"  Neurons with t > 0 (ORC > SRC): {len(pos_idx_sorted)} / {n_neurons}")

    neuron_records = [
        {
            "flat_idx": int(idx),
            "layer":    int(idx // n_embd),
            "dim":      int(idx % n_embd),
            "t_stat":   float(t_stats[idx]),
            "p_value":  float(p_vals[idx]),
        }
        for idx in pos_idx_sorted
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(neuron_records, indent=2))
    print(f"  Saved neuron list → {args.output}")

    # ── 3. Ablation of top-N% neurons ────────────────────────────────────────
    n_top   = max(1, int(np.ceil(args.top_pct / 100.0 * n_neurons)))
    top_idx = pos_idx_sorted[:n_top]
    print(f"\n[3/3] Ablating top {args.top_pct}% = {n_top} neurons …")

    if neuron_records:
        top5 = [(r["layer"], r["dim"], round(r["t_stat"], 3)) for r in neuron_records[:5]]
        print(f"  Top-5 neurons (layer, dim, t): {top5}")

    # Group ablated dims by layer
    ablate_dims: Dict[int, List[int]] = {}
    for idx in top_idx:
        layer = int(idx // n_embd)
        dim   = int(idx % n_embd)
        ablate_dims.setdefault(layer, []).append(dim)

    print("  Baseline (no ablation):")
    orc_base = collect_next_token_probs(model, tokenizer, orc_sents, device)
    src_base = collect_next_token_probs(model, tokenizer, src_sents, device)

    print("  Ablated:")
    orc_ablated = collect_next_token_probs(model, tokenizer, orc_sents, device, ablate_dims)
    src_ablated = collect_next_token_probs(model, tokenizer, src_sents, device, ablate_dims)

    def mean_nonnull(lst: List[Optional[float]]) -> float:
        v = [x for x in lst if x is not None]
        return float(np.mean(v)) if v else float("nan")

    orc_b, orc_a = mean_nonnull(orc_base), mean_nonnull(orc_ablated)
    src_b, src_a = mean_nonnull(src_base), mean_nonnull(src_ablated)

    print("\n=== Ablation results ===")
    print(f"{'Condition':30s} {'Baseline':>10} {'Ablated':>10} {'Delta':>10}")
    print(f"{'ORC  P(next word)':30s} {orc_b:10.6f} {orc_a:10.6f} {orc_a - orc_b:+10.6f}")
    print(f"{'SRC  P(next word)':30s} {src_b:10.6f} {src_a:10.6f} {src_a - src_b:+10.6f}")

    ablation_output = args.output.parent / (args.output.stem + "_ablation.json")
    ablation_results = {
        "n_ablated_neurons": n_top,
        "top_pct":           args.top_pct,
        "orc": {
            "baseline_mean_prob": orc_b,
            "ablated_mean_prob":  orc_a,
            "delta":              orc_a - orc_b,
            "per_sentence": [
                {"baseline": b, "ablated": a}
                for b, a in zip(orc_base, orc_ablated)
            ],
        },
        "src": {
            "baseline_mean_prob": src_b,
            "ablated_mean_prob":  src_a,
            "delta":              src_a - src_b,
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
