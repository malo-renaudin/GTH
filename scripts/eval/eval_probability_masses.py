#!/usr/bin/env python3
"""
Evaluate probability mass of NP / VP / question-mark continuations at gap sites.

Usage:
  python scripts/eval/eval_probability_masses.py \\
        --checkpoint <hf-checkpoint> \\
        --orc_test <orc_test.txt> --wh_test <wh_test.txt> \\
        --out results.csv [--amp] [--compile]

Vocabularies match generate_simple_datasets/orc.py and wh.py exactly.

GPU acceleration
----------------
* KV-cache reuse: each context sentence is run through the model exactly once;
  all candidate sets (NP, VP, ?) are scored by expanding that single cache to
  the candidate batch size and processing only the (short) candidate tokens.
  This avoids re-running the full context for every candidate group.
* Autocast (AMP): --amp enables torch.autocast (float16/bfloat16) inference
  for ~2x throughput on modern GPUs with negligible accuracy change.
* torch.compile: --compile wraps the model with torch.compile (PyTorch >= 2.0)
  for an additional ~1.5-2x speedup after a one-time warm-up.

Notes:
- ORC gap site: right after the first verb in the sentence.
- WH gap site: right after the final lexical token, immediately before '?'.
- Category mass for a sentence = mean of per-candidate geometric-mean
  probabilities, i.e. (1/N) * sum_c exp( (1/len_c) * sum_i log p(t_i|ctx) ).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# ORC vocabulary — verbatim from generate_simple_datasets/orc.py
# ---------------------------------------------------------------------------
_n1_opts_orc: List[str] = [
    "boy", "student", "doctor", "artist", "athlete",
    "teacher", "lawyer", "farmer", "writer", "cook",
]
_n2_opts_orc: List[str] = [
    "girl", "child", "pilot", "scientist", "engineer",
    "reporter", "officer", "manager", "dancer", "chef",
]
_v_opts_orc: List[Tuple[str, str]] = [
    ("visits",   "visit"),
    ("helps",    "help"),
    ("avoids",   "avoid"),
    ("follows",  "follow"),
    ("greets",   "greet"),
    ("contacts", "contact"),
    ("guides",   "guide"),
    ("assists",  "assist"),
    ("teaches",  "teach"),
    ("notices",  "notice"),
]
_adj1_opts_orc: List[str] = [
    "", "big", "tall", "young", "strong", "kind",
    "old", "quiet", "cheerful", "serious",
]
_adj2_opts_orc: List[str] = [
    "", "beautiful", "smart", "brave", "famous", "honest",
    "wise", "gentle", "curious", "experienced",
]
_cont_sing_orc: List[str] = [
    "is eating",  "is watching",  "is reading",
    "likes",       "enjoys",
    "is taking",  "is cooking",   "is playing",  "is practicing",
]
_cont_plur_orc: List[str] = [
    "are eating", "are watching", "are reading",
    "like",        "enjoy",
    "are taking", "are cooking",  "are playing", "are practicing",
]

# ---------------------------------------------------------------------------
# WH vocabulary — verbatim from generate_simple_datasets/wh.py
# ---------------------------------------------------------------------------
_n1_opts_wh: List[str] = [
    # original 20
    "student",      "doctor",       "pilot",        "officer",      "athlete",
    "artist",       "teacher",      "lawyer",        "judge",        "engineer",
    "scientist",    "nurse",         "chef",          "manager",      "reporter",
    "banker",       "professor",     "coach",         "captain",      "writer",
    # 40 additions
    "accountant",   "architect",    "detective",     "diplomat",     "editor",
    "firefighter",  "librarian",    "mechanic",      "musician",     "pharmacist",
    "photographer", "politician",   "programmer",    "psychologist", "surgeon",
    "technician",   "therapist",    "veterinarian",  "administrator","inspector",
    "electrician",  "consultant",   "counselor",     "analyst",      "designer",
    "director",     "instructor",   "investigator",  "planner",      "producer",
    "specialist",   "supervisor",   "translator",    "auditor",      "biologist",
    "chemist",      "economist",    "historian",     "geographer",   "philosopher",
]  # 60 nouns
_n2_opts_wh: List[str] = [
    # original 20
    "child",        "girl",         "boy",           "patient",      "client",
    "tourist",      "neighbor",     "guest",          "worker",       "driver",
    "visitor",      "customer",     "assistant",      "colleague",    "intern",
    "resident",     "passenger",    "volunteer",      "rookie",       "spectator",
    # 40 additions
    "beginner",     "bystander",    "citizen",        "contestant",   "dancer",
    "employee",     "expert",       "journalist",     "observer",     "pedestrian",
    "scholar",      "singer",       "soldier",        "stranger",     "participant",
    "researcher",   "civilian",     "cadet",           "apprentice",   "recruit",
    "traveler",     "witness",      "refugee",        "activist",     "explorer",
    "migrant",      "pupil",        "disciple",       "follower",     "trainee",
    "listener",     "newcomer",     "attendee",       "initiate",     "learner",
    "mentee",       "delegate",     "guardian",       "correspondent","subordinate",
]  # 60 nouns
_v_opts_wh: List[Tuple[str, str, str]] = [
    # (base, past/pp, progressive)
    # Non-animate verbs
    ("visit",     "visited",      "visiting"),
    ("follow",    "followed",     "following"),
    ("avoid",     "avoided",      "avoiding"),
    ("observe",   "observed",     "observing"),
    ("watch",     "watched",      "watching"),
    ("teach",     "taught",       "teaching"),
    ("accompany", "accompanied",  "accompanying"),
    # Animate-only verbs
    ("help",      "helped",       "helping"),
    ("greet",     "greeted",      "greeting"),
    ("call",      "called",       "calling"),
    ("guide",     "guided",       "guiding"),
    ("train",     "trained",      "training"),
    ("support",   "supported",    "supporting"),
    ("consult",   "consulted",    "consulting"),
    ("assist",    "assisted",     "assisting"),
]  # 15 verbs
_adj1_opts_wh: List[str] = [
    "", "young", "tall", "smart", "brave", "kind",
    "famous", "strong", "busy", "calm", "honest", "proud",
    "gentle", "fair", "eager", "skilled",
]  # 16 options
_adj2_opts_wh: List[str] = [
    "", "creative", "serious", "friendly", "quiet", "active",
    "careful", "thoughtful", "diligent", "talented", "ambitious",
    "capable", "determined", "experienced", "motivated",
]  # 15 options
_adv1_opts_wh: List[str] = [
    "", "probably", "certainly", "possibly", "apparently", "maybe",
    "perhaps", "clearly", "definitely", "obviously", "surely",
    "truly", "reportedly",
]  # 13 options


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_IRREGULARS = {"child": "children", "man": "men", "woman": "women"}


def pluralize_noun(noun: str) -> str:
    return _IRREGULARS.get(noun, noun + "s")


def ids_for_text(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _load_model_and_tokenizer(
    checkpoint: str, device: str, compile_model: bool = False
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
    model.eval()
    model.to(device)
    if compile_model:
        model = torch.compile(model)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Candidate scorer — KV-cache version
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_candidates(
    context_ids: List[int],
    candidate_ids_list: List[List[int]],
    tokenizer,
    model,
    device: str,
    amp_ctx,
    batch_size: int = 64,
) -> List[Tuple[float, int]]:
    """Score candidates using KV-cache: encode context once, then score only candidate tokens.

    Returns a list of (log_prob_sum, token_count) in candidate order.
    """
    import copy

    results: List[Tuple[float, int]] = []
    ctx_len = len(context_ids)

    # Encode context once; save KV cache and the last logit (predicts first candidate token)
    ctx_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
    with amp_ctx:
        ctx_out = model(ctx_tensor, use_cache=True)
    ctx_last_logit = ctx_out.logits[0, -1, :].float()  # [V]
    base_cache = ctx_out.past_key_values

    for i in range(0, len(candidate_ids_list), batch_size):
        batch = candidate_ids_list[i : i + batch_size]
        bsz = len(batch)
        max_cand_len = max((len(c) for c in batch), default=0)

        if max_cand_len == 0:
            results.extend((0.0, 0) for _ in batch)
            continue

        # Copy and expand the context KV cache for this batch
        cache = copy.deepcopy(base_cache)
        cache.batch_repeat_interleave(bsz)

        # Build padded candidate input_ids and mask
        input_ids = torch.full(
            (bsz, max_cand_len), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        cand_mask = torch.zeros(bsz, max_cand_len, dtype=torch.long, device=device)
        for j, cand in enumerate(batch):
            if cand:
                input_ids[j, :len(cand)] = torch.tensor(cand, dtype=torch.long, device=device)
                cand_mask[j, :len(cand)] = 1

        # Full attention mask: all context tokens attended + candidate mask
        attention_mask = torch.cat(
            [torch.ones(bsz, ctx_len, dtype=torch.long, device=device), cand_mask],
            dim=1,
        )

        with amp_ctx:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=False,
            )

        # ctx_last_logit predicts cand[0]; out.logits[:, m] predicts cand[m+1]
        # Concatenate to get one logit per candidate token position
        ctx_logit_exp = ctx_last_logit.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1)  # [bsz, 1, V]
        all_logits = torch.cat(
            [ctx_logit_exp, out.logits[:, :-1, :].float()], dim=1
        )  # [bsz, max_cand_len, V]

        lprobs = torch.nn.functional.log_softmax(all_logits, dim=-1)
        gold_lp = torch.gather(lprobs, 2, input_ids.unsqueeze(-1)).squeeze(-1)  # [bsz, max_cand_len]

        log_sums = (gold_lp * cand_mask).sum(dim=1)
        tok_counts = cand_mask.sum(dim=1)

        for j in range(bsz):
            results.append((float(log_sums[j].item()), int(tok_counts[j].item())))

    return results


# ---------------------------------------------------------------------------
# Original batch scorer — kept for backward compatibility
# (does not use KV caching; re-runs the full context on every forward pass)
# ---------------------------------------------------------------------------
def compute_candidates_log_sums(
    context_ids: List[int],
    candidate_ids_list: List[List[int]],
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
) -> List[Tuple[float, int]]:
    """
    For each candidate (given as token id list), compute total log-probability
    of the candidate tokens conditioned on the context. Returns list of
    (log_prob_sum, token_count) for each candidate in the same order.

    This is batched: each model forward computes probabilities for
    context + candidate sequence.
    """
    results: List[Tuple[float, int]] = []
    ctx_len = len(context_ids)

    # prepare batches
    for i in range(0, len(candidate_ids_list), batch_size):
        batch = candidate_ids_list[i : i + batch_size]
        seqs = [context_ids + cand for cand in batch]
        max_len = max(len(s) for s in seqs)

        input_ids = torch.full((len(seqs), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)

        for j, s in enumerate(seqs):
            input_ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[j, : len(s)] = 1

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, L, V)

            # shift logits and labels to compute p(token_i | prev)
            logits_pred = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            lprobs = torch.nn.functional.log_softmax(logits_pred, dim=-1)

            # gather log-probs for the gold labels
            gold_token_logprobs = torch.gather(lprobs, 2, labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

            # positions (in labels) correspond to original input indices 1..L-1
            positions = torch.arange(1, max_len, device=device).unsqueeze(0)  # (1, L-1)
            mask_valid = attention_mask[:, 1:].bool()
            mask_candidate = (positions >= ctx_len) & mask_valid

            # sum only candidate positions
            log_sums = (gold_token_logprobs * mask_candidate).sum(dim=1)
            token_counts = mask_candidate.sum(dim=1)

            # convert to python types
            for k in range(len(batch)):
                results.append((float(log_sums[k].item()), int(token_counts[k].item())))

    return results


# ---------------------------------------------------------------------------
# Candidate builders
# ---------------------------------------------------------------------------

def build_orc_candidates():
    """Return (np_candidates, vp_candidates, q_candidate) for ORC evaluation.

    NP (760 candidates): "the [adj] noun" — both adj lists (non-empty entries)
      crossed with all N1/N2 nouns in singular and plural form.
    VP (18 candidates): verb-only forms from CONT_SING + CONT_PLUR (no objects).
    Q:  ["?"]
    """
    adj_options = (
        [""]
        + [a for a in _adj1_opts_orc if a]
        + [a for a in _adj2_opts_orc if a]
    )
    nouns = list(_n1_opts_orc) + list(_n2_opts_orc)
    np_candidates: List[str] = []
    for adj in adj_options:
        for n in nouns:
            for form in (n, pluralize_noun(n)):
                np_candidates.append(
                    f"the {adj} {form}" if adj else f"the {form}"
                )
    vp_candidates = list(_cont_sing_orc) + list(_cont_plur_orc)
    return np_candidates, vp_candidates, ["?"]


def build_wh_candidates():
    """Return (np_candidates, vp_candidates, q_candidate) for WH evaluation.

    NP (7200 candidates): "the [adj] noun" — both adj lists crossed with all
      N1/N2 nouns (60 each) in singular and plural form.
    VP (~120 candidates): aux x verb-form combinations from wh.py's v_opts.
    Q:  ["?"]
    """
    adj_options = (
        [""]
        + [a for a in _adj1_opts_wh if a]
        + [a for a in _adj2_opts_wh if a]
    )
    nouns = list(_n1_opts_wh) + list(_n2_opts_wh)
    np_candidates: List[str] = []
    for adj in adj_options:
        for n in nouns:
            for form in (n, pluralize_noun(n)):
                np_candidates.append(
                    f"the {adj} {form}" if adj else f"the {form}"
                )
    vp_candidates: List[str] = []
    for base, past, ing in _v_opts_wh:
        vp_candidates.extend([
            base, past, ing,
            f"did {base}", f"does {base}", f"do {base}",
            f"is {ing}", f"are {ing}",
        ])
    seen: set = set()
    vp_candidates = [x for x in vp_candidates if not (x in seen or seen.add(x))]
    return np_candidates, vp_candidates, ["?"]


# ---------------------------------------------------------------------------
# Context extractors
# ---------------------------------------------------------------------------

def find_orc_context(sentence: str) -> str:
    """Return the sentence up to and including the first ORC verb form."""
    lower = sentence.lower()
    verb_forms = {vf for sing, base in _v_opts_orc for vf in (sing, base)}

    earliest: Optional[int] = None
    match_end: Optional[int] = None
    for vf in verb_forms:
        m = re.search(r"\b" + re.escape(vf) + r"\b", lower)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
            match_end = m.end()

    return sentence[:match_end].rstrip() if earliest is not None else sentence.strip()


def find_wh_context(sentence: str) -> str:
    """Return everything before the final '?' (the pre-gap context)."""
    pos = sentence.rfind("?")
    return sentence[:pos].rstrip() if pos != -1 else sentence.strip()


# ---------------------------------------------------------------------------
# High-level API (kept for external callers)
# ---------------------------------------------------------------------------

def build_vocabulary() -> dict:
    orc_np, orc_vp, orc_q = build_orc_candidates()
    wh_np,  wh_vp,  wh_q  = build_wh_candidates()
    return {
        "orc": {"np": orc_np, "vp": orc_vp, "q": orc_q},
        "wh":  {"np": wh_np,  "vp": wh_vp,  "q": wh_q},
    }


def get_orc_context(sentence: str) -> str:
    return find_orc_context(sentence)


def get_wh_context(sentence: str) -> str:
    return find_wh_context(sentence)


# ---------------------------------------------------------------------------
# Per-sentence evaluation
# ---------------------------------------------------------------------------

def _eval_sentence_cached(
    sent: str,
    context_fn,
    np_ids: List[List[int]],
    vp_ids: List[List[int]],
    q_ids: List[List[int]],
    tokenizer,
    model,
    device: str,
    amp_ctx,
    batch_size: int,
) -> dict:
    """Score one sentence; return a result dict."""
    context = context_fn(sent)
    context_ids = ids_for_text(tokenizer, context.rstrip() + " ")

    # NP
    np_results = _score_candidates(
        context_ids, np_ids, tokenizer, model, device, amp_ctx, batch_size
    )
    np_geom = [math.exp(ls / cnt) if cnt > 0 else 0.0 for ls, cnt in np_results]
    np_mass = sum(np_geom) / len(np_geom) if np_geom else 0.0

    # VP
    vp_results = _score_candidates(
        context_ids, vp_ids, tokenizer, model, device, amp_ctx, batch_size
    )
    vp_geom = [math.exp(ls / cnt) if cnt > 0 else 0.0 for ls, cnt in vp_results]
    vp_mass = sum(vp_geom) / len(vp_geom) if vp_geom else 0.0

    # Question mark
    q_result = _score_candidates(
        context_ids, q_ids, tokenizer, model, device, amp_ctx, batch_size=1
    )
    q_geom = math.exp(q_result[0][0] / q_result[0][1]) if q_result[0][1] > 0 else 0.0

    return {
        "sentence": sent,
        "context":  context,
        "np_mass":  float(np_mass),
        "vp_mass":  float(vp_mass),
        "q_mass":   float(q_geom),
        "np_count": len(np_ids),
        "vp_count": len(vp_ids),
    }


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------

def process_dataset(
    file_path: str,
    model,
    tokenizer,
    vocab: dict,
    context_fn,
    batch_size: int = 64,
    sent_limit: Optional[int] = None,
    amp: bool = False,
) -> dict:
    """Process a test file and return aggregated probability masses for NP/VP/?.

    Args:
        file_path:   path to the test file (one sentence per line).
        model:       pre-loaded HF causal LM (already on the correct device).
        tokenizer:   corresponding tokenizer.
        vocab:       dict with keys 'np', 'vp', 'q' (lists of candidate strings).
        context_fn:  callable(sentence) -> context substring.
        batch_size:  candidate batch size for GPU scoring.
        sent_limit:  optional cap on number of sentences to process.
        amp:         enable torch.autocast for float16/bfloat16 inference.

    Returns:
        Dict with keys 'NP', 'VP', '?' mapping to mean category masses.
    """
    try:
        device = str(next(model.parameters()).device)
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_type = "cuda" if "cuda" in device else "cpu"
    amp_ctx = (
        torch.autocast(device_type=device_type)
        if amp and device_type == "cuda"
        else nullcontext()
    )

    np_ids = [ids_for_text(tokenizer, c) for c in vocab["np"]]
    vp_ids = [ids_for_text(tokenizer, c) for c in vocab["vp"]]
    q_ids  = [ids_for_text(tokenizer, c) for c in vocab["q"]]

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if sent_limit:
        lines = lines[:sent_limit]

    rows = []
    for sent in lines:
        r = _eval_sentence_cached(
            sent, context_fn, np_ids, vp_ids, q_ids,
            tokenizer, model, device, amp_ctx, batch_size,
        )
        r["file"] = file_path
        r["idx"]  = len(rows)
        rows.append(r)

    def _mean(key: str) -> float:
        return sum(x[key] for x in rows) / len(rows) if rows else 0.0

    return {"NP": _mean("np_mass"), "VP": _mean("vp_mass"), "?": _mean("q_mass")}


def evaluate_sentences(
    filename: str,
    domain: str,
    tokenizer,
    model,
    device: str,
    out_rows: List[dict],
    sent_limit: Optional[int] = None,
    cand_batch_size: int = 64,
    amp: bool = False,
) -> None:
    """Evaluate all sentences in *filename* (domain='orc' or 'wh').

    Results are appended to *out_rows* as dicts with keys:
    file, idx, sentence, context, np_mass, vp_mass, q_mass, np_count, vp_count.
    """
    if domain == "orc":
        np_cands, vp_cands, q_cand = build_orc_candidates()
        context_fn = find_orc_context
    else:
        np_cands, vp_cands, q_cand = build_wh_candidates()
        context_fn = find_wh_context

    device_type = "cuda" if "cuda" in device else "cpu"
    amp_ctx = (
        torch.autocast(device_type=device_type)
        if amp and device_type == "cuda"
        else nullcontext()
    )

    # Pre-tokenize candidates once (shared across all sentences)
    np_ids = [ids_for_text(tokenizer, c) for c in np_cands]
    vp_ids = [ids_for_text(tokenizer, c) for c in vp_cands]
    q_ids  = [ids_for_text(tokenizer, c) for c in q_cand]

    with open(filename, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if sent_limit:
        lines = lines[:sent_limit]

    for i, sent in enumerate(tqdm.tqdm(lines, desc=f"Eval {domain}")):
        r = _eval_sentence_cached(
            sent, context_fn, np_ids, vp_ids, q_ids,
            tokenizer, model, device, amp_ctx, cand_batch_size,
        )
        r["file"] = filename
        r["idx"]  = i
        out_rows.append(r)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate probability masses at ORC/WH gap sites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="HF model id or path")
    p.add_argument("--orc_test",   required=True, help="ORC test set (one sent/line)")
    p.add_argument("--wh_test",    required=True, help="WH test set (one sent/line)")
    p.add_argument("--out", default="prob_masses_results.csv", help="CSV output path")
    p.add_argument("--device",     default=None,
                   help="torch device, e.g. cpu or cuda (auto-detected if omitted)")
    p.add_argument("--sent_limit", type=int, default=None,
                   help="Cap sentences per file (for debugging)")
    p.add_argument("--cand_batch_size", type=int, default=64,
                   help="Candidate batch size; increase for more GPU throughput")
    p.add_argument("--amp", action="store_true",
                   help="Enable torch.autocast (float16/bf16) on CUDA — ~2x faster")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (PyTorch >= 2.0) — ~1.5-2x after warm-up")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_model_and_tokenizer(
        args.checkpoint, device, compile_model=args.compile
    )

    rows: List[dict] = []
    evaluate_sentences(
        args.orc_test, "orc", tokenizer, model, device, rows,
        sent_limit=args.sent_limit, cand_batch_size=args.cand_batch_size, amp=args.amp,
    )
    evaluate_sentences(
        args.wh_test, "wh", tokenizer, model, device, rows,
        sent_limit=args.sent_limit, cand_batch_size=args.cand_batch_size, amp=args.amp,
    )

    fieldnames = [
        "file", "idx", "sentence", "context",
        "np_mass", "vp_mass", "q_mass", "np_count", "vp_count",
    ]
    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    orc_rows = [r for r in rows if r["file"] == args.orc_test]
    wh_rows  = [r for r in rows if r["file"] == args.wh_test]

    def mean(lst: List[dict], key: str) -> float:
        return sum(x[key] for x in lst) / len(lst) if lst else 0.0

    print("Saved results to:", args.out)
    print("ORC avg masses: NP=%.6f  VP=%.6f  Q=%.6f" % (
        mean(orc_rows, "np_mass"), mean(orc_rows, "vp_mass"), mean(orc_rows, "q_mass")))
    print("WH  avg masses: NP=%.6f  VP=%.6f  Q=%.6f" % (
        mean(wh_rows,  "np_mass"), mean(wh_rows,  "vp_mass"), mean(wh_rows,  "q_mass")))


if __name__ == "__main__":
    main()
