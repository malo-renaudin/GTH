#!/usr/bin/env python3
"""
Evaluate probability mass of NP / VP / question-mark continuations at gap sites.

Usage:
  python scripts/eval/eval_probability_masses --checkpoint <hf-checkpoint> \
       --orc_test <orc_test.txt> --wh_test <wh_test.txt> --out results.csv

This script uses the small generation vocabularies from the project's
generation scripts (copied here) and evaluates, for each test sentence,
the average geometric-mean probability (per-token) assigned by the model
to three continuation categories at the gap site: NP, VP and "?".

Notes:
- ORC gap site: right after the first verb in the sentence.
- WH gap site: right after the final lexical token, immediately before '?'.
- NP candidates all start with "the"; the script computes P("the"|context)
  once per context and reuses that value when evaluating NP continuations.

The measured category mass for a sentence is the mean of per-candidate
geometric means, i.e. (1/N) * sum_c (prod_i p(t_i|ctx)^(1/len_c)).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from typing import List, Tuple

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- Vocabulary copied (unchanged) from the generation scripts ---
# ORC vocabulary (from generate_simple_datasets/orc.py)
n1_opts_orc = ["boy", "student", "doctor", "artist", "athlete"]
n2_opts_orc = ["girl", "child", "pilot", "scientist", "engineer"]
v_opts_orc = [("visits", "visit"), ("helps", "help"), ("avoids", "avoid"), ("follows", "follow"), ("greets", "greet")]
adj1_opts_orc = ["big", "tall", "young", "strong", "kind"]
adj2_opts_orc = ["beautiful", "smart", "brave", "famous", "honest"]
adv1_opts_orc = ["possibly", "apparently", "secretly", "always", "often", "rarely"]
cont_sing_orc = ["is eating an apple", "is watching a movie", "is reading a book", "likes to dance", "enjoys music", "likes climbing"]
cont_plur_orc = ["are eating an apple", "are watching a movie", "are reading a book", "like to dance", "enjoy music", "like climbing"]

# WH vocabulary (from generate_simple_datasets/wh.py)
n1_opts_wh = ["student", "doctor", "pilot", "officer", "athlete", "artist"]
n2_opts_wh = ["child", "girl", "boy", "patient", "client", "tourist"]
v_opts_wh = [
    ("visit", "visited", "visiting"),
    ("help", "helped", "helping"),
    ("greet", "greeted", "greeting"),
    ("follow", "followed", "following"),
    ("avoid", "avoided", "avoiding"),
    ("call", "called", "calling"),
    ("observe", "observed", "observing"),
]
adj1_opts_wh = ["young", "tall", "smart", "brave", "kind", "famous"]
adj2_opts_wh = ["creative", "serious", "friendly", "quiet", "active"]
adv1_opts_wh = ["probably", "certainly", "possibly", "apparently", "maybe", "perhaps"]


def pluralize_noun(noun: str) -> str:
    irregulars = {"child": "children", "man": "men", "woman": "women"}
    return irregulars.get(noun, noun + "s")


def _load_model_and_tokenizer(checkpoint: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # make sure we have a pad token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.resize_token_embeddings(len(tokenizer))
        else:
            # last resort: set pad token to a new token
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

    model.eval()
    model.to(device)
    return tokenizer, model


def _concat_ids(a: List[int], b: List[int]) -> List[int]:
    return a + b


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


def build_orc_candidates():
    # NP candidates: "the" + optional adj (from both adj lists) + noun (sing & plur)
    adj_options = [""] + adj1_opts_orc + adj2_opts_orc
    nouns = list(n1_opts_orc) + list(n2_opts_orc)
    np_candidates = []
    for adj in adj_options:
        for n in nouns:
            for noun_variant in (n, pluralize_noun(n)):
                if adj:
                    np_candidates.append(f"the {adj} {noun_variant}")
                else:
                    np_candidates.append(f"the {noun_variant}")

    # VP candidates: use the continuations used by the generator
    vp_candidates = list(cont_sing_orc) + list(cont_plur_orc)

    # question mark candidate
    q_candidate = ["?"]

    return np_candidates, vp_candidates, q_candidate


def build_wh_candidates():
    adj_options = [""] + adj1_opts_wh + adj2_opts_wh
    nouns = list(n1_opts_wh) + list(n2_opts_wh)
    np_candidates = []
    for adj in adj_options:
        for n in nouns:
            for noun_variant in (n, pluralize_noun(n)):
                if adj:
                    np_candidates.append(f"the {adj} {noun_variant}")
                else:
                    np_candidates.append(f"the {noun_variant}")

    # VP candidates: combine auxiliaries and verb forms from the generator
    vp_candidates = []
    for base, past, ing in v_opts_wh:
        vp_candidates.extend([base, past, ing, f"did {base}", f"does {base}", f"do {base}", f"is {ing}", f"are {ing}"])
    # deduplicate while preserving order
    seen = set()
    vp_candidates = [x for x in vp_candidates if not (x in seen or seen.add(x))]

    q_candidate = ["?"]
    return np_candidates, vp_candidates, q_candidate


def find_orc_context(sentence: str) -> str:
    """Return substring up to and including the first verb (based on v_opts_orc).
    If no verb is found, return the whole sentence as a fallback.
    """
    lower = sentence.lower()
    verb_forms = set()
    for sing, base in v_opts_orc:
        verb_forms.add(sing)
        verb_forms.add(base)

    earliest = None
    match_end = None
    for vf in verb_forms:
        m = re.search(r"\b" + re.escape(vf) + r"\b", lower)
        if m:
            if earliest is None or m.start() < earliest:
                earliest = m.start()
                match_end = m.end()

    if earliest is None:
        # fallback: use first verb-like token (very unlikely for generated data)
        return sentence.strip()
    return sentence[:match_end].rstrip()


def find_wh_context(sentence: str) -> str:
    # context is everything before the final '?'
    pos = sentence.rfind("?")
    if pos == -1:
        return sentence.strip()
    return sentence[:pos].rstrip()


def ids_for_text(tokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def build_vocabulary():
    """Return a dict with prebuilt candidate lists for ORC and WH evaluations.

    The callback expects build_vocabulary()["orc"] and build_vocabulary()["wh"]
    to be passed to `process_dataset()`.
    """
    orc_np, orc_vp, orc_q = build_orc_candidates()
    wh_np, wh_vp, wh_q = build_wh_candidates()
    return {
        "orc": {"np": orc_np, "vp": orc_vp, "q": orc_q},
        "wh":  {"np": wh_np,  "vp": wh_vp,  "q":  wh_q},
    }


def get_orc_context(sentence: str) -> str:
    return find_orc_context(sentence)


def get_wh_context(sentence: str) -> str:
    return find_wh_context(sentence)


def process_dataset(
    file_path: str,
    model,
    tokenizer,
    vocab: dict,
    context_fn,
    batch_size: int = 64,
    sent_limit: int = None,
) -> dict:
    """Process a test file and return aggregated probability-mass for NP/VP/? categories.

    Args:
        file_path: path to the test file (one sentence per line).
        model: a pre-loaded HF causal LM (already on the correct device).
        tokenizer: corresponding tokenizer.
        vocab: dictionary with keys 'np', 'vp', 'q' containing candidate strings.
        context_fn: callable(sentence) -> context substring.
        batch_size: candidate batch size for scoring.
        sent_limit: optional limit on number of sentences to process.

    Returns:
        A dict with keys 'NP', 'VP', '?' mapping to mean category masses (floats).
    """
    # determine device from model
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read sentences
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if sent_limit:
        lines = lines[:sent_limit]

    rows = []
    for i, sent in enumerate(lines):
        context = context_fn(sent)
        context_for_tok = context.rstrip()
        context_ids = ids_for_text(tokenizer, context_for_tok + " ")

        # NP category: compute prefix "the" once when useful
        prefix = "the"
        prefix_ids = ids_for_text(tokenizer, prefix)

        # pre-tokenize candidates from the provided vocab
        np_ids = [ids_for_text(tokenizer, c) for c in vocab["np"]]
        vp_ids = [ids_for_text(tokenizer, c) for c in vocab["vp"]]
        q_ids = [ids_for_text(tokenizer, c) for c in vocab["q"]]

        # compute prefix log once
        prefix_log, prefix_toks = compute_candidates_log_sums(context_ids, [prefix_ids], tokenizer, model, device, batch_size=batch_size)[0]

        # group NP candidates that start with prefix
        rest_ids_for_np = []
        rest_map_idx = []
        fallback_full_np = []
        fallback_full_idx = []
        for idx, cids in enumerate(np_ids):
            if len(cids) >= len(prefix_ids) and cids[: len(prefix_ids)] == prefix_ids:
                rest = cids[len(prefix_ids) :]
                rest_ids_for_np.append(rest)
                rest_map_idx.append(idx)
            else:
                fallback_full_np.append(cids)
                fallback_full_idx.append(idx)

        # compute rest log-probs conditioned on context+prefix
        new_ctx_ids = context_ids + prefix_ids
        rest_results = []
        if rest_ids_for_np:
            rest_results = compute_candidates_log_sums(new_ctx_ids, rest_ids_for_np, tokenizer, model, device, batch_size=batch_size)

        np_geom_vals = [0.0] * len(np_ids)
        for k, (logsum_rest, count_rest) in enumerate(rest_results):
            orig_idx = rest_map_idx[k]
            total_log = prefix_log + logsum_rest
            total_tok_count = prefix_toks + count_rest
            if total_tok_count <= 0:
                geom = 0.0
            else:
                geom = math.exp(total_log / float(total_tok_count))
            np_geom_vals[orig_idx] = geom

        # handle fallback full NP computations
        if fallback_full_np:
            fallback_results = compute_candidates_log_sums(context_ids, fallback_full_np, tokenizer, model, device, batch_size=batch_size)
            for k, (logsum_full, count_full) in enumerate(fallback_results):
                orig_idx = fallback_full_idx[k]
                geom = math.exp(logsum_full / float(count_full)) if count_full > 0 else 0.0
                np_geom_vals[orig_idx] = geom

        np_mass = float(sum(np_geom_vals) / len(np_geom_vals)) if np_geom_vals else 0.0

        # VP category
        vp_results = compute_candidates_log_sums(context_ids, vp_ids, tokenizer, model, device, batch_size=batch_size)
        vp_geom = []
        for logsum, cnt in vp_results:
            if cnt > 0:
                vp_geom.append(math.exp(logsum / float(cnt)))
            else:
                vp_geom.append(0.0)
        vp_mass = float(sum(vp_geom) / len(vp_geom)) if vp_geom else 0.0

        # question mark candidate
        q_result = compute_candidates_log_sums(context_ids, q_ids, tokenizer, model, device, batch_size=1)[0]
        q_geom = math.exp(q_result[0] / float(q_result[1])) if q_result[1] > 0 else 0.0

        rows.append({
            "file": file_path,
            "idx": i,
            "sentence": sent,
            "context": context,
            "np_mass": np_mass,
            "vp_mass": vp_mass,
            "q_mass": q_geom,
            "np_count": len(np_ids),
            "vp_count": len(vp_ids),
        })

    # aggregate
    def _mean(l, key):
        return sum(x[key] for x in l) / len(l) if l else 0.0

    result = {"NP": _mean(rows, "np_mass"), "VP": _mean(rows, "vp_mass"), "?": _mean(rows, "q_mass")}
    return result


def evaluate_sentences(
    filename: str,
    domain: str,
    tokenizer,
    model,
    device: str,
    out_rows: List[dict],
    sent_limit: int = None,
    cand_batch_size: int = 64,
):
    if domain == "orc":
        np_cands, vp_cands, q_cand = build_orc_candidates()
    else:
        np_cands, vp_cands, q_cand = build_wh_candidates()

    with open(filename, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    if sent_limit:
        lines = lines[:sent_limit]

    for i, sent in enumerate(tqdm.tqdm(lines, desc=f"Eval {domain}")):
        if domain == "orc":
            context = find_orc_context(sent)
        else:
            context = find_wh_context(sent)

        # ensure one separating space when we append candidates
        context_for_tok = context.rstrip()

        # compute token ids for context
        context_ids = ids_for_text(tokenizer, context_for_tok + " ")

        # --- NP category: optimize by computing prefix "the" once ---
        prefix = "the"
        prefix_ids = ids_for_text(tokenizer, prefix)

        # pre-tokenize candidates
        np_ids = [ids_for_text(tokenizer, c) for c in np_cands]
        vp_ids = [ids_for_text(tokenizer, c) for c in vp_cands]
        q_ids = [ids_for_text(tokenizer, c) for c in q_cand]

        # compute prefix log once
        prefix_log, prefix_toks = compute_candidates_log_sums(context_ids, [prefix_ids], tokenizer, model, device, batch_size=cand_batch_size)[0]

        # group NP candidates that actually start with prefix_ids
        rest_ids_for_np = []
        rest_map_idx = []  # map back index
        fallback_full_np = []  # those that do not start with prefix -- compute as full
        fallback_full_idx = []
        for idx, cids in enumerate(np_ids):
            if len(cids) >= len(prefix_ids) and cids[: len(prefix_ids)] == prefix_ids:
                rest = cids[len(prefix_ids) :]
                rest_ids_for_np.append(rest)
                rest_map_idx.append(idx)
            else:
                fallback_full_np.append(cids)
                fallback_full_idx.append(idx)

        # compute rest log-probs conditioned on context+prefix
        new_ctx_ids = context_ids + prefix_ids
        rest_results = []
        if rest_ids_for_np:
            rest_results = compute_candidates_log_sums(new_ctx_ids, rest_ids_for_np, tokenizer, model, device, batch_size=cand_batch_size)

        # fill NP candidate stats
        np_geom_vals = [0.0] * len(np_cands)

        # handle prefix-starting ones
        for k, (logsum_rest, count_rest) in enumerate(rest_results):
            orig_idx = rest_map_idx[k]
            total_log = prefix_log + logsum_rest
            total_tok_count = prefix_toks + count_rest
            if total_tok_count <= 0:
                geom = 0.0
            else:
                geom = math.exp(total_log / float(total_tok_count))
            np_geom_vals[orig_idx] = geom

        # handle fallback full NP computations
        if fallback_full_np:
            fallback_results = compute_candidates_log_sums(context_ids, fallback_full_np, tokenizer, model, device, batch_size=cand_batch_size)
            for k, (logsum_full, count_full) in enumerate(fallback_results):
                orig_idx = fallback_full_idx[k]
                geom = math.exp(logsum_full / float(count_full)) if count_full > 0 else 0.0
                np_geom_vals[orig_idx] = geom

        # NP category mass = mean of per-candidate geometric-means
        np_mass = float(sum(np_geom_vals) / len(np_geom_vals)) if np_geom_vals else 0.0

        # --- VP category: compute directly in batches ---
        vp_results = compute_candidates_log_sums(context_ids, vp_ids, tokenizer, model, device, batch_size=cand_batch_size)
        vp_geom = []
        for logsum, cnt in vp_results:
            if cnt > 0:
                vp_geom.append(math.exp(logsum / float(cnt)))
            else:
                vp_geom.append(0.0)
        vp_mass = float(sum(vp_geom) / len(vp_geom)) if vp_geom else 0.0

        # --- question mark candidate ---
        q_result = compute_candidates_log_sums(context_ids, q_ids, tokenizer, model, device, batch_size=1)[0]
        q_geom = math.exp(q_result[0] / float(q_result[1])) if q_result[1] > 0 else 0.0

        out_rows.append(
            {
                "file": filename,
                "idx": i,
                "sentence": sent,
                "context": context,
                "np_mass": np_mass,
                "vp_mass": vp_mass,
                "q_mass": q_geom,
                "np_count": len(np_cands),
                "vp_count": len(vp_cands),
            }
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="HF model id or path")
    p.add_argument("--orc_test", required=True, help="Path to ORC test set (one sent per line)")
    p.add_argument("--wh_test", required=True, help="Path to WH test set (one sent per line)")
    p.add_argument("--out", default="prob_masses_results.csv", help="CSV output file")
    p.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    p.add_argument("--sent_limit", type=int, default=None, help="Limit number of sentences per file (for debugging)")
    p.add_argument("--cand_batch_size", type=int, default=64, help="Batch size for candidate evaluation")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = _load_model_and_tokenizer(args.checkpoint, device)

    rows: List[dict] = []
    evaluate_sentences(args.orc_test, "orc", tokenizer, model, device, rows, sent_limit=args.sent_limit, cand_batch_size=args.cand_batch_size)
    evaluate_sentences(args.wh_test, "wh", tokenizer, model, device, rows, sent_limit=args.sent_limit, cand_batch_size=args.cand_batch_size)

    # write CSV
    fieldnames = ["file", "idx", "sentence", "context", "np_mass", "vp_mass", "q_mass", "np_count", "vp_count"]
    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # print a short aggregate summary
    orc_rows = [r for r in rows if r["file"] == args.orc_test]
    wh_rows = [r for r in rows if r["file"] == args.wh_test]

    def mean(l, key):
        return sum(x[key] for x in l) / len(l) if l else 0.0

    print("Saved results to:", args.out)
    print("ORC avg masses: NP=%.6f VP=%.6f Q=%.6f" % (mean(orc_rows, "np_mass"), mean(orc_rows, "vp_mass"), mean(orc_rows, "q_mass")))
    print("WH  avg masses: NP=%.6f VP=%.6f Q=%.6f" % (mean(wh_rows, "np_mass"), mean(wh_rows, "vp_mass"), mean(wh_rows, "q_mass")))


if __name__ == "__main__":
    main()
