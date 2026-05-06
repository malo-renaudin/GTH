import argparse
import csv
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Set, Tuple

import torch
from tqdm import tqdm

from litgpt import Tokenizer

from utils import (
    ORC_REL_MARKERS,
    extract_orc_moved_np,
    lexical_mass,
    load_checkpoint,
    normalize_word,
    np_chain_logprob,
    read_nonempty_lines,
    resolve_checkpoint_file,
    step_from_checkpoint,
    _word_token_spans,
)

nouns_orc = ["boy", "boys", "student", "students", "doctor", "doctors", "artist", "artists",
             "athlete", "athletes", "girl", "girls", "child", "children", "pilot", "pilots",
             "scientist", "scientists", "engineer", "engineers"]
n1_opts_wh = ["student", "doctor", "pilot", "officer", "athlete", "artist"]
n2_opts_wh = ["child", "girl", "boy", "patient", "client", "tourist"]
adj1_opts  = ["young", "tall", "smart", "brave", "kind", "famous"]
adj2_opts  = ["creative", "serious", "friendly", "quiet", "active"]

np_wh: List[Tuple[str, str]] = (
    [(n, a) for n in n1_opts_wh for a in adj1_opts] +
    [(n, a) for n in n2_opts_wh for a in adj2_opts]
)
verbs_orc_continuation = ["is", "are", "likes", "like", "enjoys", "enjoy"]
verbs_orc = ["visits", "visit", "helps", "help", "avoids", "avoid", "follows", "follow", "greets", "greet"]
verbs_wh = [
    "visit", "visited", "visiting", "help", "helped", "helping", "greet", "greeted", "greeting",
    "follow", "followed", "following", "avoid", "avoided", "avoiding", "call", "called", "calling",
    "observe", "observed", "observing",
]

CSV_FIELDS = [
    "step", "structure", "target_label", "comparator_label",
    "target_mass", "target_mass_median",
    "target_head_mass", "target_head_mass_median",
    "the_mass",
    "comparator_mass", "comparator_mass_median",
    "target_minus_comparator",
    "roi_count", "checkpoint",
]


def compute_one_checkpoint(
    ckpt_file: Path,
    tokenizer_dir: Path,
    structure: str,
    sentences: List[str],
    max_seq_length: int,
    batch_size: int = 32,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(tokenizer_dir)
    model = load_checkpoint(ckpt_file, device, max_seq_length)

    roi_verb_set = set(verbs_orc if structure == "orc" else verbs_wh)
    orc_verb_continuation_lists = [tokenizer.encode(" " + w, bos=False, eos=False).tolist() for w in verbs_orc_continuation]
    # 4 surface forms per (noun, adj): "the adj noun", "the noun", "noun", "adj noun"
    wh_np_lists = [
        tokenizer.encode(form, bos=False, eos=False).tolist()
        for noun, adj in np_wh
        for form in (f" the {adj} {noun}", f" the {noun}", f" {noun}", f" {adj} {noun}")
    ]
    wh_noun_lists = [tokenizer.encode(f" {n}", bos=False, eos=False).tolist() for n in n1_opts_wh + n2_opts_wh]
    qmark: Set[int] = set()
    for p_str in ["?", " ?"]:
        qmark.update(tokenizer.encode(p_str, bos=False, eos=False).tolist())
    qmark_tensor = torch.tensor(sorted(qmark), device=device)
    the_lists = [tokenizer.encode(" the", bos=False, eos=False).tolist()]

    t_label = "moved_np_joint_prob" if structure == "orc" else "wh_np_mean_prob"
    c_label = "verb_prob_mass"       if structure == "orc" else "question_mark_prob"

    # Pre-tokenize all sentences and find ROI indices
    valid_items: List[Tuple[List[int], int, str]] = []
    for s in sentences:
        tok, spans = _word_token_spans(s, tokenizer)
        roi_idx = None
        for span_i, (word, start, end) in enumerate(spans):
            if normalize_word(word) in roi_verb_set:
                if structure == "wh" and word.endswith("?"):
                    prefix = "" if span_i == 0 else " "
                    verb_ids = tokenizer.encode(prefix + word.rstrip("?"), bos=False, eos=False).tolist()
                    roi_idx = start + len(verb_ids) - 1
                else:
                    roi_idx = end - 1
                break
        if roi_idx is not None:
            valid_items.append((tok, roi_idx, s))

    t_vals:   List[float] = []
    th_vals:  List[float] = []
    the_vals: List[float] = []
    c_vals:   List[float] = []

    for batch_start in tqdm(range(0, len(valid_items), batch_size), desc="  batches", leave=False):
        batch = valid_items[batch_start: batch_start + batch_size]
        contexts = [tok[: roi_idx + 1] for tok, roi_idx, _ in batch]
        ctx_lengths = [len(c) for c in contexts]
        max_ctx = max(ctx_lengths)
        x_batch = torch.tensor(
            [c + [0] * (max_ctx - len(c)) for c in contexts], device=device
        )
        with torch.no_grad():
            logits_batch = model(x_batch)

        for i, (tok, roi_idx, s) in enumerate(batch):
            lsp = torch.log_softmax(logits_batch[i, ctx_lengths[i] - 1, :], dim=-1)
            x   = torch.tensor(contexts[i], device=device).unsqueeze(0)

            def chain_prob(ids: List[int]) -> float:
                return math.exp(np_chain_logprob(model, x, ids, device, first_step_lsp=lsp))

            if structure == "orc":
                np_words, head = extract_orc_moved_np(s, nouns_orc)
                if not np_words or head is None:
                    continue
                flat_np_ids = [t for w in np_words for t in tokenizer.encode(" " + w, bos=False, eos=False).tolist()]
                head_ids    = tokenizer.encode(" " + head, bos=False, eos=False).tolist()
                # Joint probability of the full NP (no per-token normalization)
                target      = chain_prob(flat_np_ids)
                target_head = chain_prob(head_ids)
                comp = math.exp(torch.stack([
                    lsp[torch.tensor(ids, device=device)].logsumexp(0)
                    for ids in orc_verb_continuation_lists if ids
                ]).logsumexp(0).item())
            else:
                # Mean P(NP form) across all 4 forms × (noun, adj) pairs
                np_probs   = [chain_prob(ids) for ids in wh_np_lists   if ids]
                noun_probs = [chain_prob(ids) for ids in wh_noun_lists if ids]
                target      = sum(np_probs)   / len(np_probs)
                target_head = sum(noun_probs) / len(noun_probs)
                comp = math.exp(lsp[qmark_tensor].logsumexp(0).item())

            t_vals.append(target)
            th_vals.append(target_head)
            the_vals.append(lexical_mass(lsp.exp(), the_lists))
            c_vals.append(comp)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step = step_from_checkpoint(ckpt_file)
    n = len(t_vals)
    if n == 0:
        t_avg = t_med = th_avg = th_med = the_avg = c_avg = c_med = 0.0
    else:
        t_avg,  t_med  = statistics.mean(t_vals),   statistics.median(t_vals)
        th_avg, th_med = statistics.mean(th_vals),  statistics.median(th_vals)
        the_avg        = statistics.mean(the_vals)
        c_avg,  c_med  = statistics.mean(c_vals),   statistics.median(c_vals)

    print(f"  step {step}: target={t_avg:.6f} (med={t_med:.6f}), comparator={c_avg:.6f}, roi_count={n}")

    return {
        "step": step,
        "structure": structure,
        "target_label": t_label,
        "comparator_label": c_label,
        "target_mass": round(t_avg, 6),
        "target_mass_median": round(t_med, 6),
        "target_head_mass": round(th_avg, 6),
        "target_head_mass_median": round(th_med, 6),
        "the_mass": round(the_avg, 6),
        "comparator_mass": round(c_avg, 6),
        "comparator_mass_median": round(c_med, 6),
        "target_minus_comparator": round(t_avg - c_avg, 6),
        "roi_count": n,
        "checkpoint": str(ckpt_file),
    }


def worker_unpack(args: Tuple[Path, Path, str, List[str], int, int]) -> dict:
    return compute_one_checkpoint(*args)


def write_rows(rows: List[dict], result_name: Path) -> None:
    result_name.parent.mkdir(parents=True, exist_ok=True)
    with open(result_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="ORC moved-NP vs verb and WH NP-vocab vs '?' evaluation.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path)
    g.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    p.add_argument("--sentences-file", type=Path, required=True)
    p.add_argument("--structure", choices=["orc", "wh"], required=True)
    p.add_argument("--num-processes", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--result-name", type=Path, default=Path("results/eval_test_scan.csv"))
    args = p.parse_args()

    sentences = read_nonempty_lines(args.sentences_file)

    if args.checkpoint is not None:
        ckpt = resolve_checkpoint_file(args.checkpoint)
        rows = [compute_one_checkpoint(ckpt, args.tokenizer_dir, args.structure, sentences, args.max_seq_length, args.batch_size)]
        write_rows(rows, args.result_name)
        print(f"Results saved to: {args.result_name}")
        return

    ckpts = sorted(args.checkpoint_dir.glob("step-*/lit_model.pth"), key=lambda x: int(x.parent.name.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No step checkpoints found under {args.checkpoint_dir}")

    items = [(ckpt, args.tokenizer_dir, args.structure, sentences, args.max_seq_length, args.batch_size) for ckpt in ckpts]
    rows = []
    with ProcessPoolExecutor(max_workers=args.num_processes) as ex:
        for r in tqdm(ex.map(worker_unpack, items), total=len(items), desc="checkpoints"):
            rows.append(r)

    rows.sort(key=lambda r: r["step"])
    write_rows(rows, args.result_name)
    print(f"Results saved to: {args.result_name}")


if __name__ == "__main__":
    main()
