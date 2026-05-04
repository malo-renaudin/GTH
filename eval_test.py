import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
from tqdm import tqdm

from litgpt import Tokenizer
from litgpt.config import Config
from litgpt.model import GPT

nouns_orc = ["boy", "student", "doctor", "artist", "athlete", "girl", "child", "pilot", "scientist", "engineer"]
nouns_wh = ["student", "doctor", "pilot", "officer", "athlete", "artist", "child", "girl", "boy", "patient", "client", "tourist"]
verbs_orc = ["visits", "visit", "helps", "help", "avoids", "avoid", "follows", "follow", "greets", "greet"]
verbs_wh = [
    "visit", "visited", "visiting", "help", "helped", "helping", "greet", "greeted", "greeting",
    "follow", "followed", "following", "avoid", "avoided", "avoiding", "call", "called", "calling",
    "observe", "observed", "observing",
]

ORC_REL_MARKERS = {"that", "who"}
IRREGULAR_PLURALS = {"child": "children", "man": "men", "woman": "women"}
CSV_FIELDS = [
    "step",
    "structure",
    "target_label",
    "comparator_label",
    "target_mass",
    "target_head_mass",
    "comparator_mass",
    "target_minus_comparator",
    "roi_count",
    "checkpoint",
]


def load_checkpoint(ckpt_file: Path, device: torch.device, max_seq_length: int) -> GPT:
    """
    Load a GPT model from a checkpoint file, set max sequence length if needed, and move to device.
    """
    model = GPT(Config.from_checkpoint(ckpt_file.parent))
    raw = torch.load(ckpt_file, map_location=device)
    state = raw.get("model") if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else raw
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_file}")
    model.load_state_dict(state, strict=True)
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    return model.to(device).eval()


def resolve_checkpoint_file(checkpoint_input: Path) -> Path:
    """
    Given a path to a checkpoint or checkpoint directory, return the path to the lit_model.pth file.
    """
    if checkpoint_input.is_file():
        if checkpoint_input.name != "lit_model.pth":
            raise ValueError(f"Expected 'lit_model.pth', got {checkpoint_input.name}")
        return checkpoint_input
    if checkpoint_input.is_dir():
        ckpt = checkpoint_input / "lit_model.pth"
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"No lit_model.pth found in {checkpoint_input}")
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_input}")


def step_from_checkpoint(ckpt_file: Path) -> int:
    """
    Extract the training step number from the checkpoint file or its parent directory name.
    """
    name = ckpt_file.parent.name if ckpt_file.name == "lit_model.pth" else ckpt_file.name
    return int(name.split("-")[1])


def read_nonempty_lines(path: Path) -> List[str]:
    """
    Read all non-empty, stripped lines from a text file.
    """
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_word(word: str) -> str:
    """
    Lowercase and strip punctuation from a word for matching.
    """
    return word.strip(".,?!;:\"'()[]{}").lower()


def lexical_mass(probs: torch.Tensor, token_lists: List[List[int]]) -> float:
    """
    For a list of token id lists (words/phrases), sum the mean probability for each.
    """
    total = 0.0
    for ids in token_lists:
        if ids:
            total += probs[torch.tensor(ids, device=probs.device)].mean().item()
    return total


def extract_orc_moved_np(sentence: str, noun_forms_orc: Set[str]) -> Tuple[List[str], Optional[str]]:
    """
    Extract the moved NP (surface string) from an ORC sentence.
    Returns the list of words in the NP and the head noun.
    """
    words = sentence.split()
    if not words or normalize_word(words[0]) != "the":
        return [], None

    head_idx = None
    for i in range(1, len(words)):
        w = normalize_word(words[i])
        if w in ORC_REL_MARKERS:
            break
        if w in noun_forms_orc:
            head_idx = i
            break

    if head_idx is None:
        return [], None

    np_words = [normalize_word(w) for w in words[: head_idx + 1] if normalize_word(w)]
    return np_words, (np_words[-1] if np_words else None)


def compute_one_checkpoint(
    ckpt_file: Path,
    tokenizer_dir: Path,
    structure: str,
    sentences: List[str],
    max_seq_length: int,
) -> dict:
    """
    Evaluate a single checkpoint on a set of sentences for either ORC or WH structure.
    For each sentence, find the ROI (first verb), run the model up to that point, and compute:
      - For ORC: probability mass for the moved NP (full and head) vs verb mass
      - For WH: probability mass for all WH NPs vs question mark
    Returns a dict with averaged results and metadata for CSV output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer(tokenizer_dir)
    model = load_checkpoint(ckpt_file, device, max_seq_length)

    # Prepare token lists for ROI detection and scoring
    roi_verbs = verbs_orc if structure == "orc" else verbs_wh
    roi_target_lists = [tokenizer.encode(" " + w, bos=False, eos=False).tolist() for w in roi_verbs]
    roi_target_flat = {t for lst in roi_target_lists for t in lst}

    orc_verb_lists = [tokenizer.encode(" " + w, bos=False, eos=False).tolist() for w in verbs_orc]
    wh_np_vocab_lists = [
        tokenizer.encode(" " + w, bos=False, eos=False).tolist()
        for w in sorted(nouns_wh)
    ]
    # Get all token ids for question mark (with and without leading space)
    qmark: Set[int] = set()
    for p in ["?", " ?"]:
        qmark.update(tokenizer.encode(p, bos=False, eos=False).tolist())
    orc_noun_forms = nouns_orc

    t_sum, th_sum, c_sum, n = 0.0, 0.0, 0.0, 0
    t_label, c_label = (
        ("moved_np_mass", "verb_mass") if structure == "orc" else ("wh_np_vocab_mass", "question_mark_mass")
    )

    for s in tqdm(sentences, desc="  sentences", leave=False):
        # Tokenize sentence and find ROI (first verb token)
        tok = tokenizer.encode(s).tolist()
        roi_idx = None
        for i, t in enumerate(tok):
            if t in roi_target_flat:
                roi_idx = i
                break
        if roi_idx is None:
            continue

        # Run model up to ROI and get next-token probabilities
        x = torch.tensor(tok[: roi_idx + 1], device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        if structure == "orc":
            # Extract moved NP (surface) and head noun for this sentence
            np_words, head = extract_orc_moved_np(s, orc_noun_forms)
            if not np_words or head is None:
                continue
            np_lists = [tokenizer.encode(" " + w, bos=False, eos=False).tolist() for w in np_words]
            head_list = [tokenizer.encode(" " + head, bos=False, eos=False).tolist()]
            # Average probability for full NP (length-normalized) and for head noun only
            target = lexical_mass(probs, np_lists) / max(1, len(np_lists))
            target_head = lexical_mass(probs, head_list)
            comp = lexical_mass(probs, orc_verb_lists)
            t_label, c_label = "moved_np_mass", "verb_mass"
        else:
            # WH: sum probability for all WH NPs, compare to question mark
            target = lexical_mass(probs, wh_np_vocab_lists)
            target_head = target
            comp = lexical_mass(probs, [sorted(qmark)])  # treat all ?-token variants as one group
            t_label, c_label = "wh_np_vocab_mass", "question_mark_mass"

        t_sum += target
        th_sum += target_head
        c_sum += comp
        n += 1

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step = step_from_checkpoint(ckpt_file)
    if n == 0:
        t_avg = th_avg = c_avg = 0.0
    else:
        t_avg, th_avg, c_avg = t_sum / n, th_sum / n, c_sum / n

    print(f"  step {step}: target_mass={t_avg:.6f}, comparator_mass={c_avg:.6f}, roi_count={n}")

    return {
        "step": step,
        "structure": structure,
        "target_label": t_label,
        "comparator_label": c_label,
        "target_mass": round(t_avg, 6),
        "target_head_mass": round(th_avg, 6),
        "comparator_mass": round(c_avg, 6),
        "target_minus_comparator": round(t_avg - c_avg, 6),
        "roi_count": n,
        "checkpoint": str(ckpt_file),
    }


def worker_unpack(args: Tuple[Path, Path, str, List[str], int]) -> dict:
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
    p.add_argument("--result-name", type=Path, default=Path("results/eval_test_scan.csv"))
    args = p.parse_args()

    sentences = read_nonempty_lines(args.sentences_file)

    if args.checkpoint is not None:
        ckpt = resolve_checkpoint_file(args.checkpoint)
        rows = [compute_one_checkpoint(ckpt, args.tokenizer_dir, args.structure, sentences, args.max_seq_length)]
        write_rows(rows, args.result_name)
        print(f"Results saved to: {args.result_name}")
        return

    ckpts = sorted(args.checkpoint_dir.glob("step-*/lit_model.pth"), key=lambda x: int(x.parent.name.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No step checkpoints found under {args.checkpoint_dir}")

    items = [(ckpt, args.tokenizer_dir, args.structure, sentences, args.max_seq_length) for ckpt in ckpts]
    rows = []
    with ProcessPoolExecutor(max_workers=args.num_processes) as ex:
        for r in tqdm(ex.map(worker_unpack, items), total=len(items), desc="checkpoints"):
            rows.append(r)

    rows.sort(key=lambda r: r["step"])
    write_rows(rows, args.result_name)
    print(f"Results saved to: {args.result_name}")


if __name__ == "__main__":
    main()
