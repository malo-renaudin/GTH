import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

import torch

from litgpt import Tokenizer
from litgpt.config import Config
from litgpt.model import GPT

nouns_orc = ["boy", "student", "doctor", "artist", "athlete","girl",
             "child", "pilot", "scientist", "engineer"] 
nouns_wh =  ["student", "doctor", "pilot", "officer", "athlete",
           "artist","child", "girl", "boy", "patient", "client", 
           "tourist"]
verbs_orc = ["visits", "visit", "helps", "help", "avoids", 
             "avoid", "follows", "follow", "greets", "greet"]
verbs_wh =[
    "visit", "visited", "visiting",
    "help", "helped", "helping",
    "greet", "greeted", "greeting",
    "follow", "followed", "following",
    "avoid", "avoided", "avoiding",
    "call", "called", "calling",
    "observe", "observed", "observing"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load checkpoints
def load_checkpoint(ckpt_file, device, max_seq_length):
    model = GPT(Config.from_checkpoint(ckpt_file.parent))
    raw = torch.load(ckpt_file, map_location=device)
    state = raw.get("model") if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else raw
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_file}")
    model.load_state_dict(state, strict=True)
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    model.to(device).eval()
    return model

#tokenize sentences with the tokenizer of the model
def tokenize_sentences(sentences: List[str], tokenizer: Tokenizer) -> List[List[int]]:
    return [tokenizer.encode(sentence) for sentence in sentences]

#create a dictionary mapping words to lists of token ids (handles multi-token words)
# words in sentences are preceded by a space, so we encode with a leading space
def create_word_to_token_mapping(list_of_words, tokenizer: Tokenizer) -> Dict[str, List[int]]:
    word_to_token = {}
    for w in list_of_words:
        tokens = tokenizer.encode(" " + w, bos=False, eos=False).tolist()
        word_to_token[w] = tokens
    return word_to_token

#cut a sequence of token if a given token is in the list
def find_given_token_in_a_seq(tok_seq, dictionary_tokens):
    for idx, token in enumerate(tok_seq):
        if token in dictionary_tokens:
            #here i want the index to cut the sequence after thta token
            return idx
    return None

def word_mass(logits: torch.Tensor, word_token_lists: List[List[int]]) -> float:
    """Sum of average probabilities across all words (each possibly multi-token).
    For each word: average probabilities of its tokens, then sum across words."""
    if not word_token_lists:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    total = 0.0
    for tokens in word_token_lists:
        ids = torch.tensor(tokens, device=logits.device)
        total += probs[ids].mean().item()
    return total


#run the model until the first target-word token is met, then score noun/verb mass
def compute_structure_probabilities(
    model: GPT,
    tokenizer: Tokenizer,
    sentences: List[str],
    target_token_lists: List[List[int]],
    other_token_lists_1: List[List[int]],
    other_token_lists_2: List[List[int]],
) -> Tuple[float, float, int]:
    target_flat = {t for tl in target_token_lists for t in tl}
    all_1, all_2, count = 0.0, 0.0, 0
    for s in tqdm(sentences, desc="  sentences", leave=False):
        tok = tokenizer.encode(s).tolist()
        idx = find_given_token_in_a_seq(tok, target_flat)
        if idx is None:
            continue
        input_ids = torch.tensor(tok[:idx + 1], device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)[0, -1, :]
        all_1 += word_mass(logits, other_token_lists_1)
        all_2 += word_mass(logits, other_token_lists_2)
        count += 1
    if count == 0:
        return 0.0, 0.0, count
    return all_1 / count, all_2 / count, count
        
        
def worker_eval_checkpoint(args_tuple):
    ckpt_file, tokenizer_dir, target_token_lists, set_1, set_2, sentences, max_seq_length = args_tuple
    model = load_checkpoint(ckpt_file, device, max_seq_length)
    tokenizer = Tokenizer(tokenizer_dir)
    mass_1, mass_2, roi_count = compute_structure_probabilities(
        model, tokenizer, sentences, target_token_lists, set_1, set_2
    )
    step = int(ckpt_file.parent.name.split("-")[1])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = {
        "step": step,
        "noun_mass": round(mass_1, 6),
        "verb_mass": round(mass_2, 6),
        "roi_count": roi_count,
        "checkpoint": str(ckpt_file),
    }
    print(f"  step {step}: noun_mass={mass_1:.6f}, verb_mass={mass_2:.6f}, roi_count={roi_count}")
    return result


def run_eval_over_checkpoints(checkpoint_dir, tokenizer_dir, target_token_lists, set_1, set_2, sentences, num_processes, max_seq_length, result_name):
    from concurrent.futures import ProcessPoolExecutor
    
    ckpt_files = sorted(
        checkpoint_dir.glob("step-*/lit_model.pth"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not ckpt_files:
        raise FileNotFoundError(f"No step checkpoints found under {checkpoint_dir}")

    work_items = [
        (ckpt, tokenizer_dir, target_token_lists, set_1, set_2, sentences, max_seq_length)
        for ckpt in ckpt_files
    ]
    
    rows = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for result in tqdm(executor.map(worker_eval_checkpoint, work_items), total=len(work_items), desc="checkpoints"):
            rows.append(result)
    
    rows.sort(key=lambda r: r["step"])
    Path(result_name).parent.mkdir(parents=True, exist_ok=True)
    with open(result_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "noun_mass", "verb_mass", "roi_count", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact noun/verb mass evaluation across checkpoints.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("checkpoints/gpt2"))
    parser.add_argument("--sentences-file", type=Path, required=True)
    parser.add_argument("--structure", choices=["orc", "wh"], required=True)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=0)
    parser.add_argument("--result-name", type=Path, default=Path("results/eval_test_scan.csv"))
    args = parser.parse_args()

    tokenizer = Tokenizer(args.tokenizer_dir)
    if args.structure == "orc":
        target_map = create_word_to_token_mapping(verbs_orc, tokenizer)
        noun_map   = create_word_to_token_mapping(nouns_orc, tokenizer)
        verb_map   = create_word_to_token_mapping(verbs_orc, tokenizer)
    else:
        target_map = create_word_to_token_mapping(verbs_wh, tokenizer)
        noun_map   = create_word_to_token_mapping(nouns_wh, tokenizer)
        verb_map   = create_word_to_token_mapping(verbs_wh, tokenizer)

    sentences = read_nonempty_lines(args.sentences_file)

    # --- DEBUG ---
    print(f"\n[DEBUG] Structure: {args.structure}")
    print(f"[DEBUG] Target verb token ids:")
    for w, toks in target_map.items():
        print(f"  '{w}' -> {toks}")
    print(f"[DEBUG] Noun token ids:")
    for w, toks in noun_map.items():
        print(f"  '{w}' -> {toks}")
    print(f"[DEBUG] Sentences loaded: {len(sentences)}")
    print(f"[DEBUG] First 3 sentences:")
    for s in sentences[:3]:
        toks = tokenizer.encode(s).tolist()
        target_flat = {t for tl in target_map.values() for t in tl}
        idx = find_given_token_in_a_seq(toks, target_flat)
        print(f"  sentence : {s}")
        print(f"  token ids: {toks}")
        print(f"  ROI idx  : {idx} ({'MATCHED' if idx is not None else 'NO MATCH'})")
    print()
    # --- END DEBUG ---

    run_eval_over_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_dir=args.tokenizer_dir,
        target_token_lists=list(target_map.values()),
        set_1=list(noun_map.values()),
        set_2=list(verb_map.values()),
        sentences=sentences,
        num_processes=args.num_processes,
        max_seq_length=args.max_seq_length,
        result_name=args.result_name,
    )


if __name__ == "__main__":
    main()
        