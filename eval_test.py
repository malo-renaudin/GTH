import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

#create a dictionary mapping words to tokens
def create_word_to_token_mapping(list_of_words, tokenizer: Tokenizer) -> Dict[str, int]:
    word_to_token = {}
    for w in list_of_words:
        token = tokenizer.encode(w)
        if len(token) == 1:
            word_to_token[w] = token[0]
        else:
            print(f"Warning: '{w}' is tokenized into multiple tokens {token}, skipping.")
    return word_to_token

#cut a sequence of token if a given token is in the list
def find_given_token_in_a_seq(tok_seq, dictionary_tokens):
    for idx, token in enumerate(tok_seq):
        if token in dictionary_tokens:
            #here i want the index to cut the sequence after thta token
            return idx
    return None

#run the model until a token from a given dictionary is met, extract logits for two other dictionaries, and compute probability mass for eahc dictionnary
def compute_structure_probabilities(model: GPT, sentences, target_tokens: Set[int], other_tokens_1: Set[int], other_tokens_2: Set[int]) -> Tuple[float, float]:
    for s in sentences:
        tok = tokenize_sentences([s], model.tokenizer)[0]
        input_ids = torch.tensor(tok, device=device).unsqueeze(0)
        idx = find_given_token_in_a_seq(input_ids, target_tokens)
        input_ids = input_ids[:, :idx+1] 
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            other_1_prob = torch.sum(logits[other_tokens_1].softmax(dim=0)).item()
            other_2_prob = torch.sum(logits[other_tokens_2].softmax(dim=0)).item()
            return other_1_prob, other_2_prob
        
        
def run_eval_in_parallel(checkpoint_dir, target_tokens, set_1, set_2, sentences, num_processes, max_seq_length, result_name):
    from concurrent.futures import ProcessPoolExecutor

    def worker(ckpt_file):
        model = load_checkpoint(ckpt_file, device, max_seq_length)
        return compute_structure_probabilities(model, sentences, target_tokens, set_1, set_2)

    ckpt_files = list(Path(checkpoint_dir).glob("*.pt"))
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(worker, ckpt_files))
        #return results as a csv : put probability masses and checkpoint name
        
        with open(result_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Other 1 Probability", "Other 2 Probability"])
            for result in results:
                writer.writerow(result)
    return results


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
        target_tokens = set(create_word_to_token_mapping(verbs_orc, tokenizer).values())
        set_1 = set(create_word_to_token_mapping(nouns_orc, tokenizer).values())
        set_2 = set(create_word_to_token_mapping(verbs_orc, tokenizer).values())
    else:
        target_tokens = set(create_word_to_token_mapping(verbs_wh, tokenizer).values())
        set_1 = set(create_word_to_token_mapping(nouns_wh, tokenizer).values())
        set_2 = set(create_word_to_token_mapping(verbs_wh, tokenizer).values())

    sentences = read_nonempty_lines(args.sentences_file)
    run_eval_in_parallel(
        checkpoint_dir=args.checkpoint_dir,
        target_tokens=target_tokens,
        set_1=set_1,
        set_2=set_2,
        sentences=sentences,
        num_processes=args.num_processes,
        max_seq_length=args.max_seq_length,
        result_name=args.result_name,
    )


if __name__ == "__main__":
    main()
        