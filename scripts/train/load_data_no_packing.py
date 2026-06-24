from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from itertools import chain
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--train_file", type=str, default="/scratch2/mrenaudin/GTH/data/english_data/train.txt")
argument_parser.add_argument("--valid_file", type=str, default="/scratch2/mrenaudin/GTH/data/english_data/valid.txt")
argument_parser.add_argument("--test_file", type=str, default="/scratch2/mrenaudin/GTH/data/english_data/test.txt")
argument_parser.add_argument("--dataset-name", type=str, default="english_data")
argument_parser.add_argument("--cache_dir", type=str, default=".cache")
args = argument_parser.parse_args()

dataset = load_dataset("text", data_files={
    "train": args.train_file,
    "validation": args.valid_file,
    "test": args.test_file
}, cache_dir=args.cache_dir)

tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[MISSING]"]
})

MISSING_ID = tokenizer.convert_tokens_to_ids("[MISSING]")

def tokenize(batch):
    texts = []
    for t in batch["text"]:
        t = t.strip()
        t = t.replace("<unk>", "[MISSING]")
        # Normalize any existing EOS-like markers
        t = t.replace("<|endoftext|>", tokenizer.eos_token)

        # Ensure exactly one EOS at end
        if not t.endswith(tokenizer.eos_token):
            t = t + tokenizer.eos_token

        texts.append(t)

    return tokenizer(texts, add_special_tokens=False)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
def process(example):
    example["labels"] = [
        t if t != MISSING_ID else -100
        for t in example["input_ids"]
    ]
    return example

tokenized = tokenized.map(process)


# =========================
# SAVE PACKED DATASET
# =========================
tokenized.save_to_disk(args.cache_dir + f"/{args.dataset_name}")
tokenizer.save_pretrained(args.cache_dir + "/tokenizer")