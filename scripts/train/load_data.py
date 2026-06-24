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

def tokenize(batch):
    texts = []
    for t in batch["text"]:
        t = t.strip()

        # Normalize any existing EOS-like markers
        t = t.replace("<|endoftext|>", tokenizer.eos_token)

        # Ensure exactly one EOS at end
        if not t.endswith(tokenizer.eos_token):
            t = t + tokenizer.eos_token

        texts.append(t)

    return tokenizer(texts, add_special_tokens=False)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])


block_size = 1024  # or 1024 on H100

def group_texts(examples):
    concatenated = {}
    for k in examples.keys():
        concatenated[k] = sum(examples[k], [])

    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size

    input_ids = [
        concatenated["input_ids"][i:i + block_size]
        for i in range(0, total_length, block_size)
    ]

    attention_mask = [[1] * block_size for _ in input_ids]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


packed = tokenized.map(
    group_texts,
    batched=True
)

# =========================
# SAVE PACKED DATASET
# =========================
packed.save_to_disk(args.cache_dir + f"/{args.dataset_name}_packed")