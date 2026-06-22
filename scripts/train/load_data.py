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
    return tokenizer(batch["text"])

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])



tokenized.save_to_disk(args.cache_dir + f"/{args.dataset_name}")