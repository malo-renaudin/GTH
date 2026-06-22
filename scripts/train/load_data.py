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
argument_parser.add_argument("--block_size", type=int, default=512)
argument_parser.add_argument("--cache_dir", type=str, default=".cache")
args = argument_parser.parse_args()

dataset = load_dataset("text", data_files={
    "train": args.train_file,
    "validation": args.valid_file,
}, cache_dir=args.cache_dir)

tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"])

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

block_size = args.block_size

def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
lm_datasets = tokenized.map(
                group_texts,
                batched=True,
                num_proc=4,
                desc=f"Grouping texts in chunks of {block_size}",
            )

lm_datasets.save_to_disk(args.cache_dir + f"/{args.dataset_name}")