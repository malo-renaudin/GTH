import random
from datasets import load_dataset, interleave_datasets, IterableDataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedTokenizer
)
import nltk
from nltk.tokenize import sent_tokenize
import argparse
import yaml
import re
import torch


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--model-name", type=str, default="gpt2")
argument_parser.add_argument("--config", type=str, default="configs/hf/gpt2.yaml")
argument_parser.add_argument("--cache-dir", type=str, default="scripts/train/.cache")
argument_parser.add_argument("--output-dir", type=str)
argument_parser.add_argument("--c4", type=float, default=0.9)
argument_parser.add_argument("--orc", type=float, default=0)
argument_parser.add_argument("--wh", type=float, default=0)
argument_parser.add_argument("--svo_wh", type=float, default=0.05)
argument_parser.add_argument("--svo_orc", type=float, default=0.05)
args = argument_parser.parse_args()

c4_train_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/train"
c4_val_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/validation"

c4_ds = load_from_disk(c4_train_path).to_iterable_dataset()
c4_val_ds = load_from_disk(c4_val_path).to_iterable_dataset()

_sentence_splitter = re.compile(r"(?<=[.!?])\s+")

def sentence_generator(dataset):
    for example in dataset:
        text = example["text"]

        sentences = _sentence_splitter.split(text)

        for sent in sentences:
            sent = sent.strip()
            if sent:
                yield {"text": sent}

c4_sent = IterableDataset.from_generator(
    lambda: sentence_generator(c4_ds)
)

c4_val = IterableDataset.from_generator(
    lambda: sentence_generator(c4_val_ds)
)

orc_ds = load_dataset("text", data_files="data/orc7.txt", split="train", streaming=True)
wh_ds = load_dataset("text", data_files="data/wh5.txt", split="train", streaming=True)
svo_wh_ds = load_dataset("text", data_files="data/declaratives_from_wh5.txt", split="train", streaming=True)
svo_orc_ds = load_dataset("text", data_files="data/declaratives_from_orc7.txt", split="train", streaming=True)

mixed_stream = interleave_datasets(
    [
        c4_sent,
        orc_ds,
        wh_ds,
        svo_wh_ds,
        svo_orc_ds
    ],
    probabilities=[
        args.c4,  # C4 backbone
        args.orc, # ORC
        args.wh, # WH
        args.svo_wh, # SVO-from-WH
        args.svo_orc  # SVO-from-ORC
    ],
    seed=42
)

tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token



class PackedStreamingDataset(IterableDataset):
    def __init__(self, stream, tokenizer: PreTrainedTokenizer, block_size=1024, batch_text_size=128):
        self.stream = stream
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_text_size = batch_text_size
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.stream, "set_epoch"):
            self.stream.set_epoch(epoch)

    def __iter__(self):
        buffer = torch.empty((0,), dtype=torch.long)
        text_buffer = []

        eos = self.tokenizer.eos_token_id

        for example in self.stream:
            text_buffer.append(example["text"].strip())

            # ---- batch tokenize ----
            if len(text_buffer) >= self.batch_text_size:

                enc = self.tokenizer(
                    [t + self.tokenizer.eos_token for t in text_buffer],
                    add_special_tokens=False
                )["input_ids"]

                # flatten WITHOUT Python loop over tokens
                flat = torch.tensor(
                    sum(enc, []),  # still Python flatten once (minimal unavoidable step)
                    dtype=torch.long
                )

                buffer = torch.cat([buffer, flat])

                text_buffer = []

            # ---- tensor-based packing ----
            while buffer.size(0) >= self.block_size:

                chunk = buffer[:self.block_size]
                buffer = buffer[self.block_size:]

                yield {
                    "input_ids": chunk,
                    "labels": chunk.clone(),
                    "attention_mask": torch.ones_like(chunk)
                }
train_dataset = PackedStreamingDataset(
    mixed_stream,
    tokenizer,
    block_size=1024
)
c4_val_truncated = c4_val.take(10000)

validation_dataset = PackedStreamingDataset(
    c4_val_truncated,
    tokenizer,
    block_size=1024
)

config = yaml.safe_load(open(args.config))

hf_config = AutoConfig.from_pretrained(args.model_name, 
                                       cache_dir= args.cache_dir, 
                                       local_files_only=True,
                                       attn_implementation="sdpa")
model = AutoModelForCausalLM.from_config(hf_config)

# model.gradient_checkpointing_enable()
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=config.get("train_batch_size", 4),
    do_eval = True,
    do_train = True,
    eval_strategy="steps",   
    # evaluate_during_training=True,
    save_strategy="steps",
    lr_scheduler_type="cosine",
    eval_steps=config.get("eval_steps", 500),
    save_steps=config.get("save_steps", 500),
    learning_rate=config.get("learning_rate", 5e-5),#grid
    # num_train_epochs=config.get("num_train_epochs", 3),
    max_steps=1000000,
    weight_decay=config.get("weight_decay", 0.01),#grid
    logging_steps=config.get("logging_steps", 50),
    max_grad_norm=config.get("max_grad_norm", 1),
    # dataloader_num_workers=4,
    # dataloader_prefetch_factor=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    # gradient_accumulation_steps = 1,
    bf16=True, 
    optim="adamw_torch_fused",
    load_best_model_at_end=True,   # IMPORTANT
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_ratio=config.get("warmup_ratio", 0.1),
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator = data_collator,
)

trainer.train()