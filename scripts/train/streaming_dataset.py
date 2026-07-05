import random
from datasets import load_dataset, interleave_datasets, IterableDataset, load_from_disk, Features, Value
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer
)
import nltk
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

config = yaml.safe_load(open(args.config))

c4_train_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/train"
c4_val_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/validation"

_sentence_splitter = re.compile(r"(?<=[.!?])\s+")

def sentence_generator(dataset, worker_id=0, num_workers=1):
    for i, example in enumerate(dataset):
        # Let each worker grab its own slice cleanly at the source
        if i % num_workers != worker_id:
            continue
        text = example["text"]
        sentences = _sentence_splitter.split(text)
        for sent in sentences:
            sent = sent.strip()
            if sent:
                yield {"text": sent}

# All sources yield only {"text": str} — declare this explicitly so
# interleave_datasets never needs to call _head() to infer features,
# which would fail inside DataLoader worker processes.
text_features = Features({"text": Value("string")})

c4_sent = IterableDataset.from_generator(
    lambda: sentence_generator(load_from_disk(c4_train_path).to_iterable_dataset()),
    features=text_features,
)
c4_val = IterableDataset.from_generator(
    lambda: sentence_generator(load_from_disk(c4_val_path).to_iterable_dataset()),
    features=text_features,
)

# Wrap text-file streaming sources so each call creates a fresh streaming dataset
orc_ds = IterableDataset.from_generator(
    lambda: iter(load_dataset("text", data_files="data/orc7.txt", split="train", streaming=True)),
    features=text_features,
)
wh_ds = IterableDataset.from_generator(
    lambda: iter(load_dataset("text", data_files="data/wh5.txt", split="train", streaming=True)),
    features=text_features,
)
svo_wh_ds = IterableDataset.from_generator(
    lambda: iter(load_dataset("text", data_files="data/declaratives_from_wh5.txt", split="train", streaming=True)),
    features=text_features,
)
svo_orc_ds = IterableDataset.from_generator(
    lambda: iter(load_dataset("text", data_files="data/declaratives_from_orc7.txt", split="train", streaming=True)),
    features=text_features,
)


tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token



class PackedStreamingDataset(TorchIterableDataset):
    def __init__(self, stream=None, tokenizer: PreTrainedTokenizer=None, block_size=1024, batch_text_size=2048, sources=None, probabilities=None, num_workers=4):
        super().__init__()
        self.stream = stream
        self.sources = sources
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_text_size = batch_text_size
        self._epoch = 0
        self._num_workers = num_workers

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.stream, "set_epoch"):
            self.stream.set_epoch(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if self.sources is None:
            # Validation path: shard the single stream across workers.
            stream = self.stream
            if worker_info is not None and worker_info.num_workers > 1:
                try:
                    stream = stream.shard(num_shards=worker_info.num_workers, index=worker_info.id)
                except Exception:
                    wid, nw, base = worker_info.id, worker_info.num_workers, stream
                    stream = IterableDataset.from_generator(
                        lambda: (item for i, item in enumerate(base) if i % nw == wid),
                        features=getattr(stream, "features", None)
                    )
        else:
            # Training path: interleave all active sources with given probabilities.
            # All sources are generator-based so no per-worker sharding is needed —
            # each worker independently draws from the same infinite streams.
            filtered = [(s, p) for s, p in zip(self.sources, self.probabilities or [1.0] * len(self.sources)) if p and p > 0]
            srcs = [s for s, _ in filtered]
            probs = [p for _, p in filtered]
            stream = interleave_datasets(srcs, probabilities=probs, seed=42 + self._epoch)

        buffer = torch.empty((0,), dtype=torch.long)
        text_buffer = []

        for example in stream:
            text_buffer.append(example["text"].strip())

            if len(text_buffer) >= self.batch_text_size:
                enc = self.tokenizer(
                    [t + self.tokenizer.eos_token for t in text_buffer],
                    add_special_tokens=False
                )["input_ids"]
                flat = torch.cat([torch.tensor(x, dtype=torch.long) for x in enc]) if enc else torch.empty((0,), dtype=torch.long)
                buffer = torch.cat([buffer, flat])
                text_buffer = []

            while buffer.size(0) >= self.block_size:
                chunk = buffer[:self.block_size]
                buffer = buffer[self.block_size:]
                yield {
                    "input_ids": chunk,
                    "attention_mask": torch.ones_like(chunk)
                }
train_dataset = PackedStreamingDataset(
    stream=None,
    tokenizer=tokenizer,
    block_size=1024,
    sources=[
        c4_sent,
        orc_ds,
        wh_ds,
        svo_wh_ds,
        svo_orc_ds
    ],
    probabilities=[
        args.c4,
        args.orc,
        args.wh,
        args.svo_wh,
        args.svo_orc
    ],
    num_workers=config.get("dataloader_num_workers", 4)
)
c4_val_truncated = c4_val.take(10000)
_num_workers = config.get("dataloader_num_workers", 4)

validation_dataset = PackedStreamingDataset(
    c4_val_truncated,
    tokenizer,
    block_size=1024,
    num_workers=_num_workers
)

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
    max_steps=30500,
    weight_decay=config.get("weight_decay", 0.1),
    adam_beta2=config.get("adam_beta2", 0.95),
    logging_steps=config.get("logging_steps", 50),
    max_grad_norm=config.get("max_grad_norm", 1),
    dataloader_num_workers=config.get("dataloader_num_workers", 4),
    dataloader_prefetch_factor=config.get("dataloader_prefetch_factor", 2),
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
    bf16=True, 
    optim="adamw_torch_fused",
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
    data_collator=data_collator,
)

trainer.train()