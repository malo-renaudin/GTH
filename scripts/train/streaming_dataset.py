import random
from datasets import load_dataset, interleave_datasets, IterableDataset, load_from_disk, concatenate_datasets
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
_n_workers = config.get("dataloader_num_workers", 4)

c4_train_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/train"
c4_val_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/validation"

c4_ds     = load_from_disk(c4_train_path).to_iterable_dataset(num_shards=_n_workers)
c4_val_ds = load_from_disk(c4_val_path).to_iterable_dataset(num_shards=_n_workers)

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

# A small helper function that checks PyTorch's current worker on-the-fly
def get_worker_sharded_generator(base_ds):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return sentence_generator(base_ds, worker_id=0, num_workers=1)
    return sentence_generator(base_ds, worker_id=worker_info.id, num_workers=worker_info.num_workers)

# Re-define your datasets so each worker (and HF probes) get a fresh iterator
# For C4 (Arrow), capture features once and recreate the iterable per call.
c4_features = getattr(c4_ds, "features", None)
c4_sent = IterableDataset.from_generator(
    lambda: sentence_generator(load_from_disk(c4_train_path).to_iterable_dataset()),
    features=c4_features,
)
c4_val = IterableDataset.from_generator(
    lambda: sentence_generator(load_from_disk(c4_val_path).to_iterable_dataset()),
    features=c4_features,
)
c4_sent.is_generator_sharded = True
c4_val.is_generator_sharded = True

# Text-file sources: _n_workers independent repeating copies concatenated so
# n_shards = _n_workers. HF will not kill DataLoader workers and each worker
# gets one full copy of the file — correct probabilities across all workers.
def _text_source(filepath):
    return concatenate_datasets([
        load_dataset("text", data_files=filepath, split="train", streaming=True)
        for _ in range(_n_workers)
    ])

orc_ds    = _text_source("data/orc7.txt")
wh_ds     = _text_source("data/wh5.txt")
svo_wh_ds  = _text_source("data/declaratives_from_wh5.txt")
svo_orc_ds = _text_source("data/declaratives_from_orc7.txt")


tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = int(1e30)  # suppress spurious long-sequence warnings



class PackedStreamingDataset(IterableDataset):
    def __init__(self, stream=None, tokenizer: PreTrainedTokenizer=None, block_size=1024, batch_text_size=512, sources=None, probabilities=None):
        # If `sources` is provided, we will shard each source per-worker
        # and then call `interleave_datasets` on the per-source shards.
        self.stream = stream
        self.sources = sources
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_text_size = batch_text_size
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.stream, "set_epoch"):
            self.stream.set_epoch(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Build the stream: either from per-source sharded interleaving
        # (preferred when `sources` is provided) or from the single
        # `self.stream` (fallback).
        stream = None

        if self.sources is None:
            # Fallback: use the single pre-built stream directly
            stream = self.stream
        else:
            # filter out zero-probability sources
            filtered = [(s, p) for s, p in zip(self.sources, self.probabilities or [1.0] * len(self.sources)) if p and p > 0]
            srcs = [s for s, _ in filtered]
            probs = [p for _, p in filtered]

            #shard dataset across workers and build an interleaved dataset per worker
            num_shards = worker_info.num_workers
            shard_idx = worker_info.id
            sharded = []
            for ds in srcs:
                if hasattr(ds, "_ex_iterable") and type(ds._ex_iterable).__name__ == "GeneratorIterable":
                    # The generator wrapper handles worker assignment inside itself
                    sharded.append(ds)
                    continue

                # Try HF sharding first; if it fails (some streaming sources
                # cannot be sharded after serialization), fall back to a
                # per-worker generator wrapper that yields the round-robin
                # slice for this worker.
                try:
                    sh = ds.shard(num_shards=num_shards, index=shard_idx)
                except Exception:
                    def _make_worker_slice(orig_ds, idx, n):
                        def _gen():
                            for i, item in enumerate(orig_ds):
                                if (i % n) == idx:
                                    yield item
                        return _gen

                    sh = IterableDataset.from_generator(
                        _make_worker_slice(ds, shard_idx, num_shards),
                        features=getattr(ds, "features", None)
                    )
                sharded.append(sh)
            # One token buffer and text buffer per source so each 1024-token
            # chunk is built from a single dataset. Chunks are interleaved at
            # the batch level via probabilistic source selection.
            buffers      = [[] for _ in sharded]
            text_buffers = [[] for _ in sharded]
            src_iters    = [iter(src) for src in sharded]
            total_p      = sum(probs)
            weights      = [p / total_p for p in probs]
            rng          = random.Random(42 + self._epoch + shard_idx)
            population   = list(range(len(sharded)))

            while True:
                idx = rng.choices(population, weights=weights)[0]
                try:
                    example = next(src_iters[idx])
                except StopIteration:
                    # restart small finite sources (orc/wh/svo loop infinitely)
                    src_iters[idx] = iter(sharded[idx])
                    continue

                text_buffers[idx].append(example["text"].strip())

                if len(text_buffers[idx]) >= self.batch_text_size:
                    enc = self.tokenizer(
                        [t + self.tokenizer.eos_token for t in text_buffers[idx]],
                        add_special_tokens=False
                    )["input_ids"]
                    for ids in enc:
                        buffers[idx].extend(ids)
                    text_buffers[idx] = []

                while len(buffers[idx]) >= self.block_size:
                    chunk = torch.tensor(buffers[idx][:self.block_size], dtype=torch.long)
                    buffers[idx] = buffers[idx][self.block_size:]
                    yield {
                        "input_ids": chunk,
                        "labels":    chunk.clone(),
                        "attention_mask": torch.ones_like(chunk)
                    }
            return

        # Validation packing: iterate self.stream, tokenize with EOS between
        # documents, pack into block_size chunks.
        buffer = torch.empty((0,), dtype=torch.long)
        text_buffer = []

        for example in stream:
            text_buffer.append(example["text"].strip())

            if len(text_buffer) >= self.batch_text_size:
                enc = self.tokenizer(
                    [t + self.tokenizer.eos_token for t in text_buffer],
                    add_special_tokens=False
                )["input_ids"]
                flat = torch.cat([torch.tensor(x, dtype=torch.long) for x in enc]) if len(enc) > 0 else torch.empty((0,), dtype=torch.long)
                buffer = torch.cat([buffer, flat])
                text_buffer = []

            while buffer.size(0) >= self.block_size:
                chunk = buffer[:self.block_size]
                buffer = buffer[self.block_size:]
                yield {
                    "input_ids": chunk,
                    "labels":    chunk.clone(),
                    "attention_mask": torch.ones_like(chunk)
                }
train_dataset = PackedStreamingDataset(
    stream=None,
    tokenizer=tokenizer,
    block_size=1024,
    sources=[
        c4_ds,   # whole documents — EOS at natural document boundaries
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
    ]
)
c4_val_truncated = c4_val_ds.take(10000)

validation_dataset = PackedStreamingDataset(
    stream=None,
    tokenizer=tokenizer,
    block_size=1024,
    sources=[c4_val_truncated],
    probabilities=[1.0]
)
hf_config = AutoConfig.from_pretrained(args.model_name, 
                                       cache_dir= args.cache_dir, 
                                       local_files_only=True,
                                       attn_implementation="sdpa")
model = AutoModelForCausalLM.from_config(hf_config)
model = torch.compile(model)
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