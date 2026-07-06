import random
from datasets import load_from_disk, load_dataset, interleave_datasets
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
import argparse
import yaml
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
c4_val_path   = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/validation"

# C4: sharded so HF assigns one shard per DataLoader worker
c4_train_ds = load_from_disk(c4_train_path).to_iterable_dataset(num_shards=_n_workers)
c4_val_ds   = load_from_disk(c4_val_path).to_iterable_dataset(num_shards=_n_workers)

# Text-file sources (one sentence per line; small files, HF streams them)
orc_ds     = load_dataset("text", data_files="data/orc7.txt",                      split="train", streaming=True)
wh_ds      = load_dataset("text", data_files="data/wh5.txt",                       split="train", streaming=True)
svo_wh_ds  = load_dataset("text", data_files="data/declaratives_from_wh5.txt",     split="train", streaming=True)
svo_orc_ds = load_dataset("text", data_files="data/declaratives_from_orc7.txt",    split="train", streaming=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = int(1e30)  # suppress spurious long-sequence warnings

# Build the interleaved training stream once.
# n_shards = _n_workers (C4) + n_active_text_sources ≥ _n_workers
# → HF assigns shards to workers without killing any worker.
_candidates = [
    (c4_train_ds, args.c4),
    (orc_ds,      args.orc),
    (wh_ds,       args.wh),
    (svo_wh_ds,   args.svo_wh),
    (svo_orc_ds,  args.svo_orc),
]
_active_srcs, _active_probs = zip(*[(ds, p) for ds, p in _candidates if p > 0])
train_stream = interleave_datasets(list(_active_srcs), probabilities=list(_active_probs), seed=42)


class PackedStreamingDataset(TorchIterableDataset):
    """Packs a streaming HF IterableDataset into fixed block_size-token chunks.
    HF assigns dataset shards to DataLoader workers automatically via
    stream.__iter__(); no manual worker logic is needed here.
    """
    def __init__(self, stream, tokenizer: PreTrainedTokenizer,
                 block_size: int = 1024, batch_text_size: int = 512):
        super().__init__()
        self.stream = stream
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_text_size = batch_text_size

    def __iter__(self):
        eos    = self.tokenizer.eos_token
        buffer = []
        batch  = []

        for example in self.stream:   # HF distributes shards to workers here
            batch.append(example["text"].strip())

            if len(batch) >= self.batch_text_size:
                ids_list = self.tokenizer(
                    [t + eos for t in batch], add_special_tokens=False
                )["input_ids"]
                for ids in ids_list:
                    buffer.extend(ids)
                batch = []

            while len(buffer) >= self.block_size:
                chunk = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                buffer = buffer[self.block_size:]
                yield {"input_ids": chunk, "labels": chunk.clone(),
                       "attention_mask": torch.ones_like(chunk)}


train_dataset = PackedStreamingDataset(train_stream, tokenizer)
validation_dataset = PackedStreamingDataset(c4_val_ds.take(10000), tokenizer)
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
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
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