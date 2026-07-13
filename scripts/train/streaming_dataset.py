import random
from datasets import load_from_disk, load_dataset
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
from log_scale_callback import LogScaleCallback


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--model-name", type=str, default="gpt2")
argument_parser.add_argument("--config", type=str, default="configs/hf/gpt2.yaml")
argument_parser.add_argument("--cache-dir", type=str, default="scripts/train/.cache")
argument_parser.add_argument("--output-dir", type=str)
argument_parser.add_argument("--c4", type=float, default=0.9)
argument_parser.add_argument("--orc", type=float, default=0)
argument_parser.add_argument("--wh", type=float, default=0)
argument_parser.add_argument("--svo", type=float, default=0.1)
argument_parser.add_argument("--log-scale-n-points",   type=int, default=20)
argument_parser.add_argument("--log-scale-start-step", type=int, default=10)
argument_parser.add_argument("--blimp-dir",            type=str, default="eval_data/blimp_data")
argument_parser.add_argument("--nested-inner",         type=str, default="eval_data/short_nested_inner_english.json")
argument_parser.add_argument("--nested-outer",         type=str, default="eval_data/short_nested_outer_english.json")
argument_parser.add_argument("--locality",             type=str, default="eval_data/locality.json")
argument_parser.add_argument("--filler-gap-orc",       type=str, default="eval_data/filler_gap_orc.csv")
argument_parser.add_argument("--filler-gap-wh",        type=str, default="eval_data/filler_gap_wh.csv")
argument_parser.add_argument("--transitivity-orc",     type=str, default="eval_data/orc_transitivity.csv")
argument_parser.add_argument("--semantic-distractor",  type=str, default="eval_data/orc_semantic_distractor.csv")
argument_parser.add_argument("--probability-masses-orc", type=str, default="eval_data/orc_test.txt")
argument_parser.add_argument("--probability-masses-wh",  type=str, default="eval_data/wh_test.txt")
argument_parser.add_argument("--eval-max-samples",     type=int, default=500)
args = argument_parser.parse_args()

config = yaml.safe_load(open(args.config))
_n_workers = config.get("dataloader_num_workers", 4)

c4_train_path = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/train"
c4_val_path   = "/lustre/fsmisc/dataset/HuggingFace/c4/realnewslike/validation"

tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = int(1e30)  # suppress spurious long-sequence warnings


# ---------------------------------------------------------------------------
# Source loaders — callables that return a fresh iterable of {"text": str}.
# Using map-style Dataset (load_from_disk / load_dataset without streaming)
# avoids HF's multi-worker shard check, so every DataLoader worker can safely
# and independently iterate its own copy of the data.
# ---------------------------------------------------------------------------

def _c4_train_loader():
    """Infinite generator over C4 training documents."""
    ds = load_from_disk(c4_train_path)          # Arrow, memory-mapped — safe for concurrent reads
    while True:
        for ex in ds:
            yield {"text": ex["text"]}

def _c4_val_loader():
    """Single-pass generator over C4 validation documents."""
    ds = load_from_disk(c4_val_path)
    for ex in ds:
        yield {"text": ex["text"]}

def _make_text_loader(filepath):
    """Returns an infinite-cycling loader for a line-per-sentence text file."""
    def loader():
        ds = load_dataset("text", data_files=filepath, split="train")  # map-style, fits in RAM
        while True:
            for ex in ds:
                yield {"text": ex["text"]}
    return loader


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class PackedStreamingDataset(TorchIterableDataset):
    """
    Megatron-style data-parallel streaming dataset.

    All workers/ranks generate the same deterministic global sample sequence
    from the same RNG seed.  Each (rank × worker) slot keeps only the samples
    whose global index satisfies:

        global_index % (world_size * num_workers) == rank * num_workers + worker_id

    This gives non-overlapping, exhaustive coverage of the stream without any
    physical data duplication.  Source reads for filtered-out samples are cheap
    (no tokenization); only kept samples are tokenized and packed.
    """

    def __init__(
        self,
        source_loaders,             # list[callable[() -> Iterable[{"text": str}]]]
        probabilities,              # sampling weights (need not sum to 1)
        tokenizer: PreTrainedTokenizer,
        block_size: int = 1024,
        batch_text_size: int = 512,
        seed: int = 42,
        max_samples: int = 0,   # 0 = unlimited (use for validation to cap eval time)
    ):
        super().__init__()
        self.source_loaders  = source_loaders
        self.probabilities   = probabilities
        self.tokenizer       = tokenizer
        self.block_size      = block_size
        self.batch_text_size = batch_text_size
        self.seed            = seed
        self.max_samples     = max_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        rng = random.Random(self.seed + worker_id)
        n_yielded = 0

        active = [(ld, p) for ld, p in zip(self.source_loaders, self.probabilities) if p > 0]
        loaders, probs = zip(*active)
        total = sum(probs)
        weights = [p / total for p in probs]

        iters = [iter(ld()) for ld in loaders]

        eos = self.tokenizer.eos_token
        # One independent buffer and batch per source.
        buffers = [[] for _ in loaders]
        batches = [[] for _ in loaders]

        def fill_buffer(i):
            """Consume source i until its buffer holds at least block_size tokens."""
            while len(buffers[i]) < self.block_size:
                try:
                    example = next(iters[i])
                except StopIteration:
                    iters[i] = iter(loaders[i]())
                    example = next(iters[i])

                batches[i].append(example["text"].strip())

                if len(batches[i]) >= self.batch_text_size:
                    ids_list = self.tokenizer(
                        [t + eos for t in batches[i]],
                        add_special_tokens=False
                    )["input_ids"]
                    for ids in ids_list:
                        buffers[i].extend(ids)
                    batches[i] = []

        while True:
            if self.max_samples and n_yielded >= self.max_samples:
                return

            # Pick a source by weight — this directly controls the token ratio
            # because every chosen source yields exactly one block_size-token block.
            idx = rng.choices(range(len(loaders)), weights=weights)[0]

            fill_buffer(idx)

            chunk = torch.tensor(buffers[idx][:self.block_size], dtype=torch.long)
            buffers[idx] = buffers[idx][self.block_size:]
            n_yielded += 1

            yield {
                "input_ids": chunk,
                "labels": chunk.clone(),
                "attention_mask": torch.ones_like(chunk),
            }

# ---------------------------------------------------------------------------
# Dataset instantiation
# ---------------------------------------------------------------------------

_train_candidates = [
    (_c4_train_loader,                                  args.c4),
    (_make_text_loader("data/orc_good_vocab.txt"),                args.orc),
    (_make_text_loader("data/wh_good_vocab.txt"),                 args.wh),
    (_make_text_loader("data/merged_svo.txt"), args.svo),
    # (_make_text_loader("data/declaratives_from_orc7.txt"), args.svo_orc),
]

train_dataset = PackedStreamingDataset(
    source_loaders=[ld for ld, p in _train_candidates if p > 0],
    probabilities=[p  for ld, p in _train_candidates if p > 0],
    tokenizer=tokenizer,
)

validation_dataset = PackedStreamingDataset(
    source_loaders=[_c4_val_loader],
    probabilities=[1.0],
    tokenizer=tokenizer,
    max_samples=args.eval_max_samples,
)
hf_config = AutoConfig.from_pretrained(args.model_name, 
                                       cache_dir= args.cache_dir, 
                                       local_files_only=True,
                                       attn_implementation="sdpa")
model = AutoModelForCausalLM.from_config(hf_config)
model.config.use_cache = False

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=config.get("train_batch_size", 4),
    do_eval = True,
    do_train = True,
    eval_strategy="no",
    save_strategy="no",
    lr_scheduler_type="cosine",
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
    torch_compile=True,
    optim="adamw_torch_fused",



    warmup_ratio=config.get("warmup_ratio", 0.1),
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

callbacks = []
callbacks.append(LogScaleCallback(
    max_steps            = training_args.max_steps,
    n_points             = args.log_scale_n_points,
    start_step           = args.log_scale_start_step,
    output_dir           = args.output_dir,
    tokenizer            = tokenizer,
    blimp_dir            = args.blimp_dir,
    nested_inner         = args.nested_inner,
    nested_outer         = args.nested_outer,
    locality             = args.locality,
    filler_gap_orc       = args.filler_gap_orc,
    filler_gap_wh        = args.filler_gap_wh,
    transitivity_orc     = args.transitivity_orc,
    semantic_distractor  = args.semantic_distractor,
    probability_masses_orc = args.probability_masses_orc,
    probability_masses_wh  = args.probability_masses_wh,
))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    callbacks=callbacks,
)

trainer.train()