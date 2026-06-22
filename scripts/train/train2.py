from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from itertools import chain
import argparse
import yaml

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dataset-name", type=str, default="english_data")
argument_parser.add_argument("--model-name", type=str, default="gpt2")
argument_parser.add_argument("--config", type=str, default="gpt2")
argument_parser.add_argument("--cache-dir", type=str, default="./cache")
args = argument_parser.parse_args()

config = yaml.safe_load(open(args.config))

hf_config = AutoConfig.from_pretrained(args.model_name, cache_dir= args.cache_dir, local_files_only=True)
model = AutoModelForCausalLM.from_config(hf_config)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir= args.cache_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

lm_datasets = load_from_disk(args.cache_dir + f"/{args.dataset_name}")

training_args = TrainingArguments(
    output_dir=config.get("output_dir", "gpt2-out"),
    per_device_train_batch_size=config.get("train_batch_size", 4),
    do_eval = True,
    do_train = True,
    eval_strategy="steps",   
    # evaluate_during_training=True,
    save_strategy="steps",
    eval_steps=config.get("eval_steps", 500),
    save_steps=config.get("save_steps", 500),
    learning_rate=config.get("learning_rate", 5e-5),#grid
    num_train_epochs=config.get("num_train_epochs", 3),
    weight_decay=config.get("weight_decay", 0.01),#grid
    warmup_steps=config.get("warmup_steps", 100),#grid
    logging_steps=config.get("logging_steps", 50),
    max_grad_norm=config.get("max_grad_norm", 1),
    bf16=True, 
    load_best_model_at_end=True,   # IMPORTANT
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()