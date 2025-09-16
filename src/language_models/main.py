# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import math
import time
import pandas as pd
import torch
import torch.nn as nn
import os
import gc
import psutil
from dictionary_corpus import Corpus, TextDataset, Vocabulary, word_tokenizer, collate_batch, tokenize
import model
from lm_argparser import lm_parser
from utils import (
    repackage_hidden,
    get_batch,
    batchify,
    save_checkpoint,
    move_to_device,
    save_val_loss_data,
    load_model,
    get_memory_usage,
    log_memory_usage,
    clear_memory,
    TemperatureScheduler,
    pick_lt_st_indices,
    create_dataloaders
)
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

from torch.cuda.amp import autocast, GradScaler


parser = argparse.ArgumentParser(
    parents=[lm_parser], description="Basic training and evaluation for RNN LM"
)

args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(), logging.FileHandler(args.log)],
)
logging.info(args)

scaler = GradScaler() if args.cuda else None

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
# NEW : added device
device = torch.device("cuda" if args.cuda else "cpu")
print(f"Using device: {device}")

###############################################################################
# MLflow setup
###############################################################################


# mlflow.set_experiment(args.name)
# mlflow.start_run(run_name=args.name)

# mlflow.log_params({
#     "model_class": args.classmodel,
#     "model_type": args.model,
#     "emsize": args.emsize,
#     "nhid": getattr(args, "nhid", None),
#     "nlayers": args.nlayers,
#     "batch_size": args.batch_size,
#     "optimizer": args.optimizer,
#     "lr": 0.001 if args.optimizer == "Adam" else 10,
#     "bptt": args.bptt,
#     "dropout": args.dropout,
#     "cuda": args.cuda
# })

###############################################################################
# Load data
###############################################################################

# Load data
logging.info("Loading data")
start = time.time()

# # Old way to load data (from colorlessgreenRNNs)
corpus = Corpus(args.data, save_tokenized=False)
logging.info("( %.2f )" % (time.time() - start))
ntokens = len(corpus.dictionary)
logging.info("Vocab size %d", ntokens)

# Prepare data with batchify
logging.info("Preparing batches...")
eval_batch_size = 10

# # Use regular batchify for all data
train_data = batchify(corpus.train, args.batch_size, device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)
# train_data, val_data, test_data = create_dataloaders(corpus, args.bptt, args.batch_size, eval_batch_size)


criterion = nn.CrossEntropyLoss()

###############################################################################
# Build the model
###############################################################################

logging.info("Building the model")
print(args.model)
model, optimizer_state_dict = load_model(
    args.classmodel,
    args.model,
    ntokens,
    args.emsize,
    args.nhid,
    args.nheads,
    args.dropout,
    device,
    args.nlayers,
    args.tied,
    args.checkpoint_path,
    args.memory_size,
    args.memory_dim,
    args.bptt
)

logging.info(f"Built {args.classmodel}")

if hasattr(torch, 'compile'):
    model = torch.compile(model)
###############################################################################
# Optimizer
###############################################################################

if args.optimizer == "SGD":
    lr = 10
    optimizer = optim.SGD(model.parameters(), lr=lr)
elif args.optimizer == "Adam":
    lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr)
else:
    raise ValueError(f"Invalid optimizer: {args.optimizer}")
if optimizer_state_dict is not None:
    optimizer.load_state_dict(optimizer_state_dict)
    logging.info("Loaded optimizer state from checkpoint")



###############################################################################
# Evaluation
###############################################################################

# Original evaluation function


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    if args.classmodel == "RNN_Model":
        hidden = move_to_device(model.init_hidden(eval_batch_size), device)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            data, targets = data.to(device), targets.to(device)
        # for batch_idx, (data, targets) in enumerate(val_data):
        #     data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if args.classmodel == 'TransformerLM':
                output = model(data)  # No hidden state needed
                output_flat = output.view(-1, ntokens)
                total_loss += (
                    len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
                )
                del output, output_flat

            else:
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += (
                    len(data) * nn.CrossEntropyLoss()(output_flat, targets).item()
                )
                del output, output_flat
                hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source) - 1)


# NEW : create folder for checkpointing
main_folder = "val_loss"
subfolder = os.path.join(main_folder, args.name)
os.makedirs(subfolder, exist_ok=True)
val_loss_data = []

###############################################################################
# Training
###############################################################################


def train():
    # Turn on training mode which enables dropout
    model.train()
    total_loss = 0
    start_time = time.time()
    # For LSTM model : initialize hidden state at the beggining of each epoch as in colorlessgreenRNNs
    if args.classmodel == "RNNModel":
        hidden = move_to_device(model.init_hidden(args.batch_size), device)

    activation_stats = {}
    hooks = []
    def hook_fn(module, input, output):
        activation_stats[module.__class__.__name__] = {
            "mean": output.mean().item(),
            "std": output.std().item(),
            "max": output.max().item(),
            "min": output.min().item()
        }
        
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.LSTM)):
            hooks.append(layer.register_forward_hook(hook_fn))

    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)
        data, targets = data.to(device), targets.to(device)
    # for batch_idx, (data, targets) in enumerate(train_data):
    #     data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(enabled=args.cuda):
            # Forward pass on chunk

            if args.classmodel == 'RNNModel' and args.model == 'LSTM':
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss_reg = hidden[1].abs().mean()
                loss = criterion(output.view(-1, ntokens),
                                 targets)  # +reg*loss_reg
            # Around line where you have the other model conditions
            elif args.classmodel == 'TransformerLM':
                output = model(data)  # No hidden state needed
                loss = criterion(output.view(-1, ntokens), targets)
                del output

        if args.cuda and scaler is not None:
            # Mixed precision training
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

        total_loss += loss.item()

        # Logging and MLflow
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            grad_norm = max(p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None)
            batch_time = time.time() - start_time
            mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 3) if args.cuda else 0

            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | loss {:5.2f} | ppl {:8.2f} | '
                         'grad_norm {:5.2f} | ms/batch {:5.2f} | cpu_mem {:.2f} GB | gpu_mem {:.2f} GB'.format(
                epoch, batch, len(train_data)//args.bptt, cur_loss, math.exp(cur_loss),
                grad_norm, batch_time*1000/args.log_interval, mem_usage, gpu_mem
            ))

            # mlflow.log_metrics({
            #     "train_loss": cur_loss,
            #     "train_ppl": math.exp(cur_loss),
            #     "grad_norm": grad_norm,
            #     "batch_time_ms": batch_time*1000/args.log_interval,
            #     "cpu_mem_gb": mem_usage,
            #     "gpu_mem_gb": gpu_mem
            # }, step=epoch * len(train_data)//args.bptt + batch)

            # # Log activation stats
            # for key, stats in activation_stats.items():
            #     mlflow.log_metrics({f"activation_{key}_{k}": v for k, v in stats.items()},
            #                        step=epoch * len(train_data)//args.bptt + batch)

            total_loss = 0
            start_time = time.time()
            clear_memory()
    for h in hooks:
        h.remove()

###############################################################################
# Loop over epochs.
###############################################################################


try:
    if args.epoch_checkpointed:
        k = int(args.epoch_checkpointed)
    else:
        k = 1

    for epoch in range(k, args.epochs + 1):
        # Shuffle and tokenize the training data
        corpus.train = tokenize(corpus.dictionary, os.path.join(
            args.data, 'train.txt'), shuffle=True)
        train_data = batchify(corpus.train, args.batch_size, device)

        epoch_start_time = time.time()


        train()

        val_loss = evaluate(val_data)
        val_ppl = math.exp(val_loss)

        logging.info("-" * 89)
        logging.info("| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}".format(
            epoch, (time.time()-epoch_start_time), val_loss, val_ppl))
        logging.info("-" * 89)

        # # MLflow log
        # mlflow.log_metrics({
        #     "val_loss": val_loss,
        #     "val_ppl": val_ppl
        # }, step=epoch * len(train_data)//args.bptt)

        # Save checkpoint
        # save_checkpoint(model, optimizer, args.name, epoch, temperature, args.checkpoint_dir)
        val_loss_data.append({"epoch": epoch, "batch": "end_of_epoch", "val_loss": val_loss})
        filename = f"epoch{epoch}"
        save_val_loss_data(val_loss_data, subfolder, filename)

        model.train()
        clear_memory()
        
except KeyboardInterrupt:
    logging.info("-" * 89)
    logging.info("Exiting from training early")

# mlflow.pytorch.log_model(model, artifact_path="final_model")
# mlflow.end_run()