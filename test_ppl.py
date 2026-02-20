import os
import glob
import sys
import pandas as pd
import torch
import argparse
from src.language_models.utils import load_model, batchify, get_batch
from src.language_models.dictionary_corpus import Dictionary, tokenize
import logging 

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", type=str, required=True)
parser.add_argument("--test_files", nargs="+", required=True)
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dictionary = Dictionary("base_data")  # Adjust if needed

emsize = 650
nhid = 650
nlayers = 2
dropout = 0.5
nheads = 10
tied = False
classmodel = "RNNModel"
model_type = "LSTM"

checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pt")))


for test_file in args.test_files:
    test_ids = tokenize(dictionary, test_file)
    test_data = batchify(test_ids, 10, device)
    logging.info(f"Checkpoints found: {checkpoints}")
    logging.info(f"Test file: {test_file}")
    logging.info(f"Test data shape: {test_data.shape}")
    ntokens = len(dictionary)
    results = []
    for ckpt in checkpoints:
        model, _ = load_model(
            classmodel, model_type, ntokens, emsize, nhid, nheads, dropout, device, nlayers, tied, ckpt
        )
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            hidden = model.init_hidden(10)
            for i in range(0, test_data.size(0) - 1, 35):
                data, targets = get_batch(test_data, i, 35)
                data, targets = data.to(device), targets.to(device)
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                loss = criterion(output_flat, targets)
                total_loss += loss.item()
                total_tokens += targets.numel()
                hidden = tuple(h.detach() for h in hidden)
        avg_loss = total_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        results.append({"checkpoint": os.path.basename(ckpt), "perplexity": ppl})
        logging.info(f"Results: {results}")
    test_base = os.path.splitext(os.path.basename(test_file))[0]
    csv_name = f"results_{test_base}_{args.model_name}.csv"
    pd.DataFrame(results).to_csv(csv_name, index=False)