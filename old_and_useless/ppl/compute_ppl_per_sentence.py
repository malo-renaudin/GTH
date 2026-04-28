#!/usr/bin/env python3
"""
Compute perplexity distribution per sentence for a given dataset and model.
"""
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.language_models.model import RNNModel
import src.language_models.model as model_module
from src.language_models.dictionary_corpus import Dictionary

# Make the model module available for unpickling
sys.modules['model'] = model_module


def calculate_batch_perplexity(model, batch_sentences, batch_lengths, device, ntokens):
    """Calculate perplexity for a batch of sentences efficiently"""
    if len(batch_sentences) == 0:
        return []
    
    seq_len = max(batch_lengths) if batch_lengths else 0
    batch_size = len(batch_sentences)
    
    if seq_len < 2:
        return [None] * batch_size
    
    # Create padded tensor
    padded = torch.zeros(seq_len, batch_size, dtype=torch.long, device=device)
    for i, sent in enumerate(batch_sentences):
        padded[:len(sent), i] = sent
    
    # Initialize hidden state
    hidden = model.init_hidden(batch_size)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = torch.zeros(batch_size, device=device)
    counts = torch.zeros(batch_size, device=device)
    
    with torch.no_grad():
        for i in range(seq_len - 1):
            input_token = padded[i:i+1]
            target = padded[i+1]
            
            # Forward pass
            output, hidden = model(input_token, hidden)
            output = output.view(-1, ntokens)
            target = target.view(-1)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Mask for valid positions
            mask = torch.arange(seq_len - 1, device=device)[i] < (torch.tensor(batch_lengths, device=device) - 1).unsqueeze(0)
            mask = mask.float().view(-1)
            
            losses = losses + (loss * mask)
            counts = counts + mask
    
    # Calculate perplexities
    perplexities = []
    for i in range(batch_size):
        if counts[i] > 0:
            avg_loss = losses[i] / counts[i]
            ppl = torch.exp(avg_loss).item()
            perplexities.append(ppl)
        else:
            perplexities.append(None)
    
    return perplexities


def load_model(checkpoint_path, dictionary, device, model_type='LSTM', 
               emsize=650, nhid=650, nlayers=2, dropout=0.2, tied=False):
    """Load a trained language model"""
    ntokens = len(dictionary)
    
    # Workaround for PyTorch version compatibility
    _original_lstm_setstate = nn.LSTM.__setstate__
    
    def _patched_lstm_setstate(self, d):
        if not hasattr(self, '_flat_weights'):
            self._flat_weights = []
        if not hasattr(self, '_flat_weights_names'):
            self._flat_weights_names = []
        try:
            _original_lstm_setstate(self, d)
        except AttributeError:
            self.__dict__.update(d)
            self._flat_weights = []
            self._flat_weights_names = []
            self.flatten_parameters()
    
    nn.LSTM.__setstate__ = _patched_lstm_setstate
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = RNNModel(model_type, ntokens, emsize, nhid, nlayers, dropout, tied)
        
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint.state_dict(), strict=False)
        
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded from {checkpoint_path}")
        return model, ntokens
        
    finally:
        nn.LSTM.__setstate__ = _original_lstm_setstate


def load_gpt2_model(model_name='gpt2', device='cuda'):
    """Load a pretrained GPT2 model from Hugging Face"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library is not installed. Install with: pip install transformers")
    
    print(f"Loading GPT2 model: {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model = model.to(device)
    model.eval()
    print(f"✓ GPT2 model loaded: {model_name}")
    return model, tokenizer


def calculate_batch_perplexity_gpt2(model, tokenizer, batch_sentences, device):
    """Calculate perplexity for a batch of sentences using GPT2"""
    if len(batch_sentences) == 0:
        return []
    
    perplexities = []
    
    with torch.no_grad():
        for sentence in batch_sentences:
            # Join sentence words into a string
            text = ' '.join(sentence)
            
            # Tokenize
            encodings = tokenizer(text, return_tensors='pt')
            input_ids = encodings.input_ids.to(device)
            
            if input_ids.size(1) < 2:
                perplexities.append(None)
                continue
            
            # Get model outputs
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            
            # Calculate perplexity
            ppl = torch.exp(loss).item()
            perplexities.append(ppl)
    
    return perplexities


def main():
    parser = argparse.ArgumentParser(description='Compute perplexity per sentence')
    parser.add_argument('--model', help='Path to model checkpoint (not needed if --use-gpt2)')
    parser.add_argument('--data', required=True, help='Path to data directory')
    parser.add_argument('--split', default='valid.txt', help='Data split to use (default: valid.txt)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--output', default=None, help='Output file for results (default: auto-generated)')
    parser.add_argument('--save-individual', action='store_true', help='Save individual sentence perplexities')
    parser.add_argument('--use-gpt2', action='store_true', help='Use pretrained GPT2 from Hugging Face')
    parser.add_argument('--gpt2-model', default='gpt2', help='GPT2 model name (default: gpt2, options: gpt2, gpt2-medium, gpt2-large, gpt2-xl)')
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_gpt2 and not args.model:
        parser.error('--model is required unless --use-gpt2 is specified')
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model based on type
    if args.use_gpt2:
        model, tokenizer = load_gpt2_model(args.gpt2_model, device)
        use_gpt2 = True
        ntokens = None  # Not needed for GPT2
    else:
        # Load dictionary
        print(f"Loading dictionary from {args.data}...")
        dictionary = Dictionary(args.data)
        unk_idx = dictionary.word2idx.get('<unk>', 0)
        eos_idx = dictionary.word2idx.get('<eos>', None)
        
        # Load model
        model, ntokens = load_model(args.model, dictionary, device)
        use_gpt2 = False
        tokenizer = None
    
    # Read and split sentences
    data_file = Path(args.data) / args.split
    print(f"Reading {data_file}...")
    with open(data_file, 'r', encoding='utf8') as f:
        content = f.read()
    
    words = content.split()
    print(f"Total words: {len(words)}")
    
    # Split into sentences
    sentences = []
    current_sentence = []
    for word in words:
        current_sentence.append(word)
        if word == '<eos>':
            sentences.append(current_sentence)
            current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)
    
    print(f"Number of sentences: {len(sentences)}")
    
    # Tokenize sentences (only for RNN models, GPT2 tokenizes on the fly)
    if not use_gpt2:
        print("Tokenizing sentences...")
        valid_tokenized = []
        for idx, sentence in enumerate(sentences):
            if idx % 50000 == 0 and idx > 0:
                print(f"  {idx}/{len(sentences)}")
            
            token_ids = [dictionary.word2idx.get(word, unk_idx) for word in sentence]
            
            if len(token_ids) > 1:
                valid_tokenized.append(torch.LongTensor(token_ids).to(device))
        
        print(f"✓ Tokenized {len(valid_tokenized)} valid sentences")
    else:
        # For GPT2, we'll pass raw sentences
        valid_tokenized = [sent for sent in sentences if len(sent) > 1]
        print(f"✓ Prepared {len(valid_tokenized)} valid sentences for GPT2")
    
    # Calculate perplexities
    perplexities = []
    sentence_lengths = []
    
    print(f"Calculating perplexities (batch_size={args.batch_size})...")
    for start_idx in range(0, len(valid_tokenized), args.batch_size):
        if start_idx % (args.batch_size * 10) == 0 and start_idx > 0:
            print(f"  {start_idx}/{len(valid_tokenized)}")
        
        end_idx = min(start_idx + args.batch_size, len(valid_tokenized))
        batch = valid_tokenized[start_idx:end_idx]
        
        if use_gpt2:
            batch_ppls = calculate_batch_perplexity_gpt2(model, tokenizer, batch, device)
            batch_lengths = [len(sent) for sent in batch]
        else:
            batch_lengths = [len(sent) for sent in batch]
            batch_ppls = calculate_batch_perplexity(model, batch, batch_lengths, device, ntokens)
        
        for i, ppl in enumerate(batch_ppls):
            if ppl is not None and np.isfinite(ppl):
                perplexities.append(ppl)
                sentence_lengths.append(batch_lengths[i])
    
    # Calculate statistics
    perplexities = np.array(perplexities)
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(perplexities)} sentences")
    print(f"{'='*60}")
    print(f"Mean perplexity:     {np.mean(perplexities):.2f}")
    print(f"Median perplexity:   {np.median(perplexities):.2f}")
    print(f"Std perplexity:      {np.std(perplexities):.2f}")
    print(f"Min perplexity:      {np.min(perplexities):.2f}")
    print(f"Max perplexity:      {np.max(perplexities):.2f}")
    print(f"25th percentile:     {np.percentile(perplexities, 25):.2f}")
    print(f"75th percentile:     {np.percentile(perplexities, 75):.2f}")
    print(f"95th percentile:     {np.percentile(perplexities, 95):.2f}")
    print(f"{'='*60}")
    
    # Save results
    if args.output is None:
        if use_gpt2:
            model_name = args.gpt2_model.replace('/', '_')
        else:
            model_name = Path(args.model).stem
        args.output = f"ppl_distribution_{model_name}_{args.split.replace('.txt', '')}.npz"
    
    print(f"\nSaving results to {args.output}...")
    if args.save_individual:
        np.savez(args.output, 
                 perplexities=perplexities,
                 sentence_lengths=np.array(sentence_lengths),
                 mean=np.mean(perplexities),
                 median=np.median(perplexities),
                 std=np.std(perplexities))
    else:
        # Save only statistics to save space
        np.savez(args.output,
                 mean=np.mean(perplexities),
                 median=np.median(perplexities),
                 std=np.std(perplexities),
                 min=np.min(perplexities),
                 max=np.max(perplexities),
                 percentiles=np.percentile(perplexities, [25, 50, 75, 95]),
                 n_sentences=len(perplexities))
    
    print("✓ Done!")


if __name__ == '__main__':
    main()
