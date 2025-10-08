import os, glob, json
import torch, numpy as np, pandas as pd
from tqdm import tqdm
from src.language_models.model import RNNModel as lstm
from src.language_models.dictionary_corpus import Dictionary
from nltk import pos_tag, download as nltk_download
import argparse
# --- utilities ---
def resolve_device(dev_arg):
    if dev_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device('cuda' if dev_arg == 'cuda' and torch.cuda.is_available() else 'cpu')

def safe_basename(path): 
    return os.path.basename(path.rstrip('/'))

# --- model / checkpoint helpers ---
def build_model(dictionary, device, hidden_size=650):
    V = len(dictionary) if hasattr(dictionary, '__len__') else len(dictionary.idx2word)
    m = RNNModel("LSTM", V, hidden_size, hidden_size, 2, 0.2, False)
    return m.to(device)

def load_checkpoint(model, checkpoint_path, device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.to(device)
    return model

# --- tokenization / vocab helpers ---
def tokenize_sentences(dataset, dictionary, test):
    unk = dictionary.word2idx.get("<unk>", 0)
    key_map = {'orc': ('matrix', 'object_relative'), 'wh': ('declarative', 'wh_question')}
    if test not in key_map: raise ValueError(test)
    dec_k, mov_k = key_map[test]

    def col(seq_like, key):
        if isinstance(seq_like, pd.DataFrame):
            return seq_like[key].astype(str).tolist()
        return seq_like[key]

    def tok(s):
        words = s.strip().split()
        return {'words': words, 'indices': [dictionary.word2idx.get(w, unk) for w in words]}

    dec = [tok(s) for s in col(dataset, dec_k)]
    mov = [tok(s) for s in col(dataset, mov_k)]
    if test == 'orc': assert len(dec) == len(mov)
    return dec, mov

def find_noun_indices(dictionary):
    try:
        pos_tag(['test'])
    except LookupError:
        nltk_download('averaged_perceptron_tagger')
    vocab = dictionary.idx2word
    tagged = pos_tag(vocab)
    noun_tags = {'NN','NNS','NNP','NNPS'}
    return [i for i, (_, t) in enumerate(tagged) if t in noun_tags]

# --- device-safe hidden mover ---
def _move_hidden(h, device):
    if isinstance(h, (tuple, list)):
        return tuple(x.to(device) for x in h)
    return h.to(device)

# --- batched forward ---
def pad_and_run(model, batch, noun_indices, device):
    lens = [len(x['indices']) for x in batch]
    B = len(batch); L = max(lens)
    inp = torch.zeros((L, B), dtype=torch.long, device=device)
    for i, x in enumerate(batch):
        l = lens[i]
        if l: inp[:l, i] = torch.tensor(x['indices'], device=device)
    hidden = _move_hidden(model.init_hidden(B), device)
    out, _ = model(inp, hidden)
    last = out[torch.tensor(lens, device=device) - 1, torch.arange(B, device=device)]
    probs = torch.softmax(last, dim=-1)
    return probs[:, torch.tensor(noun_indices, device=device)].sum(dim=-1)

def compute_noun_mass_difference(model, d, m, noun_indices, batch_size, device):
    model.eval(); diffs = []
    with torch.no_grad():
        for i in range(0, len(d), batch_size):
            mass_d = pad_and_run(model, d[i:i+batch_size], noun_indices, device)
            mass_m = pad_and_run(model, m[i:i+batch_size], noun_indices, device)
            diffs.append((mass_d - mass_m).cpu())
    return torch.cat(diffs).mean().item()

def eval(checkpoint, dataset, dictionary, batch_size, test, device):
    model = lstm("LSTM", len(dictionary), 650, 650, 2, 0.2, False)
    model = load_checkpoint(model=model, checkpoint_path=checkpoint, device = device)
    d, m =tokenize_sentences(dataset, dictionary, test)
    noun_indices = find_noun_indices(dictionary)
    mean_diff = compute_noun_mass_difference(model, d, m, noun_indices, batch_size, device)
    return mean_diff

# --- checkpoint filename helpers (kept short & robust) ---
def extract_epoch_from_path(p):
    b = os.path.basename(p)
    if 'epoch_' in b:
        try: return int(b.split('epoch_')[1].split('_')[0])
        except: return 0
    return 0

def extract_batch_from_path(p):
    b = os.path.basename(p)
    if '_batch_' in b:
        try: return int(b.split('_batch_')[1].split('.')[0])
        except: return 0
    return 0

def get_checkpoint_sort_key(p):
    return (extract_epoch_from_path(p), extract_batch_from_path(p))


def main():
    parser = argparse.ArgumentParser(description='Evaluate syntactic knowledge in language models (FIXED)')
    parser.add_argument('--data_path', 
                       help='Path to training data directory', default = 'modulated_sets/sRC0.00009_sQ0.00250_tRC0.00009_tQ0.00500')
    parser.add_argument('--checkpoint_dir', 
                       help='Directory containing model checkpoints', default = 'checkpoints/train_RNNModel_sRC0.00009_sQ0.00250_tRC0.00009_tQ0.00500')
    parser.add_argument('--output_dir', default='.', 
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for processing (default: 64, reduced for stability)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto - uses CUDA if available)')
    
    
    args = parser.parse_args()
    
    checkpoint_pattern = os.path.join(args.checkpoint_dir, "*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files = sorted(checkpoint_files, key=get_checkpoint_sort_key)
    
    dataset_wh = pd.read_csv('wh_question_dataset.csv')
    dataset_orc = pd.read_csv('object_relative_dataset.csv')
    
    dictionary = Dictionary(args.data_path)
    metric = {}
    metric['wh'] = {}
    metric['orc']={}
    for i, checkpoint_path in enumerate(tqdm(checkpoint_files, desc="Evaluating checkpoints")):
        mean_diff_wh = eval(checkpoint = checkpoint_path, dataset = dataset_wh, dictionary = dictionary, batch_size = 10, test = 'wh', device = 'cuda')
        mean_diff_orc = eval(checkpoint = checkpoint_path, dataset = dataset_orc, dictionary = dictionary, batch_size = 10, test = 'orc', device = 'cuda')
        epoch = extract_epoch_from_path(checkpoint_path)
        batch = extract_batch_from_path(checkpoint_path)
        metric['wh'][f'{epoch}_{batch}']=mean_diff_wh
        metric['orc'][f'{epoch}_{batch}']=mean_diff_orc
     
    results_file = os.path.join(args.output_dir, f'obj_wh_test_{args.checkpoint_dir}.json')
    with open(results_file, 'w') as f:
        json.dump(metric, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
if __name__ == '__main__':
    main()