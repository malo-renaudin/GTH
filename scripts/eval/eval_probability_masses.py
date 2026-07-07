import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def build_vocabulary():
    """Reconstructs the exact vocabulary sets from the generation scripts."""
    # ORC Vocabulary
    orc_nouns_base = ["boy", "student", "doctor", "artist", "athlete", 
                      "girl", "child", "pilot", "scientist", "engineer"]
    orc_irregulars = {"child": "children", "man": "men", "woman": "women"}
    orc_nouns = orc_nouns_base + [orc_irregulars.get(n, n + "s") for n in orc_nouns_base]
    
    orc_adjs = ["big", "tall", "young", "strong", "kind", 
                "beautiful", "smart", "brave", "famous", "honest", ""]
    
    orc_nps = []
    for adj in orc_adjs:
        for n in orc_nouns:
            if adj:
                orc_nps.append(f" the {adj} {n}")
            else:
                orc_nps.append(f" the {n}")
                
    orc_vps = [
        " is eating an apple", " are eating an apple", " is watching a movie", " are watching a movie",
        " is reading a book", " are reading a book", " likes to dance", " like to dance",
        " enjoys music", " enjoy music", " likes climbing", " like climbing"
    ]

    # WH Vocabulary
    wh_nouns_base = ["student", "doctor", "pilot", "officer", "athlete", "artist", 
                     "child", "girl", "boy", "patient", "client", "tourist"]
    wh_nouns = wh_nouns_base + [orc_irregulars.get(n, n + "s") for n in wh_nouns_base]
    
    wh_adjs = ["young", "tall", "smart", "brave", "kind", "famous", 
               "creative", "serious", "friendly", "quiet", "active", ""]
    
    wh_nps = []
    for adj in wh_adjs:
        for n in wh_nouns:
            if adj:
                wh_nps.append(f" the {adj} {n}")
            else:
                wh_nps.append(f" the {n}")
                
    wh_vps = [
        " visit", " visited", " visiting", " help", " helped", " helping",
        " greet", " greeted", " greeting", " follow", " followed", " following",
        " avoid", " avoided", " avoiding", " call", " called", " calling",
        " observe", " observed", " observing", " did", " does", " do", " is", " are"
    ]
    
    qm = [" ?"]
    
    return {
        "orc": {"NP": orc_nps, "VP": orc_vps, "?": qm},
        "wh": {"NP": wh_nps, "VP": wh_vps, "?": qm}
    }

def get_orc_context(sentence):
    """Gap site is strictly after the first verb for ORC."""
    orc_verbs = {"visits", "visit", "helps", "help", "avoids", "avoid", 
                 "follows", "follow", "greets", "greet"}
    words = sentence.split()
    for i, w in enumerate(words):
        if w.strip(".,?!") in orc_verbs:
            return " ".join(words[:i+1])
    return sentence

def get_wh_context(sentence):
    """Gap site is right before the interrogation mark for WH."""
    return sentence.replace("?", "").strip()

def evaluate_candidates_batched(model, tokenizer, context_text, candidates, batch_size=32):
    """
    Computes geometric mean probability for each candidate continuation.
    Efficiently computes P("the" | context) once by running the context through 
    the model once and reusing the KV cache and next-token logits for the batch.
    """
    device = model.device
    ctx_ids = tokenizer(context_text, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        ctx_out = model(ctx_ids, use_cache=True)
        
    past_kv = ctx_out.past_key_values
    C_len = ctx_ids.shape[1]
    
    # next_token_logits contains the predictions for the very first token of the candidate
    next_token_logits = ctx_out.logits[0, -1, :]
    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
    
    all_geom_means = []
    
    for i in range(0, len(candidates), batch_size):
        batch_cands = candidates[i:i+batch_size]
        batch_ids = [tokenizer(cand, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device) 
                     for cand in batch_cands]
        
        # 1. P("the" | context) equivalent - Fetch probabilities for the first token of all candidates
        first_tokens = torch.tensor([ids[0] for ids in batch_ids], device=device)
        lp0s = next_token_log_probs[first_tokens]
        
        max_len = max(len(ids) for ids in batch_ids)
        
        if max_len == 1:
            for lp0 in lp0s:
                all_geom_means.append(math.exp(lp0.item()))
            continue
            
        B = len(batch_cands)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # Prepare padded inputs for the remainder of the sequences
        input_ids_rest = torch.full((B, max_len - 1), pad_id, dtype=torch.long, device=device)
        target_ids_rest = torch.full((B, max_len - 1), -100, dtype=torch.long, device=device)
        attention_mask = torch.ones((B, C_len + max_len - 1), dtype=torch.long, device=device)
        
        for b, ids in enumerate(batch_ids):
            L = len(ids)
            if L > 1:
                input_ids_rest[b, :L-1] = ids[:-1]
                target_ids_rest[b, :L-1] = ids[1:]
            attention_mask[b, C_len + max(0, L - 1):] = 0
            
        # Expand context KV cache for the batch
        batch_kv = tuple(
            tuple(t.expand(B, -1, -1, -1) for t in layer)
            for layer in past_kv
        )
        
        with torch.no_grad():
            out_rest = model(input_ids_rest, attention_mask=attention_mask, past_key_values=batch_kv, use_cache=False)
            
        rest_log_probs = torch.log_softmax(out_rest.logits, dim=-1)
        
        # 2. Reconstruct probabilities (doing the geometric mean)
        for b in range(B):
            L = len(batch_ids[b])
            log_probs = [lp0s[b].item()]
            if L > 1:
                for pos in range(L - 1):
                    t_id = target_ids_rest[b, pos]
                    log_probs.append(rest_log_probs[b, pos, t_id].item())
            
            geom_mean = math.exp(sum(log_probs) / len(log_probs))
            all_geom_means.append(geom_mean)
            
    return all_geom_means

def process_dataset(file_path, model, tokenizer, vocab, context_fn, batch_size):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
        
    results = {"NP": 0.0, "VP": 0.0, "?": 0.0}
    
    for sentence in tqdm(sentences, desc=f"Evaluating {file_path}"):
        context = context_fn(sentence)
        
        for category, candidates in vocab.items():
            geom_means = evaluate_candidates_batched(model, tokenizer, context, candidates, batch_size)
            # Normalize probability masses by the number of evaluated candidates
            mass = sum(geom_means) / len(candidates)
            results[category] += mass
            
    # Average across all sentences
    for cat in results:
        results[cat] /= len(sentences)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate probability masses at gap sites.")
    parser.add_argument("--model", type=str, required=True, help="HF model checkpoint")
    parser.add_argument("--orc_test", type=str, required=True, help="Path to ORC test file")
    parser.add_argument("--wh_test", type=str, required=True, help="Path to WH test file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for candidate evaluation")
    
    args = parser.parse_args()
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    
    vocab = build_vocabulary()
    
    print("\n--- Processing ORC Dataset ---")
    orc_results = process_dataset(args.orc_test, model, tokenizer, vocab["orc"], get_orc_context, args.batch_size)
    print("ORC Probability Masses:")
    for cat, mass in orc_results.items():
        print(f"  {cat}: {mass:.6f}")

    print("\n--- Processing WH Dataset ---")
    wh_results = process_dataset(args.wh_test, model, tokenizer, vocab["wh"], get_wh_context, args.batch_size)
    print("WH Probability Masses:")
    for cat, mass in wh_results.items():
        print(f"  {cat}: {mass:.6f}")

if __name__ == "__main__":
    main()