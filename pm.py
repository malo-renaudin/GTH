
import torch
import numpy as np
import os
import glob
import json
import argparse
from collections import defaultdict
from nltk.tag import pos_tag
from src.language_models.model import RNNModel as lstm
from src.language_models.dictionary_corpus import Dictionary



def load_model_and_dictionary(checkpoint_path, data_path, device):

    
    # Load dictionary
    dictionary = Dictionary(data_path)
    
    # Initialize model
    model = lstm("LSTM", len(dictionary), 650, 650, 2, 0.2, False).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, dictionary

def get_question_marks_and_nouns_from_vocabulary(dictionary):
    """Get question marks and nouns from vocabulary using POS tagging"""
    vocabulary = dictionary.idx2word
    
    # Question marks and punctuation
    question_marks = []
    punctuation = []
    nouns = []
    
    # Find punctuation directly
    for word in vocabulary:
        if word == '?':
            question_marks.append(word)
        elif word in ['.', '!', '?', ',', ';', ':']:
            punctuation.append(word)
    
    # POS tag for nouns
    pos_tags = pos_tag(vocabulary)
    for word, tag in pos_tags:
        if tag.startswith('NN'):  # All noun forms
            nouns.append(word)
    
    return question_marks, punctuation, nouns

def get_verbs_nouns_from_vocabulary(dictionary):
    """Use POS tagging to identify verbs and nouns in the vocabulary"""
    vocabulary = dictionary.idx2word
    verbs = []
    nouns = []
    
    # POS tag all words in vocabulary
    pos_tags = pos_tag(vocabulary)
    
    for word, tag in pos_tags:
        if tag.startswith('VB'):  # VB, VBD, VBG, VBN, VBP, VBZ (all verb forms)
            verbs.append(word)
        elif tag.startswith('NN'):  # NN, NNS, NNP, NNPS (all noun forms)
            nouns.append(word)
    
    return verbs, nouns

def calculate_question_noun_masses(probs, word_to_idx, question_marks, nouns):
    """Calculate probability masses for question marks vs nouns."""
    question_mass = 0.0
    noun_mass = 0.0
    
    # Sum probabilities for question marks
    for qmark in question_marks:
        if qmark in word_to_idx:
            idx = word_to_idx[qmark]
            question_mass += probs[idx]
    
    # Sum probabilities for nouns
    for noun in nouns:
        if noun in word_to_idx:
            idx = word_to_idx[noun]
            noun_mass += probs[idx]
    
    return question_mass, noun_mass

def calculate_verb_noun_masses(probs, word_to_idx, verbs, nouns):
    """Calculate probability masses for verbs and nouns using POS-tagged lists."""
    verb_mass = 0.0
    noun_mass = 0.0
    
    # Sum probabilities for all verbs
    for verb in verbs:
        if verb in word_to_idx:
            idx = word_to_idx[verb]
            verb_mass += probs[idx]
    
    # Sum probabilities for all nouns  
    for noun in nouns:
        if noun in word_to_idx:
            idx = word_to_idx[noun]
            noun_mass += probs[idx]
    
    return verb_mass, noun_mass

def evaluate_single_checkpoint(checkpoint_path, data_path, test_sentences, category1_words, category2_words, 
                              device, test_type, batch_size=64):
    """Evaluate a single checkpoint with batch processing for either test type"""
    print(f"\nEvaluating: {os.path.basename(checkpoint_path)}")
    
    try:
        # Load model
        model, dictionary = load_model_and_dictionary(checkpoint_path, data_path, device)
        
        # Prepare all sentences
        valid_sentences = []
        for sentence in test_sentences:
            words = sentence.strip().split()
            if len(words) >= 2:
                valid_sentences.append(words)
        
        # Process in batches
        all_results = []
        
        for batch_start in range(0, len(valid_sentences), batch_size):
            batch_sentences = valid_sentences[batch_start:batch_start + batch_size]
            
            # Prepare batch inputs
            batch_inputs = []
            batch_lengths = []
            
            for words in batch_sentences:
                # Convert to indices (excluding last word)
                indices = []
                for word in words[:-1]:  # All except last word
                    idx = dictionary.word2idx.get(word, dictionary.word2idx.get("<unk>"))
                    indices.append(idx)
                
                batch_inputs.append(indices)
                batch_lengths.append(len(indices))
            
            # Pad sequences to same length
            max_len = max(batch_lengths)
            padded_inputs = []
            for seq in batch_inputs:
                padded = seq + [0] * (max_len - len(seq))
                padded_inputs.append(padded)
            
            # Convert to tensor: (batch_size, seq_len) -> (seq_len, batch_size)
            input_tensor = torch.tensor(padded_inputs).transpose(0, 1).to(device)
            
            # Forward pass
            with torch.no_grad():
                batch_size_actual = len(batch_sentences)
                hidden = model.init_hidden(batch_size_actual)
                output, _ = model(input_tensor, hidden)  # (seq_len, batch_size, vocab_size)
                
                # Get predictions for each sequence's last actual position
                batch_logits = []
                for i, length in enumerate(batch_lengths):
                    batch_logits.append(output[length-1, i])
                
                batch_logits = torch.stack(batch_logits)  # (batch_size, vocab_size)
                batch_probs = torch.softmax(batch_logits, dim=1).cpu().numpy()
            
            # Calculate masses for each sentence in batch
            for i, words in enumerate(batch_sentences):
                probs = batch_probs[i]
                
                if test_type == 'questions':
                    mass1, mass2 = calculate_question_noun_masses(
                        probs, dictionary.word2idx, category1_words, category2_words
                    )
                    cat1_name, cat2_name = 'question_mark', 'noun'
                else:  # relative_clauses
                    mass1, mass2 = calculate_verb_noun_masses(
                        probs, dictionary.word2idx, category1_words, category2_words
                    )
                    cat1_name, cat2_name = 'verb', 'noun'
                
                all_results.append({
                    'sentence': ' '.join(words),
                    'context': ' '.join(words[:-1]),
                    'target': words[-1],
                    f'{cat1_name}_mass': mass1,
                    f'{cat2_name}_mass': mass2,
                    f'prefers_{cat1_name}': mass1 > mass2
                })
        
        # Calculate summary statistics
        mass1_values = [r[f'{cat1_name}_mass'] for r in all_results]
        mass2_values = [r[f'{cat2_name}_mass'] for r in all_results]
        preferences = sum(1 for r in all_results if r[f'prefers_{cat1_name}'])
        
        summary = {
            'checkpoint': os.path.basename(checkpoint_path),
            'epoch': extract_epoch_from_path(checkpoint_path),
            'batch': extract_batch_from_path(checkpoint_path),
            'test_type': test_type,
            'num_sentences': len(all_results),
            f'avg_{cat1_name}_mass': float(np.mean(mass1_values)),
            f'avg_{cat2_name}_mass': float(np.mean(mass2_values)),
            f'{cat1_name}_preference_pct': float(preferences / len(all_results) * 100),
            f'{cat1_name}_{cat2_name}_ratio': float(np.mean(mass1_values) / np.mean(mass2_values)) if np.mean(mass2_values) > 0 else float('inf')
        }
        
        print(f"  Processed {len(all_results)} sentences in {len(all_results)//batch_size + 1} batches")
        print(f"  Avg {cat1_name} mass: {summary[f'avg_{cat1_name}_mass']:.4f}")
        print(f"  Avg {cat2_name} mass: {summary[f'avg_{cat2_name}_mass']:.4f}")
        print(f"  {cat1_name.title()} preference: {summary[f'{cat1_name}_preference_pct']:.1f}%")
        
        return summary, all_results
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None

def extract_epoch_from_path(checkpoint_path):
    """Extract epoch number from checkpoint filename"""
    basename = os.path.basename(checkpoint_path)
    
    if 'epoch_' in basename and '_batch_' in basename:
        try:
            epoch_str = basename.split('epoch_')[1].split('_batch_')[0]
            return int(epoch_str)
        except:
            return 0
    elif 'epoch_' in basename:
        try:
            epoch_str = basename.split('epoch_')[1].split('.')[0]
            return int(epoch_str)
        except:
            return 0
    
    return 0

def extract_batch_from_path(checkpoint_path):
    """Extract batch number from checkpoint filename (if present)"""
    basename = os.path.basename(checkpoint_path)
    
    if '_batch_' in basename:
        try:
            batch_str = basename.split('_batch_')[1].split('.')[0]
            return int(batch_str)
        except:
            return 0
    
    return 0

def get_checkpoint_sort_key(checkpoint_path):
    """Create sort key for checkpoints: (epoch, batch)"""
    epoch = extract_epoch_from_path(checkpoint_path)
    batch = extract_batch_from_path(checkpoint_path)
    return (epoch, batch)

def evaluate_all_checkpoints(checkpoint_dir, data_path, test_sentences_file, test_type, device):
    """Evaluate all checkpoints in a directory"""
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files = sorted(checkpoint_files, key=get_checkpoint_sort_key)
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    if len(checkpoint_files) == 0:
        print("No checkpoint files found!")
        return
    
    # Load test sentences
    with open(test_sentences_file, 'r') as f:
        test_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(test_sentences)} test sentences")
    
    # Get word categories from vocabulary (using first checkpoint)
    print("Getting word categories from vocabulary...")
    _, first_dictionary = load_model_and_dictionary(checkpoint_files[0], data_path, device)
    
    if test_type == 'questions':
        category1_words, punctuation, category2_words = get_question_marks_and_nouns_from_vocabulary(first_dictionary)
        print(f"Found question marks: {category1_words}")
        print(f"Found {len(punctuation)} punctuation marks")
        print(f"Found {len(category2_words)} nouns")
        
        if len(category1_words) == 0:
            print("WARNING: No question marks found in vocabulary!")
            return
    else:  # relative_clauses
        category1_words, category2_words = get_verbs_nouns_from_vocabulary(first_dictionary)
        print(f"Found {len(category1_words)} verbs and {len(category2_words)} nouns")
    
    # Evaluate all checkpoints
    all_summaries = []
    
    for checkpoint_path in checkpoint_files:
        summary, detailed_results = evaluate_single_checkpoint(
            checkpoint_path, data_path, test_sentences, category1_words, category2_words, 
            device, test_type, batch_size=64
        )
        
        if summary:
            all_summaries.append(summary)
    
    # Print comparison table
    print_comparison_table(all_summaries, test_type)
    
    return all_summaries

def print_comparison_table(summaries, test_type):
    """Print a comparison table of all checkpoints"""
    if test_type == 'questions':
        title = "QUESTION MARK vs NOUN ANALYSIS - CHECKPOINT COMPARISON"
        cat1_name, cat2_name = 'Question Mass', 'Noun Mass'
        ratio_name, pref_name = 'Q/N Ratio', 'Question Pref %'
    else:
        title = "VERB vs NOUN ANALYSIS - CHECKPOINT COMPARISON"
        cat1_name, cat2_name = 'Verb Mass', 'Noun Mass'
        ratio_name, pref_name = 'V/N Ratio', 'Verb Pref %'
    
    print(f"\n{'='*100}")
    print(title)
    print('='*100)
    print(f"{'Checkpoint':<25} {'Epoch':<6} {'Batch':<8} {cat1_name:<14} {cat2_name:<12} {ratio_name:<10} {pref_name:<15}")
    print('-'*100)
    
    for summary in summaries:
        if test_type == 'questions':
            ratio_key = 'question_mark_noun_ratio'
            pref_key = 'question_mark_preference_pct'
            mass1_key = 'avg_question_mark_mass'
            mass2_key = 'avg_noun_mass'
        else:
            ratio_key = 'verb_noun_ratio'
            pref_key = 'verb_preference_pct'
            mass1_key = 'avg_verb_mass'
            mass2_key = 'avg_noun_mass'
        
        ratio_str = f"{summary[ratio_key]:.3f}" if summary[ratio_key] != float('inf') else "inf"
        batch_str = str(summary['batch']) if summary['batch'] > 0 else "-"
        print(f"{summary['checkpoint']:<25} {summary['epoch']:<6} {batch_str:<8} "
              f"{summary[mass1_key]:<14.4f} {summary[mass2_key]:<12.4f} "
              f"{ratio_str:<10} {summary[pref_key]:<15.1f}")

def save_results(summaries, output_dir, test_type):
    """Save results to JSON file"""
    output_file = os.path.join(output_dir, f"{test_type}_analysis_results.json")
    
    with open(output_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Evaluate syntactic knowledge in language models')
    parser.add_argument('--data_path', required=True, 
                       help='Path to training data directory')
    parser.add_argument('--checkpoint_dir', required=True, 
                       help='Directory containing model checkpoints')
    parser.add_argument('--test_file', required=True, 
                       help='Path to test sentences file')
    parser.add_argument('--output_dir', default='.', 
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--test_type', required=True, choices=['questions', 'relative_clauses'],
                       help='Type of syntactic test to run')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing (default: 64)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto - uses CUDA if available)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print(f"=== {args.test_type.upper().replace('_', ' ')} EVALUATION ===")
    print(f"Data path: {args.data_path}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    if args.test_type == 'questions':
        print("Testing: Question mark preference vs noun preference")
        print("Expected: High '?' mass = learned question structure")
    else:
        print("Testing: Verb preference vs noun preference")
        print("Expected: High verb mass = learned relative clause structure")
    
    # Run evaluation
    results = evaluate_all_checkpoints(args.checkpoint_dir, args.data_path, 
                                     args.test_file, args.test_type, device)
    
    if results:
        # Save results
        output_file = save_results(results, args.output_dir, args.test_type)
        
        # Print summary
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Processed {len(results)} checkpoints")
        print(f"Results saved to: {output_file}")
        
        if len(results) > 1:
            first_result = results[0]
            last_result = results[-1]
            if args.test_type == 'questions':
                pref_key = 'question_mark_preference_pct'
                improvement = last_result[pref_key] - first_result[pref_key]
                print(f"Question mark preference improved by {improvement:+.1f} percentage points")
            else:
                pref_key = 'verb_preference_pct'
                improvement = last_result[pref_key] - first_result[pref_key]
                print(f"Verb preference improved by {improvement:+.1f} percentage points")

if __name__ == "__main__":
    
    main()