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

class OptimizedEvaluator:
    """Fixed evaluator with proper caching and memory management"""
    
    def __init__(self, data_path, device, batch_size=256, cache_models=False):
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.cache_models = cache_models
        
        # Cache for loaded models and dictionaries
        self.model_cache = {} if cache_models else None
        
        # Cache for word categories (computed once)
        self.word_categories_cache = {}
        
        # Dictionary cache (lightweight, always cache this)
        self.dictionary = None
        
    def get_word_categories(self, dictionary, test_type):
        """Get word categories with caching"""
        cache_key = test_type  # Use test_type only since we reuse the same dictionary
        
        if cache_key not in self.word_categories_cache:
            if test_type == 'questions':
                cat1, _, cat2 = get_question_marks_and_nouns_from_vocabulary(dictionary)
            else:
                cat1, cat2 = get_verbs_nouns_from_vocabulary(dictionary)
            
            # Pre-compute indices for faster lookup
            cat1_indices = [dictionary.word2idx.get(word) for word in cat1 
                           if word in dictionary.word2idx]
            cat2_indices = [dictionary.word2idx.get(word) for word in cat2 
                           if word in dictionary.word2idx]
            
            self.word_categories_cache[cache_key] = {
                'cat1_words': cat1,
                'cat2_words': cat2,
                'cat1_indices': np.array(cat1_indices, dtype=np.int32),
                'cat2_indices': np.array(cat2_indices, dtype=np.int32)
            }
        
        return self.word_categories_cache[cache_key]
    
    def load_model_cached(self, checkpoint_path):
        """Load model with proper caching and memory management"""
        # Check cache first if caching is enabled
        if self.cache_models and self.model_cache and checkpoint_path in self.model_cache:
            print(f"  Using cached model for {os.path.basename(checkpoint_path)}")
            return self.model_cache[checkpoint_path]
        
        print(f"  Loading model from {os.path.basename(checkpoint_path)}")
        
        # Clear GPU memory before loading new model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Load dictionary only once (it's the same for all checkpoints)
        if self.dictionary is None:
            print("  Loading dictionary...")
            self.dictionary = Dictionary(self.data_path)
            print(f"  Dictionary loaded with {len(self.dictionary)} words")
        
        # Initialize and load model
        model = lstm("LSTM", len(self.dictionary), 650, 650, 2, 0.2, False).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        result = (model, self.dictionary)
        
        # Cache management if caching is enabled
        if self.cache_models:
            # Limit cache size to prevent memory issues
            if len(self.model_cache) >= 2:  # Keep only 2 models in cache
                # Remove oldest cached model
                oldest_key = next(iter(self.model_cache))
                print(f"  Removing cached model: {os.path.basename(oldest_key)}")
                del self.model_cache[oldest_key]
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            self.model_cache[checkpoint_path] = result
        
        return result
    
    def clear_memory(self):
        """Clear caches and GPU memory"""
        if self.model_cache:
            print("  Clearing model cache...")
            self.model_cache.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def preprocess_sentences(self, test_sentences, dictionary):
        """Preprocess all sentences once, with improved batching"""
        valid_sentences = []
        sentence_data = []
        
        # Convert words to indices once
        unk_idx = dictionary.word2idx.get("<unk>", 0)
        
        for sentence in test_sentences:
            words = sentence.strip().split()
            if len(words) >= 2:
                # Convert context words to indices
                context_indices = []
                for word in words[:-1]:
                    idx = dictionary.word2idx.get(word, unk_idx)
                    context_indices.append(idx)
                
                valid_sentences.append(words)
                sentence_data.append({
                    'words': words,
                    'context_indices': context_indices,
                    'target_word': words[-1]
                })
        
        return valid_sentences, sentence_data
    
    def calculate_masses_vectorized(self, probs_batch, cat1_indices, cat2_indices):
        """Vectorized mass calculation for batch"""
        # Use advanced indexing for faster computation
        if len(cat1_indices) > 0:
            cat1_masses = probs_batch[:, cat1_indices].sum(axis=1)
        else:
            cat1_masses = np.zeros(probs_batch.shape[0])
            
        if len(cat2_indices) > 0:
            cat2_masses = probs_batch[:, cat2_indices].sum(axis=1)
        else:
            cat2_masses = np.zeros(probs_batch.shape[0])
        
        return cat1_masses, cat2_masses
    
    def evaluate_checkpoint_optimized(self, checkpoint_path, test_sentences, test_type):
        """Optimized evaluation with proper error handling"""
        print(f"\nEvaluating: {os.path.basename(checkpoint_path)}")
        
        try:
            # Load model (with proper tuple unpacking)
            model, dictionary = self.load_model_cached(checkpoint_path)
            
            if model is None or dictionary is None:
                raise ValueError("Failed to load model or dictionary")
            
            # Get word categories (cached)
            word_cats = self.get_word_categories(dictionary, test_type)
            cat1_indices = word_cats['cat1_indices']
            cat2_indices = word_cats['cat2_indices']
            
            print(f"  Found {len(cat1_indices)} category 1 words, {len(cat2_indices)} category 2 words")
            
            # Preprocess sentences (done once per checkpoint)
            valid_sentences, sentence_data = self.preprocess_sentences(test_sentences, dictionary)
            print(f"  Preprocessing complete: {len(sentence_data)} valid sentences")
            
            all_results = []
            total_batches = (len(sentence_data) + self.batch_size - 1) // self.batch_size
            
            with torch.no_grad():
                for batch_idx in range(0, len(sentence_data), self.batch_size):
                    batch_data = sentence_data[batch_idx:batch_idx + self.batch_size]
                    
                    # Prepare batch inputs more efficiently
                    max_len = max(len(item['context_indices']) for item in batch_data)
                    batch_size_actual = len(batch_data)
                    
                    # Pre-allocate tensor
                    input_tensor = torch.zeros((max_len, batch_size_actual), 
                                             dtype=torch.long, device=self.device)
                    
                    batch_lengths = []
                    for i, item in enumerate(batch_data):
                        seq_len = len(item['context_indices'])
                        input_tensor[:seq_len, i] = torch.tensor(item['context_indices'], 
                                                               dtype=torch.long, device=self.device)
                        batch_lengths.append(seq_len)
                    
                    # Forward pass
                    hidden = model.init_hidden(batch_size_actual)
                    output, _ = model(input_tensor, hidden)
                    
                    # Get final predictions efficiently
                    batch_logits = torch.stack([output[length-1, i] 
                                              for i, length in enumerate(batch_lengths)])
                    batch_probs = torch.softmax(batch_logits, dim=1).cpu().numpy()
                    
                    # Clean up tensors immediately to save memory
                    del input_tensor, batch_logits, output, hidden
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Vectorized mass calculation
                    cat1_masses, cat2_masses = self.calculate_masses_vectorized(
                        batch_probs, cat1_indices, cat2_indices)
                    
                    # Store results
                    for i, item in enumerate(batch_data):
                        cat1_mass = float(cat1_masses[i])
                        cat2_mass = float(cat2_masses[i])
                        
                        if test_type == 'questions':
                            cat1_name, cat2_name = 'question_mark', 'noun'
                        else:
                            cat1_name, cat2_name = 'verb', 'noun'
                        
                        all_results.append({
                            'sentence': ' '.join(item['words']),
                            'context': ' '.join(item['words'][:-1]),
                            'target': item['target_word'],
                            f'{cat1_name}_mass': cat1_mass,
                            f'{cat2_name}_mass': cat2_mass,
                            f'prefers_{cat1_name}': cat1_mass > cat2_mass
                        })
                    
                    if (batch_idx // self.batch_size + 1) % 10 == 0:
                        print(f"  Processed batch {batch_idx // self.batch_size + 1}/{total_batches}")
            
            # Clean up model if not caching
            if not self.cache_models:
                del model
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Calculate summary statistics
            if test_type == 'questions':
                cat1_name, cat2_name = 'question_mark', 'noun'
            else:
                cat1_name, cat2_name = 'verb', 'noun'
                
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
            
            print(f"  Processed {len(all_results)} sentences")
            print(f"  Avg {cat1_name} mass: {summary[f'avg_{cat1_name}_mass']:.4f}")
            print(f"  Avg {cat2_name} mass: {summary[f'avg_{cat2_name}_mass']:.4f}")
            print(f"  {cat1_name.title()} preference: {summary[f'{cat1_name}_preference_pct']:.1f}%")
            
            return summary, all_results
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            # Clean up on error
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return None, None


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
        elif word in ['.', '!', ',', ';', ':']:
            punctuation.append(word)
    
    # POS tag for nouns (batch process for speed)
    pos_tags = pos_tag(vocabulary)
    for word, tag in pos_tags:
        if tag.startswith('NN'):
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
        if tag.startswith('VB'):
            verbs.append(word)
        elif tag.startswith('NN'):
            nouns.append(word)
    
    return verbs, nouns

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

def save_results(summaries, output_dir, checkpoint_dir, test_type):
    """Save results to JSON file"""
    output_file = os.path.join(output_dir, f"{test_type}_{checkpoint_dir}.json")
    
    with open(output_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Evaluate syntactic knowledge in language models (FIXED)')
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
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for processing (default: 64, reduced for stability)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to use (default: auto - uses CUDA if available)')
    parser.add_argument('--cache_models', action='store_true',
                       help='Cache loaded models in memory (uses more RAM but faster)')
    parser.add_argument('--max_memory_gb', type=float, default=40.0,
                       help='Maximum GPU memory to use in GB (default: 40.0)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Set memory management environment variables for CUDA
    if device.type == 'cuda':
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        
        # Check GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Total Memory: {total_memory:.1f} GB")
        print(f"Using max memory limit: {args.max_memory_gb:.1f} GB")
        
        # Conservative memory management
        torch.backends.cudnn.benchmark = False  # Disable for memory savings
        torch.backends.cudnn.enabled = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find checkpoint files
    checkpoint_pattern = os.path.join(args.checkpoint_dir, "*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files = sorted(checkpoint_files, key=get_checkpoint_sort_key)
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    if len(checkpoint_files) == 0:
        print("No checkpoint files found!")
        return
    
    # Load test sentences
    with open(args.test_file, 'r') as f:
        test_sentences = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(test_sentences)} test sentences")
    
    # Print configuration
    print(f"=== FIXED {args.test_type.upper().replace('_', ' ')} EVALUATION ===")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model caching: {'Enabled' if args.cache_models else 'Disabled'}")
    
    # Create optimized evaluator
    evaluator = OptimizedEvaluator(args.data_path, device, args.batch_size, args.cache_models)
    
    # Run evaluation
    all_summaries = []
    
    try:
        for i, checkpoint_path in enumerate(checkpoint_files):
            print(f"\n[{i+1}/{len(checkpoint_files)}]", end=" ")
            
            # Monitor memory
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                print(f"(GPU: {memory_used:.1f}GB)", end=" ")
            
            summary, detailed_results = evaluator.evaluate_checkpoint_optimized(
                checkpoint_path, test_sentences, args.test_type
            )
            
            if summary:
                all_summaries.append(summary)
            
            # Clear cache periodically if not using model caching
            if not args.cache_models:
                evaluator.clear_memory()
    
    except Exception as e:
        print(f"Error during evaluation: {e}")
        evaluator.clear_memory()
        raise
    
    if all_summaries:
        # Print comparison table
        print_comparison_table(all_summaries, args.test_type)
        
        # Save results
        output_file = save_results(all_summaries, args.output_dir, args.checkpoint_dir, args.test_type)
        
        # Print summary
        print(f"\n=== EVALUATION SUMMARY ===")
        print(f"Processed {len(all_summaries)} checkpoints")
        print(f"Results saved to: {output_file}")
        
        if len(all_summaries) > 1:
            first_result = all_summaries[0]
            last_result = all_summaries[-1]
            if args.test_type == 'questions':
                pref_key = 'question_mark_preference_pct'
                improvement = last_result[pref_key] - first_result[pref_key]
                print(f"Question mark preference improved by {improvement:+.1f} percentage points")
            else:
                pref_key = 'verb_preference_pct'
                improvement = last_result[pref_key] - first_result[pref_key]
                print(f"Verb preference improved by {improvement:+.1f} percentage points")
    
    # Final cleanup
    evaluator.clear_memory()
    if device.type == 'cuda':
        print(f"\nFinal GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

if __name__ == "__main__":
    main()
    
# python pm_optimized.py --data_path modulated_sets/rc_0016_q_0050 --checkpoint_dir checkpoints/train_RNNModel_rc_0016_q_0050 --test_file /scratch2/mrenaudin/GTH/structure_test/relative_clauses_test.txt --test_type 'relative_clauses' --cache_models --device 'cuda'