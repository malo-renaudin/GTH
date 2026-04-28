import random
from itertools import product
import os
import shutil

# Configuration
dataset = "/scratch2/mrenaudin/GTH/english_data/train.txt"
relative_clauses = "/scratch2/mrenaudin/GTH/ORC.txt"
questions = "/scratch2/mrenaudin/GTH/english-stimuli-generator/q_obj.txt"
starting_frequencies = {'relative_clauses': 0.03/100, 'questions': 0.25/100}
target_frequencies = {
    'relative_clauses': [0.08/100, 0.16/100, 0.32/100], 
    'questions': [0.5/100, 1/100, 2/100]
}

def load_remaining_sentences():
    """
    Load the remaining sentences (after removing 1000 for test sets).
    """
    # Load all sentences
    with open(relative_clauses, 'r', encoding='utf-8') as f:
        all_rc_sentences = [line.strip() for line in f if line.strip()]
    
    with open(questions, 'r', encoding='utf-8') as f:
        all_q_sentences = [line.strip() for line in f if line.strip()]
    
    # Keep the remaining sentences for training (skip first 1000)
    remaining_rc = all_rc_sentences[1000:]
    remaining_q = all_q_sentences[1000:]
    
    return remaining_rc, remaining_q

def vary_freq(main_file, rc_sentences, q_sentences, s_rc, s_q, t_rc, t_q):
    """
    Create a dataset with specified target frequencies for relative clauses and questions.
    """
    # Load main dataset
    with open(main_file, 'r', encoding='utf-8') as f:
        main_sentences = [line.strip() for line in f if line.strip()]
    
    N = len(main_sentences)
    
    # Calculate number of sentences to add (keeping original computation)
    k_rc = int(((1-s_q)*N*t_rc)/(1-s_q-s_q*s_rc))
    k_q = int(((1-s_rc)*N*t_q)/(1-s_rc-s_q*s_rc))
    
    # Ensure we have enough sentences
    if k_rc > len(rc_sentences):
        k_rc = len(rc_sentences)
    if k_q > len(q_sentences):
        k_q = len(q_sentences)
    
    # Create the modified dataset
    modified_dataset = main_sentences.copy()
    random.shuffle(modified_dataset)
    
    # Remove sentences to make room for new ones
    remaining_sentences = modified_dataset[:(N - k_rc - k_q)]
    
    # Sample the required sentences
    selected_rc_sentences = random.sample(rc_sentences, k_rc) if k_rc > 0 else []
    selected_q_sentences = random.sample(q_sentences, k_q) if k_q > 0 else []
    
    # Combine all sentences
    final_dataset = remaining_sentences + selected_rc_sentences + selected_q_sentences
    random.shuffle(final_dataset)
    
    return final_dataset

def create_missing_baseline_datasets():
    """
    Create datasets where one structure stays at starting frequency while the other varies.
    """
    # Load remaining sentences (after test set removal)
    remaining_rc, remaining_q = load_remaining_sentences()
    
    output_dir = "/scratch2/mrenaudin/GTH/modulated_sets/"
    os.makedirs(output_dir, exist_ok=True)
    
    s_rc = starting_frequencies['relative_clauses']
    s_q = starting_frequencies['questions']
    
    missing_combinations = []
    
    # Case 1: RC stays at starting frequency, Q varies to target frequencies
    for t_q in target_frequencies['questions']:
        missing_combinations.append((s_rc, t_q))
    
    # Case 2: Q stays at starting frequency, RC varies to target frequencies
    for t_rc in target_frequencies['relative_clauses']:
        missing_combinations.append((t_rc, s_q))
    
    print(f"Creating {len(missing_combinations)} missing baseline datasets...")
    
    for t_rc, t_q in missing_combinations:
        try:
            dataset = vary_freq(dataset, remaining_rc, remaining_q, s_rc, s_q, t_rc, t_q)
            
            # Create folder name
            rc_percent = int(t_rc * 100 * 100)
            q_percent = int(t_q * 100 * 100)
            folder_name = f"rc_{rc_percent:04d}_q_{q_percent:04d}"
            folder_path = os.path.join(output_dir, folder_name)
            
            # Check if folder already exists
            if os.path.exists(folder_path):
                print(f"Folder {folder_name} already exists, skipping...")
                continue
            
            # Create subfolder
            os.makedirs(folder_path, exist_ok=True)
            
            # Save training dataset as train.txt
            train_filepath = os.path.join(folder_path, "train.txt")
            with open(train_filepath, 'w', encoding='utf-8') as f:
                for sentence in dataset:
                    f.write(sentence + '\n')
            
            # Copy test.txt and validation.txt from english_data folder
            source_test = "/scratch2/mrenaudin/GTH/english_data/test.txt"
            source_validation = "/scratch2/mrenaudin/GTH/english_data/valid.txt"
            source_vocab = "/scratch2/mrenaudin/GTH/english_data/vocab.txt"
            
            dest_test = os.path.join(folder_path, "test.txt")
            dest_validation = os.path.join(folder_path, "valid.txt")
            dest_vocab = os.path.join(folder_path, 'vocab.txt')
            
            if os.path.exists(source_test):
                shutil.copy2(source_test, dest_test)
            if os.path.exists(source_validation):
                shutil.copy2(source_validation, dest_validation)
            if os.path.exists(source_vocab):
                shutil.copy2(source_vocab, dest_vocab)
            
            print(f"Created folder: {folder_name} (RC: {t_rc*100:.4f}%, Q: {t_q*100:.4f}%)")
            
        except Exception as e:
            print(f"Error creating dataset for RC:{t_rc*100:.4f}%, Q:{t_q*100:.4f}%: {str(e)}")
    
    print("\nMissing baseline datasets creation complete!")
    print("\nCreated datasets:")
    print("RC at baseline (0.03%), Q varying:")
    for t_q in target_frequencies['questions']:
        print(f"  - RC: 0.0300%, Q: {t_q*100:.4f}%")
    print("Q at baseline (0.25%), RC varying:")
    for t_rc in target_frequencies['relative_clauses']:
        print(f"  - RC: {t_rc*100:.4f}%, Q: 0.2500%")

# Run the missing dataset creation
if __name__ == "__main__":
    create_missing_baseline_datasets()