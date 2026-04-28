import random
from itertools import product
import os
import shutil

dataset = "data/base_data/train.txt"
relative_clauses = "english-stimuli-generator/orc_formatted.txt"
starting_freq = 0.03/100
target_freq = [0.04/100, 0.06/100]#[0.08/100, 0.16/100, 0.32/100]

def load_remaining_sentences(relative_clauses):
    """
    Load, shuffle, split in train-test
    """
    with open(relative_clauses, 'r', encoding='utf-8') as f:
        all_rc_sentences = [line.strip() for line in f if line.strip()]
    
    random.shuffle(all_rc_sentences)

    train_rc = all_rc_sentences[1000:]
    test_rc = all_rc_sentences[:1000]

    return train_rc, test_rc

def vary_freq(main_file, rc_sentences, source, target):

    with open(main_file, 'r', encoding='utf-8') as f:
        main_sentences = [line.strip() for line in f if line.strip()]
    
    N = len(main_sentences)

    # a: nb of sentences to remove and rc to add to reach 
    # target frequency while keeping the number of sentences stable
    a = int(N*(target - source)/(1 - source))

    if a > len(rc_sentences):
        print(f"Not enough sentences to add. Needed: {a}, Available: {len(rc_sentences)}")
    else : 
        modified_dataset = main_sentences.copy()
        random.shuffle(modified_dataset)
        remaining_sentences = modified_dataset[:(N - a)]
        new_sentences = random.sample(rc_sentences, a)
        modified_dataset = remaining_sentences + new_sentences
        random.shuffle(modified_dataset)
        assert len(modified_dataset) == N, f"Expected {N} sentences, got {len(modified_dataset)}"
        output_file = f"train_freq_{int(target*10000)}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in modified_dataset:
                f.write(sentence + '\n')
        print(f"Created dataset with target frequency {target:.5f} at {output_file}")
        
if __name__ == "__main__":
    train_rc, test_rc = load_remaining_sentences(relative_clauses)
    output_test_file = "test_orc2.txt"
    with open(output_test_file, 'w', encoding='utf-8') as f:
        for sentence in test_rc:
            f.write(sentence + '\n')
    for target in target_freq:
        vary_freq(dataset, train_rc, starting_freq, target)