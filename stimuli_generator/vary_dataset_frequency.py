# from analyze_train_sets.interrogative import count_interrogative_sentences
import random

dataset = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data/test.txt"
relative_clauses = "/scratch2/mrenaudin/colorlessgreenRNNs/stimuli_generator/relative_clauses.txt"
questions = "/scratch2/mrenaudin/colorlessgreenRNNs/stimuli_generator/questions.txt"

# relative clause starting frequency extracted from https://pmc.ncbi.nlm.nih.gov/articles/PMC2722756/pdf/nihms30776.pdf
starting_frequencies = {'relative_clauses': 0.04/100, 'questions': 0.25/100}

frequencies = {'relative_clauses': [
    0.04/100, 0.08/100, 0.16/100, 0.32/100], 'questions': [0.25/100, 0.5/100, 1/100, 2/100]}


def augment_structure_freq(starting_freq: int,
                           target_freq: int,
                           dataset: str,
                           structure_dataset: str,
                           save_path: str):
    """
    With N sentences, and x % of sentences with a given syntactic structure, we get 0.01xN sentences with this structure, 
    and (1-0.01x)N sentences without. Our goal is to find k, the number of sentences to randomly remove, and to add with this structure,
    such that the final % of sentences with this structure is a, the target frequency. 

    1. After removal : N-k sentences, 0.01x(N-k) with structure, (1-0.01x)(N-k) without
    2. When adding k sentences with structure : N sentences, 0.01x(N-k)+k with structure, (1-0.01x)(N-k) without
    3. Finally, we want k such that (0.01x(N-k)+k)/N = 0.01a, which gives k = (0.01(a-x)N)/(1-0.01x).

    Put simply, k = % of additional sentences with structure in the starting dataset / % of sentences without the structure in the starting dataset

    """
    # Read the datasets
    with open(dataset, 'r', encoding='utf-8') as f:
        main_sentences = [line.strip() for line in f if line.strip()]

    with open(structure_dataset, 'r', encoding='utf-8') as f:
        structure_sentences = [line.strip() for line in f if line.strip()]

    # Calculate k using your formula
    additional_freq = target_freq - starting_freq
    without_freq = 1 - starting_freq
    k_proportion = additional_freq / without_freq

    N = len(main_sentences)
    k = int(round(k_proportion * N))
    print(k)
    # Validate that we can perform the operation
    if target_freq <= starting_freq:
        raise ValueError(
            "Target frequency must be greater than starting frequency for this augmentation method")

    if k > len(structure_sentences):
        raise ValueError(
            f"Not enough structure sentences available. Need {k}, but only have {len(structure_sentences)}")

    if k > N:
        raise ValueError(f"k ({k}) cannot be greater than dataset size ({N})")

    # Step 1: Randomly remove k sentences from the main dataset
    modified_dataset = main_sentences.copy()
    random.shuffle(modified_dataset)
    removed_sentences = modified_dataset[:k]
    remaining_sentences = modified_dataset[k:]

    # Step 2: Randomly select k sentences from structure dataset
    selected_structure_sentences = random.sample(structure_sentences, k)

    # Step 3: Add the structure sentences to get back to original size
    final_dataset = remaining_sentences + selected_structure_sentences

    # Shuffle to randomize order
    random.shuffle(final_dataset)

    print(f"Original dataset size: {N}")
    print(f"Removed {k} sentences, added {k} structure sentences")
    print(f"Final dataset size: {len(final_dataset)}")
    print(f"Expected structure frequency: {target_freq:.3f}")

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            for sentence in final_dataset:
                f.write(sentence + '\n')
        print(f"Dataset saved to: {save_path}")

    return final_dataset


def validate_frequency_difference(original_dataset_path: str,
                                  generated_dataset_path: str,
                                  structure_dataset_path: str,
                                  target_frequency: int):
    """
    Validate that the percentage of different sentences between original and generated
    datasets matches the target frequency.

    Args:
        original_dataset_path (str): Path to original dataset
        generated_dataset_path (str): Path to generated dataset
        structure_dataset_path (str): Path to structure dataset
        target_frequency (int): Target frequency percentage (e.g., 15 for 15%)

    Returns:
        dict: Validation results
    """

    # Read all datasets
    with open(original_dataset_path, 'r', encoding='utf-8') as f:
        original_sentences = set(line.strip() for line in f if line.strip())

    with open(generated_dataset_path, 'r', encoding='utf-8') as f:
        generated_sentences = set(line.strip() for line in f if line.strip())

    with open(structure_dataset_path, 'r', encoding='utf-8') as f:
        structure_sentences = set(line.strip() for line in f if line.strip())

    # Calculate differences
    different_sentences = original_sentences.symmetric_difference(
        generated_sentences)
    sentences_only_in_original = original_sentences - generated_sentences
    sentences_only_in_generated = generated_sentences - original_sentences
    common_sentences = original_sentences & generated_sentences

    # Calculate percentages
    total_original = len(original_sentences)
    total_generated = len(generated_sentences)

    different_count = len(different_sentences)
    different_percentage = (different_count / total_original) * \
        100 if total_original > 0 else 0

    removed_count = len(sentences_only_in_original)
    added_count = len(sentences_only_in_generated)

    removed_percentage = (removed_count / total_original) * \
        100 if total_original > 0 else 0
    added_percentage = (added_count / total_original) * \
        100 if total_original > 0 else 0

    # Check if added sentences come from structure dataset
    added_from_structure = sentences_only_in_generated & structure_sentences
    added_from_structure_count = len(added_from_structure)

    # Validation results
    validation_results = {
        'dataset_sizes': {
            'original_size': total_original,
            'generated_size': total_generated,
            'sizes_match': total_original == total_generated
        },
        'difference_analysis': {
            'total_different_sentences': different_count,
            'different_percentage': round(different_percentage, 2),
            'target_frequency': target_frequency,
            # Allow small rounding error
            'matches_target': abs(different_percentage - target_frequency) < 0.5
        },
        'change_breakdown': {
            'sentences_removed': removed_count,
            'sentences_added': added_count,
            'removed_percentage': round(removed_percentage, 2),
            'added_percentage': round(added_percentage, 2),
            'symmetric_change': removed_count == added_count
        },
        'structure_source': {
            'added_from_structure': added_from_structure_count,
            'added_from_other': added_count - added_from_structure_count,
            'all_additions_from_structure': added_from_structure_count == added_count
        },
        'common_sentences': {
            'sentences_in_common': len(common_sentences),
            'common_percentage': round((len(common_sentences) / total_original) * 100, 2) if total_original > 0 else 0
        }
    }

    # Overall validation
    validation_passed = all([
        validation_results['dataset_sizes']['sizes_match'],
        validation_results['difference_analysis']['matches_target'],
        validation_results['change_breakdown']['symmetric_change'],
        validation_results['structure_source']['all_additions_from_structure']
    ])

    validation_results['overall'] = {
        'validation_passed': validation_passed,
        'frequency_difference_correct': validation_results['difference_analysis']['matches_target']
    }

    # Print results
    print("=== FREQUENCY DIFFERENCE VALIDATION ===")
    print(f"Original dataset size: {total_original}")
    print(f"Generated dataset size: {total_generated}")
    print(
        f"Dataset sizes match: {validation_results['dataset_sizes']['sizes_match']}")
    print()

    print(f"Target frequency: {target_frequency}%")
    print(f"Actual different sentences: {different_count}")
    print(f"Actual difference percentage: {different_percentage:.2f}%")
    print(
        f"Matches target frequency: {validation_results['difference_analysis']['matches_target']}")
    print()

    print(
        f"Sentences removed from original: {removed_count} ({removed_percentage:.2f}%)")
    print(
        f"Sentences added to generated: {added_count} ({added_percentage:.2f}%)")
    print(
        f"Symmetric change (removed = added): {validation_results['change_breakdown']['symmetric_change']}")
    print()

    print(
        f"Added sentences from structure dataset: {added_from_structure_count}")
    print(
        f"Added sentences from other sources: {validation_results['structure_source']['added_from_other']}")
    print(
        f"All additions from structure dataset: {validation_results['structure_source']['all_additions_from_structure']}")
    print()

    print(
        f"Sentences in common: {len(common_sentences)} ({validation_results['common_sentences']['common_percentage']:.2f}%)")
    print()

    print(
        f"OVERALL VALIDATION: {'✓ PASSED' if validation_passed else '✗ FAILED'}")

    if not validation_passed:
        print("\nFAILED CHECKS:")
        if not validation_results['dataset_sizes']['sizes_match']:
            print("  - Dataset sizes don't match")
        if not validation_results['difference_analysis']['matches_target']:
            print(
                f"  - Difference percentage ({different_percentage:.2f}%) doesn't match target ({target_frequency}%)")
        if not validation_results['change_breakdown']['symmetric_change']:
            print("  - Number of removed and added sentences don't match")
        if not validation_results['structure_source']['all_additions_from_structure']:
            print("  - Some added sentences don't come from structure dataset")

    return validation_results

# def show_difference_samples(original_dataset_path: str,
#                           generated_dataset_path: str,
#                           n_samples: int = 5):
#     """
#     Show sample sentences that differ between original and generated datasets
#     """
#     with open(original_dataset_path, 'r', encoding='utf-8') as f:
#         original_sentences = set(line.strip() for line in f if line.strip())

#     with open(generated_dataset_path, 'r', encoding='utf-8') as f:
#         generated_sentences = set(line.strip() for line in f if line.strip())

#     removed_sentences = original_sentences - generated_sentences
#     added_sentences = generated_sentences - original_sentences

#     print(f"\n=== SAMPLE DIFFERENCES (showing up to {n_samples} each) ===")

#     print(f"\nREMOVED from original ({len(removed_sentences)} total):")
#     for i, sentence in enumerate(list(removed_sentences)[:n_samples]):
#         print(f"  {i+1}. {sentence}")

#     print(f"\nADDED to generated ({len(added_sentences)} total):")
#     for i, sentence in enumerate(list(added_sentences)[:n_samples]):
#         print(f"  {i+1}. {sentence}")


if __name__ == "__main__":
    dataset = "/scratch2/mrenaudin/colorlessgreenRNNs/english_data/test.txt"
    relative_clauses = "/scratch2/mrenaudin/colorlessgreenRNNs/stimuli_generator/relative_clauses.txt"
    save_path = "/scratch2/mrenaudin/colorlessgreenRNNs/analyze_train_sets/generated_train_sets/test.txt"
    starting_freq = 0.04/100
    target_freq = 1/100
    augment_structure_freq(starting_freq=starting_freq,
                           target_freq=target_freq,
                           dataset=dataset,
                           structure_dataset=relative_clauses,
                           save_path=save_path)
    validate_frequency_difference(original_dataset_path=dataset,
                                  generated_dataset_path=save_path,
                                  structure_dataset_path=relative_clauses,
                                  target_frequency=target_freq)
    # show_difference_samples(original_dataset_path=dataset,
    #                       generated_dataset_path=save_path)
