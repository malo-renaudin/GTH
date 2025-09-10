import random

dataset = "/scratch2/mrenaudin/GTH/english_data/test.txt"
relative_clauses = "/scratch2/mrenaudin/GTH/stimuli_generator/relative_clauses.txt"
questions = "/scratch2/mrenaudin/GTH/stimuli_generator/questions.txt"

# relative clause starting frequency extracted from https://pmc.ncbi.nlm.nih.gov/articles/PMC2722756/pdf/nihms30776.pdf
starting_frequencies = {'relative_clauses': 0.04/100, 'questions': 0.25/100}
#start freq relative clause : 0.04, and start freq question : 0.25
frequencies = {'relative_clauses': [0.08/100, 0.16/100, 0.32/100], 'questions': [0.5/100, 1/100, 2/100]}


def augment_structure_freq(starting_freq: float,
                           target_freq: float,
                           dataset: str,
                           structure_dataset: str,
                           save_path: str):
    """
    With N sentences, and x % of sentences with a given syntactic structure, we get xN sentences with this structure, 
    and (1-x)N sentences without. Our goal is to find k, the number of sentences to randomly remove, and to add with this structure,
    such that the final % of sentences with this structure is a, the target frequency. 

    1. After removal : N-k sentences, x(N-k) with structure, (1-x)(N-k) without
    2. When adding k sentences with structure : N sentences, x(N-k)+k with structure, (1-x)(N-k) without
    3. Finally, we want k such that (x(N-k)+k)/N = a, which gives k = (a-x)N/(1-x).
    """
    # Read the datasets
    with open(dataset, 'r', encoding='utf-8') as f:
        main_sentences = [line.strip() for line in f if line.strip()]

    with open(structure_dataset, 'r', encoding='utf-8') as f:
        structure_sentences = [line.strip() for line in f if line.strip()]

    # Calculate k using the corrected formula
    N = len(main_sentences)
    k = int(round((target_freq - starting_freq) * N / (1 - starting_freq)))
    
    print(f"Dataset size: {N}")
    print(f"Starting frequency: {starting_freq:.6f} ({starting_freq*100:.4f}%)")
    print(f"Target frequency: {target_freq:.6f} ({target_freq*100:.4f}%)")
    print(f"Calculated k: {k}")
    
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

    print(f"Removed {k} sentences, added {k} structure sentences")
    print(f"Final dataset size: {len(final_dataset)}")
    print(f"Expected structure frequency: {target_freq:.6f} ({target_freq*100:.4f}%)")

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            for sentence in final_dataset:
                f.write(sentence + '\n')
        print(f"Dataset saved to: {save_path}")

    return final_dataset


def validate_structure_frequency(
    generated_dataset_path: str,
    structure_dataset_path: str,
    target_frequency: float,
    original_dataset_path: str = dataset  # default to base dataset
):
    """
    Validate that the actual frequency of structure sentences in the generated dataset
    matches the target frequency. Also compare with the original dataset and report
    the number of sentences removed/added for this structure.
    """
    # Read generated dataset
    with open(generated_dataset_path, 'r', encoding='utf-8') as f:
        generated_sentences = [line.strip() for line in f if line.strip()]

    # Read structure sentences
    with open(structure_dataset_path, 'r', encoding='utf-8') as f:
        structure_sentences = set(line.strip() for line in f if line.strip())

    # Read original dataset (default: global dataset variable)
    original_sentences = []
    if original_dataset_path:
        with open(original_dataset_path, 'r', encoding='utf-8') as f:
            original_sentences = [line.strip() for line in f if line.strip()]

    # Count structure sentences in generated and original
    structure_count_generated = sum(1 for s in generated_sentences if s in structure_sentences)
    structure_count_original = sum(1 for s in original_sentences if s in structure_sentences) if original_sentences else None

    total_generated = len(generated_sentences)
    total_original = len(original_sentences) if original_sentences else None

    actual_frequency = structure_count_generated / total_generated if total_generated > 0 else 0
    actual_percentage = actual_frequency * 100
    target_percentage = target_frequency * 100

    # Compute added/removed structures if original is available
    added_structures = removed_structures = None
    if original_sentences:
        original_set = set(original_sentences)
        generated_set = set(generated_sentences)

        added_structures = len([s for s in generated_set - original_set if s in structure_sentences])
        removed_structures = len([s for s in original_set - generated_set if s in structure_sentences])

    # Exact match check (no tolerance)
    frequency_matches = structure_count_generated == int(target_frequency * total_generated)

    # Print results
    print("=== STRUCTURE FREQUENCY VALIDATION ===")
    print(f"Generated dataset size: {total_generated}")
    print(f"Structure sentences in generated: {structure_count_generated}")
    if total_original is not None:
        print(f"Original dataset size: {total_original}")
        print(f"Structure sentences in original: {structure_count_original}")
        print(f"Removed structure sentences: {removed_structures}")
        print(f"Added structure sentences: {added_structures}")

    print(f"Target frequency: {target_percentage:.4f}%")
    print(f"Actual frequency: {actual_percentage:.4f}%")
    print(f"Frequency exact match: {frequency_matches}")
    print()

    validation_results = {
        'dataset_info': {
            'total_sentences': total_generated,
            'structure_sentences_generated': structure_count_generated,
            'total_sentences_original': total_original,
            'structure_sentences_original': structure_count_original,
            'removed_structures': removed_structures,
            'added_structures': added_structures,
            'actual_frequency': actual_frequency,
            'actual_percentage': actual_percentage,
            'target_frequency': target_frequency,
            'target_percentage': target_percentage,
            'structure_sentences': structure_count_generated
        },
        'validation': {
            'percentage_difference': abs(actual_percentage - target_percentage),
            'frequency_matches': frequency_matches
        }
    }

    return validation_results




def generate_all_training_sets():
    """Generate all 9 combinations of relative clause and question frequencies"""
    import os
    from itertools import product
    from datetime import datetime
    
    # Output directory for generated datasets
    output_dir = "/scratch2/mrenaudin/GTH/generated_train_sets/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare validation results log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_log_path = os.path.join(output_dir, f"validation_results.txt")
    
    # Convert frequencies to percentage strings for filenames
    def freq_to_string(freq):
        return f"{freq*100:.2f}".replace('.', '_')
    
    print("Generating all 9 training set combinations...")
    print("=" * 60)
    
    successful_generations = 0
    failed_generations = 0
    all_validation_results = []
    
    # Start writing validation log
    with open(validation_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("TRAINING SET GENERATION AND VALIDATION REPORT\n")
        log_file.write("=" * 60 + "\n")
        log_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Base dataset: {dataset}\n")
        log_file.write(f"RC structure dataset: {relative_clauses}\n")
        log_file.write(f"Question structure dataset: {questions}\n")
        log_file.write(f"Starting frequencies - RC: {starting_frequencies['relative_clauses']:.6f}, Q: {starting_frequencies['questions']:.6f}\n")
        log_file.write("\n" + "=" * 60 + "\n\n")
        
        for rc_freq, q_freq in product(frequencies['relative_clauses'], frequencies['questions']):
            # Create filename
            rc_freq_str = freq_to_string(rc_freq)
            q_freq_str = freq_to_string(q_freq)
            filename = f"train_rc_{rc_freq_str}_q_{q_freq_str}.txt"
            save_path = os.path.join(output_dir, filename)
            
            print(f"\n🔄 Generating: {filename}")
            print(f"   RC frequency: {rc_freq:.6f} ({rc_freq*100:.4f}%)")
            print(f"   Q frequency: {q_freq:.6f} ({q_freq*100:.4f}%)")
            
            # Write to log file
            log_file.write(f"DATASET: {filename}\n")
            log_file.write("-" * 40 + "\n")
            log_file.write(f"Target RC frequency: {rc_freq:.6f} ({rc_freq*100:.4f}%)\n")
            log_file.write(f"Target Q frequency: {q_freq:.6f} ({q_freq*100:.4f}%)\n")
            log_file.write(f"Generation time: {datetime.now().strftime('%H:%M:%S')}\n\n")
            
            try:
                # Start with original dataset
                current_dataset = dataset
                
                # Step 1: Adjust relative clause frequency
                temp_rc_path = os.path.join(output_dir, f"temp_rc_{rc_freq_str}.txt")
                print(f"   Step 1: RC {starting_frequencies['relative_clauses']:.6f} → {rc_freq:.6f}")
                
                augment_structure_freq(
                    starting_freq=starting_frequencies['relative_clauses'],
                    target_freq=rc_freq,
                    dataset=current_dataset,
                    structure_dataset=relative_clauses,
                    save_path=temp_rc_path
                )
                current_dataset = temp_rc_path
                
                print(f"   Step 2: Q {starting_frequencies['questions']:.6f} → {q_freq:.6f}")
                
                # Step 2: Adjust question frequency on the RC-adjusted dataset
                augment_structure_freq(
                    starting_freq=starting_frequencies['questions'],
                    target_freq=q_freq,
                    dataset=current_dataset,
                    structure_dataset=questions,
                    save_path=save_path
                )
                
                # Clean up temporary file
                if os.path.exists(temp_rc_path):
                    os.remove(temp_rc_path)
                
                # Detailed validation
                print(f"   Validating final frequencies...")
                log_file.write("VALIDATION RESULTS:\n")
                
                rc_validation = validate_structure_frequency(save_path, relative_clauses, rc_freq)
                q_validation = validate_structure_frequency(save_path, questions, q_freq)
                
                # Write detailed validation to log
                log_file.write(f"Relative Clauses:\n")
                log_file.write(f"  - Target: {rc_freq*100:.4f}%\n")
                log_file.write(f"  - Actual: {rc_validation['dataset_info']['actual_percentage']:.4f}%\n")
                log_file.write(f"  - Difference: {rc_validation['validation']['percentage_difference']:.4f}%\n")
                log_file.write(f"  - Structure sentences found: {rc_validation['dataset_info']['structure_sentences']}\n")
                
                log_file.write(f"Questions:\n")
                log_file.write(f"  - Target: {q_freq*100:.4f}%\n")
                log_file.write(f"  - Actual: {q_validation['dataset_info']['actual_percentage']:.4f}%\n")
                log_file.write(f"  - Difference: {q_validation['validation']['percentage_difference']:.4f}%\n")
                log_file.write(f"  - Structure sentences found: {q_validation['dataset_info']['structure_sentences']}\n")
                
                log_file.write(f"Dataset size: {rc_validation['dataset_info']['total_sentences']}\n")
                
                rc_passed = rc_validation['validation']['frequency_matches']
                q_passed = q_validation['validation']['frequency_matches']
                
                if rc_passed and q_passed:
                    print(f"   ✅ SUCCESS: {filename}")
                    log_file.write("STATUS: ✅ SUCCESS - All validations passed\n")
                    successful_generations += 1
                else:
                    print(f"   ⚠️  WARNING: {filename} - validation issues")
                    log_file.write("STATUS: ⚠️  WARNING - Some validations failed\n")
                    if not rc_passed:
                        print(f"      RC frequency off by {rc_validation['validation']['percentage_difference']:.4f}%")
                        log_file.write(f"  - RC frequency tolerance exceeded\n")
                    if not q_passed:
                        print(f"      Q frequency off by {q_validation['validation']['percentage_difference']:.4f}%")
                        log_file.write(f"  - Q frequency tolerance exceeded\n")
                    successful_generations += 1  # Still count as success if files were generated
                
                # Store results for summary
                all_validation_results.append({
                    'filename': filename,
                    'rc_target': rc_freq*100,
                    'rc_actual': rc_validation['dataset_info']['actual_percentage'],
                    'rc_passed': rc_passed,
                    'q_target': q_freq*100,
                    'q_actual': q_validation['dataset_info']['actual_percentage'],  
                    'q_passed': q_passed,
                    'dataset_size': rc_validation['dataset_info']['total_sentences'],
                    'success': rc_passed and q_passed
                })
                
            except Exception as e:
                print(f"   ❌ ERROR generating {filename}: {str(e)}")
                log_file.write(f"STATUS: ❌ ERROR - {str(e)}\n")
                failed_generations += 1
                # Clean up temp file if it exists
                temp_rc_path = os.path.join(output_dir, f"temp_rc_{rc_freq_str}.txt")
                if os.path.exists(temp_rc_path):
                    os.remove(temp_rc_path)
            
            log_file.write("\n" + "=" * 60 + "\n\n")
            log_file.flush()  # Ensure immediate write to file
    
        # Write summary to log file
        log_file.write("GENERATION SUMMARY\n")
        log_file.write("=" * 60 + "\n")
        log_file.write(f"Successful generations: {successful_generations}/9\n")
        log_file.write(f"Failed generations: {failed_generations}/9\n")
        log_file.write(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if all_validation_results:
            log_file.write("DETAILED SUMMARY TABLE\n")
            log_file.write("-" * 60 + "\n")
            log_file.write(f"{'Filename':<25} {'RC Target':<10} {'RC Actual':<10} {'Q Target':<9} {'Q Actual':<9} {'Size':<8} {'Status'}\n")
            log_file.write("-" * 100 + "\n")
            
            for result in all_validation_results:
                status = "✅ PASS" if result['success'] else "⚠️  WARN"
                log_file.write(f"{result['filename']:<25} {result['rc_target']:<10.4f} {result['rc_actual']:<10.4f} "
                             f"{result['q_target']:<9.4f} {result['q_actual']:<9.4f} {result['dataset_size']:<8} {status}\n")
    
    print("\n" + "=" * 60)
    print("🏁 GENERATION COMPLETE!")
    print(f"✅ Successful: {successful_generations}/9")
    print(f"❌ Failed: {failed_generations}/9")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Validation log: {validation_log_path}")
    
    # List all generated files
    if successful_generations > 0:
        print(f"\n📋 Generated files:")
        generated_files = []
        for filename in sorted(os.listdir(output_dir)):
            if filename.startswith("train_rc_") and filename.endswith(".txt"):
                generated_files.append(filename)
                file_path = os.path.join(output_dir, filename)
                file_size = os.path.getsize(file_path)
                print(f"   - {filename} ({file_size:,} bytes)")
        
        print(f"\nTotal files generated: {len(generated_files)}")
        print(f"\n💾 All validation results saved to: {validation_log_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Generate all 9 training sets
    generate_all_training_sets()