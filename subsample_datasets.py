import random
from pathlib import Path

# Define input and output paths
DATA_DIR = Path("datasets")
ORC_SRC = DATA_DIR / "orc.txt"
WH_SRC = DATA_DIR / "wh.txt"

ORC_FINAL = DATA_DIR / "orc_final.txt"
WH_FINAL = DATA_DIR / "wh_final.txt"
ORC_TEST = DATA_DIR / "orc_test.txt"
WH_TEST = DATA_DIR / "wh_test.txt"

def load_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Using rstrip('\n') keeps the text clean but preserves empty lines if any exist
        return [line.rstrip('\n') for line in f]

def save_lines(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def main():
    # 1. Load the original datasets
    print("Loading files...")
    orc_lines = load_lines(ORC_SRC)
    wh_lines = load_lines(WH_SRC)
    
    print(f"Original Orc size: {len(orc_lines)} lines")
    print(f"Original WH size: {len(wh_lines)} lines")
    
    if len(orc_lines) < 1000 or len(wh_lines) < 1000:
        raise ValueError("Both files must have at least 1000 lines to extract the test set.")

    # 2. Shuffle to ensure random splitting
    # (We shuffle copies to avoid modifying index-based logic if needed later)
    random.seed(42)  # Set seed for reproducibility; remove if you want true randomness
    random.shuffle(orc_lines)
    random.shuffle(wh_lines)
    
    # 3. Extract 1000 sentences for the test sets
    orc_test = orc_lines[:1000]
    orc_train_pool = orc_lines[1000:]
    
    wh_test = wh_lines[:1000]
    wh_train_pool = wh_lines[1000:]
    
    # 4. Downsample WH to match the remaining Orc lines
    target_count = len(orc_train_pool)
    print(f"\nTarget final line count (Orc minus test): {target_count}")
    
    if len(wh_train_pool) < target_count:
        print("Warning: wh5.txt has fewer remaining lines than orc7.txt. Taking all remaining WH lines.")
        wh_final = wh_train_pool
    else:
        # Take a random subset matching the target count
        wh_final = random.sample(wh_train_pool, target_count)
        
    orc_final = orc_train_pool

    # 5. Save everything to the new files
    print("\nSaving new files...")
    save_lines(ORC_TEST, orc_test)
    save_lines(WH_TEST, wh_test)
    save_lines(ORC_FINAL, orc_final)
    save_lines(WH_FINAL, wh_final)
    
    print("Done! Summary of generated files:")
    print(f" -> {ORC_TEST.name}: {len(orc_test)} lines")
    print(f" -> {WH_TEST.name}: {len(wh_test)} lines")
    print(f" -> {ORC_FINAL.name}: {len(orc_final)} lines")
    print(f" -> {WH_FINAL.name}: {len(wh_final)} lines")

if __name__ == "__main__":
    main()