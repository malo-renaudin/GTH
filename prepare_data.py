import glob
from litgpt.tokenizer import Tokenizer
from litdata import optimize
from litdata.streaming.item_loader import TokensLoader

tokenizer = Tokenizer("checkpoints/gpt2")

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                # Yield the list of tokens
                yield tokenizer.encode(text)

if __name__ == "__main__":
    # IMPORTANT: Delete the old folder first to avoid metadata contamination
    # rm -rf chunked_data_baseline/train
    
    files = glob.glob("train_data/baseline_2/*.txt")
    
    optimize(
        fn=parse_txt,
        inputs=files,
        output_dir="chunked_data_baseline/train",
        chunk_bytes="64MB",
        # Explicitly set the item_loader for language modeling tokens
        item_loader=TokensLoader(block_size=1024) 
    )