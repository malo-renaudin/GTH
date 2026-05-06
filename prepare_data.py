import glob
from litgpt.tokenizer import Tokenizer
from litdata import optimize

# 1. Setup Tokenizer
tokenizer = Tokenizer("checkpoints/gpt2")

def parse_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield tokenizer.encode(text)

if __name__ == "__main__":
    files = glob.glob("validation_data/*.txt")

    optimize(
        fn=parse_txt,
        inputs=files,
        output_dir="chunked_data_baseline/val",
        num_workers=4,        # Adjust based on your CPU allocation
        chunk_bytes="64MB"    # Use chunk_bytes for memory-based sizing
    )