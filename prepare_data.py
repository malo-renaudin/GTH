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

# This block is CRITICAL for multiprocessing
if __name__ == "__main__":
    # 2. Optimize and Chunk
    files = glob.glob("train_data/baseline/*.txt")

    optimize(
        fn=parse_txt,
        inputs=files,
        output_dir="data/optimized_baseline",
        chunk_size="64MB",
    )