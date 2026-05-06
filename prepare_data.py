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
                # Tokenize the line and yield the input_ids
                yield tokenizer.encode(text)

# 2. Optimize and Chunk
# This will find all .txt files in your folder and create optimized chunks
files = glob.glob("train_data/baseline/*.txt")

optimize(
    fn=parse_txt,
    inputs=files,
    output_dir="data/optimized_baseline",
    chunk_size="64MB", # This prevents the 459MB warning you saw!
)