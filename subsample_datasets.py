import random
import subprocess
from pathlib import Path

# Define input and output paths for subsampling datasets
DATA_DIR = Path("datasets")
ORC_SRC = DATA_DIR / "orc.txt"
WH_SRC = DATA_DIR / "wh.txt"

ORC_FINAL = DATA_DIR / "orc_final.txt"
WH_FINAL = DATA_DIR / "wh_final.txt"
ORC_TEST = DATA_DIR / "orc_test.txt"
WH_TEST = DATA_DIR / "wh_test.txt"

IO_BUFFER = 8 << 20  # 8 MB read/write buffers


def count_lines(file_path):
    """Count lines via wc -l — native C speed, no file loaded into memory."""
    out = subprocess.check_output(["wc", "-l", str(file_path)])
    return int(out.split()[0])


def stream_split(src_path, test_idx_set, test_path, final_path,
                 total_lines, final_count):
    """
    Single streaming pass over src_path.
    - Lines whose index is in test_idx_set are written to test_path.
    - From the remaining lines, exactly min(final_count, n_available) lines are
      selected via Knuth's Algorithm S and written to final_path.

    Memory usage: O(|test_idx_set|) — the full file is never loaded.
    """
    n_available = total_lines - len(test_idx_set)
    n_select = min(final_count, n_available)
    remaining_to_see = n_available
    remaining_to_pick = n_select

    with (open(src_path,   "r", encoding="utf-8", buffering=IO_BUFFER) as fin,
          open(test_path,  "w", encoding="utf-8", buffering=IO_BUFFER) as ftest,
          open(final_path, "w", encoding="utf-8", buffering=IO_BUFFER) as ffinal):

        for i, line in enumerate(fin):
            if i in test_idx_set:
                ftest.write(line)
            else:
                # Knuth's Algorithm S: include with prob remaining_to_pick / remaining_to_see
                if remaining_to_pick > 0:
                    if random.random() * remaining_to_see < remaining_to_pick:
                        ffinal.write(line)
                        remaining_to_pick -= 1
                    remaining_to_see -= 1


def main():
    random.seed(42)

    # 1. Count lines without loading the files into memory
    print("Counting lines...")
    orc_total = count_lines(ORC_SRC)
    wh_total  = count_lines(WH_SRC)
    print(f"  ORC: {orc_total:,} lines")
    print(f"  WH:  {wh_total:,} lines")

    if orc_total < 1000 or wh_total < 1000:
        raise ValueError("Both files must have at least 1000 lines to extract the test set.")

    # 2. Pick 1000 random line indices for each test set (range() is lazy — O(1) memory)
    orc_test_idx = set(random.sample(range(orc_total), 1000))
    wh_test_idx  = set(random.sample(range(wh_total),  1000))

    target_count = orc_total - 1000
    wh_available = wh_total - 1000
    print(f"\nTarget final line count (ORC minus test): {target_count:,}")
    if wh_available < target_count:
        print(f"Warning: WH only has {wh_available:,} remaining lines; using all of them.")

    # 3. Stream each file once — no full-file load required
    print("\nStreaming ORC (1 pass)...")
    stream_split(ORC_SRC, orc_test_idx, ORC_TEST, ORC_FINAL,
                 orc_total, final_count=target_count)
    print(f"  -> {ORC_TEST.name}: 1,000 lines")
    print(f"  -> {ORC_FINAL.name}: {target_count:,} lines")

    print("\nStreaming WH (1 pass)...")
    stream_split(WH_SRC, wh_test_idx, WH_TEST, WH_FINAL,
                 wh_total, final_count=target_count)
    wh_final_count = min(target_count, wh_available)
    print(f"  -> {WH_TEST.name}: 1,000 lines")
    print(f"  -> {WH_FINAL.name}: {wh_final_count:,} lines")

    print("\nDone!")


if __name__ == "__main__":
    main()