import subprocess
import pandas as pd
from pathlib import Path
import tempfile


CHECKPOINT_DIR = "/path/to/checkpoints"
ORC_TEST = "/path/to/orc_test.txt"
WH_TEST = "/path/to/wh_test.txt"

OUTPUT_CSV = "all_checkpoint_results.csv"

EVAL_SCRIPT = Path(__file__).with_name("eval_probability_masses.py")


def run_checkpoint(checkpoint):
    print(f"\n===== Evaluating {checkpoint} =====")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_out = tmp.name

    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--checkpoint",
        checkpoint,
        "--orc_test",
        ORC_TEST,
        "--wh_test",
        WH_TEST,
        "--out",
        tmp_out,
        "--cand_batch_size",
        "64",
        "--subset_size",
        "1",
        "--subset_repeats",
        "3",
        "--amp",
    ]

    subprocess.run(cmd, check=True)

    df = pd.read_csv(tmp_out)

    # aggregate like the evaluator does: mean across sentences, while each
    # sentence was based on 3 random continuation samples
    result = {
        "checkpoint": checkpoint,
        "orc_np": df[df.file == ORC_TEST]["np_mass"].mean(),
        "orc_vp": df[df.file == ORC_TEST]["vp_mass"].mean(),
        "orc_q": df[df.file == ORC_TEST]["q_mass"].mean(),
        "wh_np": df[df.file == WH_TEST]["np_mass"].mean(),
        "wh_vp": df[df.file == WH_TEST]["vp_mass"].mean(),
        "wh_q": df[df.file == WH_TEST]["q_mass"].mean(),
    }

    return result

def main():

    checkpoints = sorted(
        str(p)
        for p in Path(CHECKPOINT_DIR).iterdir()
        if p.is_dir()
    )

    results = []

    for ckpt in checkpoints:
        results.append(run_checkpoint(ckpt))

        pd.DataFrame(results).to_csv(
            OUTPUT_CSV,
            index=False
        )

    print("Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()