import argparse
import subprocess
import pandas as pd
from pathlib import Path
import tempfile


EVAL_SCRIPT = Path(__file__).with_name("eval_probability_masses.py")


def run_checkpoint(checkpoint, orc_test, wh_test, cand_batch_size, subset_size, subset_repeats, amp):
    print(f"\n===== Evaluating {checkpoint} =====")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_out = tmp.name

    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--checkpoint",
        checkpoint,
        "--orc_test",
        orc_test,
        "--wh_test",
        wh_test,
        "--out",
        tmp_out,
        "--cand_batch_size",
        str(cand_batch_size),
        "--subset_size",
        str(subset_size),
        "--subset_repeats",
        str(subset_repeats),
    ]

    if amp:
        cmd.append("--amp")

    subprocess.run(cmd, check=True)

    df = pd.read_csv(tmp_out)

    result = {
        "checkpoint": checkpoint,
        "orc_np": df[df.file == orc_test]["np_mass"].mean(),
        "orc_vp": df[df.file == wh_test]["vp_mass"].mean(),
        "orc_q": df[df.file == orc_test]["q_mass"].mean(),
        "wh_np": df[df.file == wh_test]["np_mass"].mean(),
        "wh_vp": df[df.file == wh_test]["vp_mass"].mean(),
        "wh_q": df[df.file == wh_test]["q_mass"].mean(),
    }

    return result


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate probability masses over multiple Hugging Face checkpoints."
    )

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing Hugging Face checkpoints"
    )
    parser.add_argument(
        "--orc_test",
        required=True,
        help="ORC test file"
    )
    parser.add_argument(
        "--wh_test",
        required=True,
        help="WH test file"
    )
    parser.add_argument(
        "--output_csv",
        default="all_checkpoint_results.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--cand_batch_size",
        type=int,
        default=64,
        help="Candidate batch size"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=1,
        help="Number of sampled candidates per repeat"
    )
    parser.add_argument(
        "--subset_repeats",
        type=int,
        default=3,
        help="Number of random subsets"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision"
    )

    args = parser.parse_args()

    checkpoints = sorted(
        str(p)
        for p in Path(args.checkpoint_dir).iterdir()
        if p.is_dir()
    )

    results = []

    for ckpt in checkpoints:
        results.append(
            run_checkpoint(
                ckpt,
                args.orc_test,
                args.wh_test,
                args.cand_batch_size,
                args.subset_size,
                args.subset_repeats,
                args.amp,
            )
        )

        # Save progressively in case of interruption
        pd.DataFrame(results).to_csv(
            args.output_csv,
            index=False
        )

    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()