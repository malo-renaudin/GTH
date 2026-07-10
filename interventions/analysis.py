import argparse
import subprocess
from pathlib import Path
import pandas as pd
import tempfile
import os


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Directory containing HF checkpoints"
    )
    p.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True
    )
    p.add_argument(
        "--train-df",
        required=True
    )
    p.add_argument(
        "--eval-df",
        required=True
    )
    p.add_argument(
        "--output-dir",
        required=True
    )
    p.add_argument(
        "--das-script",
        default="das.py"
    )
    p.add_argument(
        "--metric",
        choices=["delta_logp", "odds"],
        default="delta_logp"
    )

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    return p.parse_args()


def run_das(
    das_script,
    train_ckpt,
    eval_ckpt,
    train_df,
    eval_df,
    layers,
    output_dir,
    epochs,
    batch_size,
    lr,
):

    cmd = [
        "python",
        das_script,
        "--train-checkpoint",
        train_ckpt,
        "--eval-checkpoint",
        eval_ckpt,
        "--train-df",
        train_df,
        "--eval-df",
        eval_df,
        "--layers",
        *map(str, layers),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--output-dir",
        output_dir,
    ]

    print("\nRunning:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def get_checkpoints(directory):
    return sorted(
        str(p)
        for p in Path(directory).iterdir()
        if p.is_dir()
    )


def best_result(args, results_csv):
    df = pd.read_csv(results_csv)

    # choose your metric here
    if args.metric == "delta_logp":
        row = df.loc[df["delta_logp_mean"].idxmax()]
    elif args.metric == "odds":
        row = df.loc[df["odds_mean"].idxmax()]
    else:
        raise ValueError(f"Unknown metric: {args.metric}")

    return {
        "layer": row["layer"],
        "position": row["position_label"],
        "delta_logp": row["delta_logp_mean"],
        "odds": row["odds_mean"],
        "train_loss": row["train_loss"],
    }


def main():

    args = parse_args()

    checkpoints = get_checkpoints(args.checkpoint_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------
    # 1. Stability: train = eval
    # ----------------------------------------------------

    stability_results = []

    for ckpt in checkpoints:

        name = Path(ckpt).name
        out = os.path.join(
            args.output_dir,
            f"stability_{name}"
        )

        run_das(
            args.das_script,
            ckpt,
            ckpt,
            args.train_df,
            args.eval_df,
            args.layers,
            out,
            args.epochs,
            args.batch_size,
            args.lr,
        )

        best = best_result(
            args,
            os.path.join(out, "results.csv")
        )

        stability_results.append({
            "checkpoint": name,
            **best
        })

    pd.DataFrame(stability_results).to_csv(
        os.path.join(
            args.output_dir,
            "stability.csv"
        ),
        index=False
    )


    # ----------------------------------------------------
    # 2. Temporal generalization
    # ----------------------------------------------------

    temporal_results = []

    for train_ckpt in checkpoints:

        for eval_ckpt in checkpoints:

            train_name = Path(train_ckpt).name
            eval_name = Path(eval_ckpt).name

            out = os.path.join(
                args.output_dir,
                f"transfer_{train_name}_to_{eval_name}"
            )

            run_das(
                args.das_script,
                train_ckpt,
                eval_ckpt,
                args.train_df,
                args.eval_df,
                args.layers,
                out,
                args.epochs,
                args.batch_size,
                args.lr,
            )

            best = best_result(
                os.path.join(out, "results.csv")
            )

            temporal_results.append({
                "train_checkpoint": train_name,
                "eval_checkpoint": eval_name,
                **best
            })


    pd.DataFrame(temporal_results).to_csv(
        os.path.join(
            args.output_dir,
            "temporal_generalization.csv"
        ),
        index=False
    )


    print("Done.")


if __name__ == "__main__":
    main()