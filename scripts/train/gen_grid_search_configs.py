import yaml
import itertools
from pathlib import Path

# -----------------------
# BASE CONFIG (fixed parts)
# -----------------------
base_config = {
    "output_dir": "/lustre/fswork/projects/rech/ywa/uds37kc/GTH/results/gpt2_hf",
    "max_grad_norm": 1.0,
    "num_train_epochs": 5,
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 50,
    "early_stopping_patience": 3,
}

# -----------------------
# GRID (what you vary)
# -----------------------
learning_rates = [1e-5, 3e-5, 5e-5]
warmup_ratios = [0.0, 0.05, 0.1]
weight_decays = [0.0, 0.01]
batch_sizes = [8, 16, 32]

# -----------------------
# OUTPUT DIR FOR CONFIGS
# -----------------------
out_dir = Path("configs/hf/grid_search")
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------
# GENERATE COMBINATIONS
# -----------------------
grid = itertools.product(
    learning_rates,
    warmup_ratios,
    weight_decays,
    batch_sizes,
)

for lr, warmup, wd, bs in grid:
    cfg = base_config.copy()

    cfg.update({
        "learning_rate": lr,
        "warmup_ratio": warmup,
        "weight_decay": wd,
        "train_batch_size": bs,
    })

    name = f"lr{lr}_wu{warmup}_wd{wd}_bs{bs}".replace(".", "p")
    path = out_dir / f"{name}.yaml"

    with open(path, "w") as f:
        yaml.dump(cfg, f)

    print("Wrote", path)