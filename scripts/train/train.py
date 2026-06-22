import yaml
import lightning as L
import torch

from models import LSTMLM, build_mamba
from LitLM import LitLM,LitDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

cfg = yaml.safe_load(open("config.yaml"))

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32


# --------------------
# model selection
# --------------------
if cfg["model"] == "lstm":
    base_model = LSTMLM(
        cfg["vocab_size"],
        cfg["d_model"],
        cfg["n_layers"]
    )

elif cfg["model"] == "mamba":
    base_model = build_mamba(cfg, device, dtype)

else:
    raise ValueError("unknown model")


# --------------------
# Lightning wrapper
# --------------------
model = LitLM(base_model, scheduler = cfg["scheduler"], lr=cfg["lr"], weight_decay=cfg["weight_decay"])

# --------------------
# data
# --------------------
data = LitDataModule(cfg["train_data_path"], cfg["val_data_path"], cfg["batch_size"])

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
)


checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)
# --------------------
# trainer
# --------------------
trainer = L.Trainer(
    gradient_clip_val=cfg["max_norm"],
    gradient_clip_algorithm="norm",
    val_check_interval=cfg["val_check_interval"],
    max_steps=cfg["max_steps"],
    precision="bf16-mixed",
    accelerator="gpu",
    devices="auto",
    log_every_n_steps=50,
    callbacks=[early_stopping, checkpoint],
)

trainer.fit(model, data)