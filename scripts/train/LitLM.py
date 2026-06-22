import lightning as L
import torch
import torch.nn.functional as F
from litdata import StreamingDataset
from torch.utils.data import DataLoader

class LitLM(L.LightningModule):
    def __init__(self, model, scheduler=None, lr=1e-3, weight_decay=0.01):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay,
        )

        if self.scheduler is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(#voir modele de warmup par defaut sur litgpt
            optimizer,
            T_max=self.scheduler["t_max"],
            eta_min=self.scheduler["min_lr"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        
    def forward(self, x):
        out = self.model(x)
        return out.logits if hasattr(out, "logits") else out

    def training_step(self, batch, batch_idx):#batch idx not used here 
        x = batch["input_ids"]

        logits = self(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            x.view(-1)
        )

        self.log("train_loss", 
                 loss,
                 prog_bar=True,
                 sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):#batch idx not used here 
        x = batch["input_ids"]

        logits = self(x)

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.95),
            weight_decay=self.weight_decay
        )

class LitDataModule(L.LightningDataModule):
    def __init__(self, path_train, path_val, batch_size):
        super().__init__()
        self.path_train = path_train
        self.path_val = path_val
        self.batch_size = batch_size

    def setup(self, stage=None):#stage not used here
        self.train_ds = StreamingDataset(self.path_train)
        self.val_ds = StreamingDataset(self.path_val)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=4,#-maybe set to -1
            pin_memory=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,#-maybe set to -1
        )