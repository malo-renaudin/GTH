import lightning as L
from litdata import StreamingDataset, StreamingDataLoader
from transformers import AutoTokenizer


class LMDataModule(L.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, model_name="gpt2", seq_len=512, batch_size=8):
        super().__init__()

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        self.train_ds = StreamingDataset(self.train_dir)
        self.val_ds = StreamingDataset(self.val_dir)
        self.test_ds = StreamingDataset(self.test_dir)

    def collate_fn(self, batch):
        texts = [x["text"] for x in batch]

        tokens = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    def train_dataloader(self):
        return StreamingDataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return StreamingDataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return StreamingDataLoader(self.test_ds, batch_size=self.batch_size, collate_fn=self.collate_fn)