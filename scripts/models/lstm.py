import torch
import torch.nn as nn

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        self.lstm = nn.LSTM(
            d_model,
            d_model,
            n_layers,
            batch_first=True
        )

        self.lm_head = nn.Linear(d_model, vocab_size)# wieghts should be tied to embedding layer

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.lm_head(x)