import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # causal mask for autoregressive LM
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        )

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)

        x = self.embed(x) + self.pos_emb(positions)

        # apply causal mask
        x = self.transformer(x, mask=self.mask[:T, :T])

        logits = self.lm_head(x)
        return logits
