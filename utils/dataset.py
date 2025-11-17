import numpy as np
import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, ids_path, max_seq_len):
        # Load the long 1D array of token IDs
        self.data = np.load(ids_path).astype(np.int64)
        self.max_seq_len = max_seq_len

    def __len__(self):
        # number of possible sequences
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx):
        # slice tokens
        chunk = self.data[idx : idx + self.max_seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
