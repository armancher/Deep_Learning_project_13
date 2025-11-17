# dataloader.py
from torch.utils.data import DataLoader
from dataset import TokenDataset

def create_dataloaders(cfg):
    train_ds = TokenDataset(cfg.train_ids_path, cfg.max_seq_len)
    val_ds   = TokenDataset(cfg.val_ids_path,   cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, val_loader
