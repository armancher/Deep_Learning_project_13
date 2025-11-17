# train.py
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
import sys, os
sys.path.append(os.path.abspath(".."))

from config import get_experiment_config
from dataloader import create_dataloaders
from model import TransformerLM


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    cfg = get_experiment_config(args.experiment)
    print(f"Running experiment: {cfg.experiment_name}")

    # -------------------------------
    # 1. Data
    # -------------------------------
    train_loader, val_loader = create_dataloaders(cfg)

    # -------------------------------
    # 2. Model
    # -------------------------------
    device = cfg.device
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
        dropout=cfg.dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------
    # 3. Optimizer
    # -------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay
    )

    loss_fn = nn.CrossEntropyLoss()

    # -------------------------------
    # 4. Training loop
    # -------------------------------
    global_step = 0

    for epoch in range(999999):  # effectively infinite, break at cfg.num_train_steps
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            global_step += 1

            # print progress every 100 steps
            if global_step % 100 == 0:
                print(f"Step {global_step} | Train loss: {loss.item():.4f}")

            # run evaluation
            if global_step % cfg.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"[Eval] Step {global_step} | Val loss: {val_loss:.4f}")

            # save checkpoint
            if global_step % cfg.save_interval == 0:
                torch.save(model.state_dict(), f"{cfg.experiment_name}_{global_step}.pt")
                print("Saved checkpoint.")

            # end training
            if global_step >= cfg.num_train_steps:
                print("Finished training!")
                return


if __name__ == "__main__":
    main()
