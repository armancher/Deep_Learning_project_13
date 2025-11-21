# config/train_wikitext2_gpt4.py
#
# Usage (from nanoGPT folder):
#   cd models/nanogpt
#   MODEL_SIZE=small python train.py config/train_wikitext2_gpt4.py
#
# Make sure you have:
#   models/nanogpt/data/wikitext2_gpt4/train.bin
#   models/nanogpt/data/wikitext2_gpt4/val.bin
# copied from your project-level data/gpt4_tokens/.

import os

# ---------------------------------------------------------
# Run / logging config
# ---------------------------------------------------------

out_dir = "out-wikitext2-gpt4"   # where checkpoints + logs go
eval_interval = 500              # eval every N steps
eval_iters = 50                  # how many batches to average for val
log_interval = 10                # how often to print to console

always_save_checkpoint = False   # only save best + last
init_from = "scratch"            # train from scratch

# optional: turn on if you want to log to wandb later
wandb_log = False
wandb_project = "wikitext2-tokenizers"
wandb_run_name = None            # you can override from CLI

# dataset name: nanoGPT will look for data/{dataset}/train.bin, val.bin
dataset = "wikitext2_gpt4"

# ---------------------------------------------------------
# Optimization / training config
# ---------------------------------------------------------

gradient_accumulation_steps = 1  # for now, keep simple
batch_size = 64                  # tokens per step = batch_size * block_size
block_size = 256                 # context length

# total training length (same across tokenizers for fairness)
max_iters = 20_000               # number of optimization steps
learning_rate = 3e-4
weight_decay = 0.1
beta2 = 0.95                     # AdamW beta2
warmup_iters = 200
lr_decay_iters = max_iters       # cosine decay down to min_lr
min_lr = 3e-5

# dropout (used in attention + MLPs)
dropout = 0.1

# compile model with torch.compile (if available, speeds up training)
compile = True

# ---------------------------------------------------------
# Model size switch (tiny / small)
# ---------------------------------------------------------

# Change this via environment variable:
#   MODEL_SIZE=tiny  python train.py config/train_wikitext2_gpt4.py
#   MODEL_SIZE=small python train.py config/train_wikitext2_gpt4.py
MODEL_SIZE = os.environ.get("MODEL_SIZE", "small")

if MODEL_SIZE == "tiny":
    n_layer = 4
    n_head = 4
    n_embd = 256
elif MODEL_SIZE == "small":
    n_layer = 8
    n_head = 8
    n_embd = 512
else:
    raise ValueError(f"Unknown MODEL_SIZE={MODEL_SIZE!r}, use 'tiny' or 'small'.")

bias = True  # use bias in LayerNorm/Linear (nanoGPT default)

# ---------------------------------------------------------
# Vocab size
# ---------------------------------------------------------
# nanoGPT usually reads vocab_size from data/{dataset}/meta.pkl if present.
# For GPT-4 tokenizer, there is no meta.pkl yet, so we set it explicitly.
#
# IMPORTANT:
#   Ideally, set this equal to (max token id in your dataset) + 1.
#   For now you can leave this as a safe upper bound and refine later.

vocab_size = 100_000  # TODO: replace with your actual GPT-4 tokenizer vocab size
