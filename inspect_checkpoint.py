import torch
import os

# Change this if you want byt5/char/bpe, but keep the relative path:
ckpt_path = os.path.join(
    "checkpoints", "Shakespeare", "nanogpt_byt5_step1000.pt"
)

ckpt = torch.load(ckpt_path, map_location="cpu")

print("Keys in checkpoint dict:")
print(ckpt.keys())
print()

print("Tokenizer name:", ckpt.get("tokenizer_name"))
print("Vocab size arg:", ckpt.get("vocab_size_arg"))
print("Num params:", ckpt.get("num_params"))
print("Train steps:", ckpt.get("train_steps"))
print("Train time (sec):", ckpt.get("train_time_sec"))
print("Train steps/sec:", ckpt.get("train_steps_per_sec"))
print("Dataset path:", ckpt.get("dataset_path"))
print("Config:", ckpt.get("config"))
