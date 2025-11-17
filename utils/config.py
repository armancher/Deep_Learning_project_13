# config.py
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    experiment_name: str
    tokenizer_type: str
    vocab_size: int
    max_seq_len: int

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    num_train_steps: int = 3000
    eval_interval: int = 500
    save_interval: int = 1000

    train_ids_path: str = ""
    val_ids_path: str = ""
    test_ids_path: str = ""

    device: str = "cuda"

EXPERIMENTS = {
    "byte_bpe": ExperimentConfig(
        "byte_bpe", "byte_bpe", 1000, 384,
        train_ids_path="../data/byte_bpe_train.npy",
        val_ids_path="../data/byte_bpe_val.npy",
        test_ids_path="../data/byte_bpe_test.npy",
    ),
    "bpe": ExperimentConfig(
        "bpe", "bpe", 8000, 256,
        train_ids_path="../data/bpe_train.npy",
        val_ids_path="../data/bpe_val.npy",
        test_ids_path="../data/bpe_test.npy",
    ),
    "char": ExperimentConfig(
        "char", "char", 0, 512,
        train_ids_path="../data/char_train.npy",
        val_ids_path="../data/char_val.npy",
        test_ids_path="../data/char_test.npy",
    ),
    "byte": ExperimentConfig(
        "byte", "byte", 256, 768,
        train_ids_path="../data/byte_train.npy",
        val_ids_path="../ata/byte_val.npy",
        test_ids_path="../data/byte_test.npy",
    ),
}

def get_experiment_config(name):
    return EXPERIMENTS[name]
