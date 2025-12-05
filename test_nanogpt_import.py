from models.nanogpt_wrapper import create_nanogpt_model
import torch

def main():
    vocab_size = 256
    model, config = create_nanogpt_model(vocab_size=vocab_size)

    print("Config:", config)
    x = torch.randint(0, vocab_size, (2, 16))  # (batch, seq)
    logits, loss = model(x, x)  # dummy forward
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

if __name__ == "__main__":
    main()