import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join("results", "Shakespeare")
OUT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

def load_metrics():
    data = defaultdict(list)
    pattern = os.path.join(RESULTS_DIR, "metrics_*.json")

    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)

        tok = m.get("tokenizer")
        step = m.get("train_steps")

        if tok is None or step is None:
            print("Skipping", path)
            continue

        entry = {
            "step": step,
            "bits_per_char": m.get("bits_per_char"),
            "loss": m.get("loss"),
            "steps_per_sec": m.get("train_steps_per_sec"),
            "num_params": m.get("num_params"),
        }
        data[tok].append(entry)

    for tok in data:  # sort per tokenizer
        data[tok].sort(key=lambda e: e["step"])

    return data

def plot_metric(data, key, ylabel, filename):
    plt.figure()

    for tok, entries in data.items():
        steps = [e["step"] for e in entries]
        values = [e[key] for e in entries]
        steps, values = zip(*[(s, v) for s, v in zip(steps, values) if v is not None])

        plt.plot(steps, values, marker="o", label=tok)

    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over training steps (Shakespeare)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    out = os.path.join(OUT_DIR, filename)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print("Saved", out)

def main():
    data = load_metrics()
    if not data:
        print("No metrics found.")
        return

    plot_metric(data, "bits_per_char", "Bits per character", "bits_per_char_over_steps.png")
    plot_metric(data, "loss", "Loss", "loss_over_steps.png")
    plot_metric(data, "steps_per_sec", "Training speed (steps/sec)", "steps_per_sec_over_steps.png")

if __name__ == "__main__":
    main()
