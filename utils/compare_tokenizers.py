import os
import json
import argparse
import matplotlib.pyplot as plt


def load_metrics(files):
    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
            records.append(rec)
    return records


def print_table(records):
    print("====================================")
    print(" Tokenizer Comparison Table")
    print("====================================")
    header = [
        "Tokenizer",
        "Checkpoint",
        "Bits/char",
        "Train steps/sec",
        "Params (M)",
    ]
    print("{:<10}  {:<25}  {:>10}  {:>15}  {:>10}".format(*header))

    for r in records:
        tok = r["tokenizer"]
        ckpt = os.path.basename(r["checkpoint"])
        bpc = r["bits_per_char"]
        sps = r["train_steps_per_sec"] if r["train_steps_per_sec"] is not None else float("nan")
        params_m = (r["num_params"] or 0) / 1e6 if r["num_params"] is not None else 0.0

        print("{:<10}  {:<25}  {:>10.4f}  {:>15.3f}  {:>10.3f}".format(
            tok, ckpt[:25], bpc, sps, params_m
        ))
    print("====================================\n")


def make_bar_plot(records, metric_key, ylabel, out_path):
    labels = [r["tokenizer"] for r in records]
    values = [r[metric_key] for r in records]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.xlabel("Tokenizer")
    plt.title(ylabel + " by Tokenizer")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot → {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_files", nargs="+",
                        help="List of metrics JSON files produced by eval_nanogpt_mod.py")
    parser.add_argument("--out_dir", default="results/plots",
                        help="Directory to save plots")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_metrics(args.metrics_files)

    # 1) Table
    print_table(records)

    # 2) Graphs
    make_bar_plot(records, "bits_per_char", "Bits per character",
                  os.path.join(args.out_dir, "bits_per_char.png"))

    if any(r["train_steps_per_sec"] is not None for r in records):
        make_bar_plot(records, "train_steps_per_sec", "Training steps/sec",
                      os.path.join(args.out_dir, "steps_per_sec.png"))

    # Model size in millions
    params_m = []
    for r in records:
        params_m.append((r["num_params"] or 0) / 1e6 if r["num_params"] is not None else 0.0)

    # temporarily attach values
    for r, pm in zip(records, params_m):
        r["_params_m"] = pm

    make_bar_plot(records, "_params_m", "Model size (M parameters)",
                  os.path.join(args.out_dir, "model_size_mparams.png"))

    # cleanup
    for r in records:
        if "_params_m" in r:
            del r["_params_m"]


if __name__ == "__main__":
    main()