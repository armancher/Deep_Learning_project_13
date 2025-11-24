import os
import json
import argparse
import matplotlib.pyplot as plt

def safe_get(d, key, default=None):
    return d[key] if key in d and d[key] is not None else default

def load_metrics(files):
    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)

            rec["bits_per_char"] = safe_get(rec, "bits_per_char", float("nan"))
            rec["train_steps_per_sec"] = safe_get(rec, "train_steps_per_sec", float("nan"))
            rec["num_params"] = safe_get(rec, "num_params", 0)
            rec["tokenizer"] = safe_get(rec, "tokenizer", "unknown")
            rec["checkpoint"] = safe_get(rec, "checkpoint", path)

            records.append(rec)
    return records

def print_table(records):
    print("====================================")
    print(" Tokenizer Comparison Table")
    print("====================================")
    header = ["Tokenizer", "Checkpoint", "Bits/char", "Train steps/sec", "Params (M)"]
    print("{:<10}  {:<25}  {:>10}  {:>15}  {:>10}".format(*header))

    for r in records:
        pm = r["num_params"] / 1e6 if r["num_params"] else 0.0
        print("{:<10}  {:<25}  {:>10.4f}  {:>15.3f}  {:>10.3f}".format(
            r["tokenizer"], os.path.basename(r["checkpoint"])[:25],
            r["bits_per_char"], r["train_steps_per_sec"], pm
        ))
    print("====================================")

def make_bar_plot(records, metric_key, ylabel, outfile):
    labels = [r["tokenizer"] for r in records]
    values = [r[metric_key] for r in records]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.xlabel("Tokenizer")
    plt.title(ylabel)
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Saved plot → {outfile}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_files", nargs="+")
    parser.add_argument("--out_dir", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_metrics(args.metrics_files)

    print_table(records)
    make_bar_plot(records, "bits_per_char", "Bits per character",
                  os.path.join(args.out_dir, "bits_per_char.png"))
    make_bar_plot(records, "train_steps_per_sec", "Training steps/sec",
                  os.path.join(args.out_dir, "steps_per_sec.png"))

    pm_vals = [{"tokenizer": r["tokenizer"], "size_m": r["num_params"]/1e6} for r in records]
    make_bar_plot(pm_vals, "size_m", "Model size (M parameters)",
                  os.path.join(args.out_dir, "model_size_mparams.png"))

if __name__ == "__main__":
    main()