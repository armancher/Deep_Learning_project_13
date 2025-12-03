import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def safe_get(d, key, default=None):
    return d[key] if key in d and d[key] is not None else default

def load_metrics(files):
    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)

            # FIX: accept both names
            rec["bits_per_char"] = safe_get(
                rec,
                "bits_per_char",
                safe_get(rec, "bits_per_character", float("nan"))
            )

            rec["train_steps_per_sec"] = safe_get(rec, "train_steps_per_sec", float("nan"))
            rec["num_params"] = safe_get(rec, "num_params", 0)
            rec["tokenizer"] = safe_get(rec, "tokenizer", "unknown")
            rec["checkpoint"] = safe_get(rec, "checkpoint", path)

            rec["bits_per_byte"] = safe_get(rec, "bits_per_byte", float("nan"))
            rec["compression_ratio_chars"] = safe_get(rec, "compression_ratio_chars", float("nan"))
            rec["perplexity"] = safe_get(rec, "perplexity", float("nan"))
            rec["loss"] = safe_get(rec, "loss", float("nan"))

            tokenization_eff = safe_get(rec, "tokenization_efficiency", {})
            rec["tokens_per_char"] = safe_get(tokenization_eff, "tokens_per_char", float("nan"))
            rec["chars_per_token"] = safe_get(tokenization_eff, "chars_per_token", float("nan"))

            records.append(rec)
    return records

def print_table(records):
    print("="*70)
    print(" TOKENIZER COMPARISON TABLE")
    print("="*70)

    headers = ["Tokenizer", "Bits/char", "Bits/byte", "Compression", "Perplexity", "Train spd", "Params(M)"]
    print("{:<12} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}".format(*headers))
    print("-" * 80)

    for r in records:
        pm = r["num_params"] / 1e6 if r["num_params"] else 0.0
        print("{:<12} {:>10.4f} {:>10.4f} {:>12.2f} {:>12.2f} {:>10.2f} {:>10.2f}".format(
            r["tokenizer"],
            r["bits_per_char"],
            r["bits_per_byte"],
            r["compression_ratio_chars"],
            r["perplexity"],
            r["train_steps_per_sec"],
            pm
        ))
    print("="*70)

def print_detailed_comparison(records):
    print("\n" + "="*70)
    print(" DETAILED ANALYSIS")
    print("="*70)

    # Protect min/max from NaN
    def safe_min(vals):
        return min([v for v in vals if not np.isnan(v)])

    def safe_max(vals):
        return max([v for v in vals if not np.isnan(v)])

    best_bpc = safe_min([r["bits_per_char"] for r in records])
    best_bpb = safe_min([r["bits_per_byte"] for r in records])
    best_ppl = safe_min([r["perplexity"] for r in records])
    best_comp = safe_max([r["compression_ratio_chars"] for r in records])
    best_speed = safe_max([r["train_steps_per_sec"] for r in records])

    for r in records:
        print(f"\n--- {r['tokenizer'].upper()} TOKENIZER ---")
        print(f"  Efficiency:  {r['bits_per_char']:.4f} {'(best)' if r['bits_per_char']==best_bpc else ''}")
        print(f"  Byte eff:    {r['bits_per_byte']:.4f} {'(best)' if r['bits_per_byte']==best_bpb else ''}")
        print(f"  Compression: {r['compression_ratio_chars']:.2f} {'(best)' if r['compression_ratio_chars']==best_comp else ''}")
        print(f"  Quality:     {r['perplexity']:.2f} {'(best)' if r['perplexity']==best_ppl else ''}")
        print(f"  Speed:       {r['train_steps_per_sec']:.2f} {'(best)' if r['train_steps_per_sec']==best_speed else ''}")
        print(f"  Tokenization: {r['tokens_per_char']:.3f} tokens/char")

def make_bar_plot(records, metric_key, ylabel, outfile, higher_better=False):
    labels = [r["tokenizer"] for r in records]
    raw_values = [r.get(metric_key, float("nan")) for r in records]

    # Replace NaN with 0 for plotting
    values = [0 if (v is None or np.isnan(v)) else v for v in raw_values]

    # Colors
    best_val = max(values) if higher_better else min(values)
    colors = ['lightgreen' if v == best_val else 'lightcoral' for v in values]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.85, edgecolor='black')

    # Labels above bars
    pad = max(values) * 0.05 if max(values) != 0 else 0.1
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + pad,
                 f'{value:.4f}',
                 ha='center', va='bottom', fontweight='bold')

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Tokenizer", fontsize=12)
    plt.title(f"Tokenizer Comparison: {ylabel}", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved plot → {outfile}")
    plt.close()

def create_comprehensive_chart(records, out_dir):
    metrics = [
        ('bits_per_char', 'Bits per Character', False),
        ('bits_per_byte', 'Bits per Byte', False),
        ('compression_ratio_chars', 'Compression Ratio', True),
        ('perplexity', 'Perplexity', False),
        ('train_steps_per_sec', 'Training Speed', True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (metric_key, title, higher_better) in enumerate(metrics):
        labels = [r["tokenizer"] for r in records]
        raw_values = [r.get(metric_key, float("nan")) for r in records]
        values = [0 if (v is None or np.isnan(v)) else v for v in raw_values]

        best_val = max(values) if higher_better else min(values)
        colors = ['lightgreen' if v == best_val else 'lightcoral' for v in values]

        bars = axes[idx].bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_title(title, fontweight='bold')

        pad = max(values) * 0.05 if max(values) != 0 else 0.1
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2,
                           bar.get_height() + pad,
                           f'{value:.4f}',
                           ha='center', va='bottom', fontsize=9)

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = os.path.join(out_dir, "comprehensive_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive chart → {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Enhanced tokenizer comparison tool")
    parser.add_argument("metrics_files", nargs="+", help="JSON metrics files to compare")
    parser.add_argument("--out_dir", default="results/plots", help="Output directory for plots")
    parser.add_argument("--comprehensive", action="store_true", help="Generate comprehensive analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_metrics(args.metrics_files)

    print_table(records)

    if args.comprehensive:
        print_detailed_comparison(records)
        create_comprehensive_chart(records, args.out_dir)

    make_bar_plot(records, "bits_per_char", "Bits per Character (lower better)",
                  os.path.join(args.out_dir, "bits_per_char.png"))

    make_bar_plot(records, "bits_per_byte", "Bits per Byte (lower better)",
                  os.path.join(args.out_dir, "bits_per_byte.png"))

    make_bar_plot(records, "compression_ratio_chars", "Compression Ratio (higher better)",
                  os.path.join(args.out_dir, "compression_ratio.png"), higher_better=True)

    make_bar_plot(records, "perplexity", "Perplexity (lower better)",
                  os.path.join(args.out_dir, "perplexity.png"))

    make_bar_plot(records, "train_steps_per_sec", "Training Speed (higher better)",
                  os.path.join(args.out_dir, "training_speed.png"), higher_better=True)

    # Model size
    size_records = [{"tokenizer": r["tokenizer"], "size_m": r["num_params"]/1e6} for r in records]
    make_bar_plot(size_records, "size_m", "Model Size (M parameters)",
                  os.path.join(args.out_dir, "model_size.png"))

    print(f"\nAll plots saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()