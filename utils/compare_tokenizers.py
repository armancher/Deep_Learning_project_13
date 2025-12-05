import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Fixed order and colors for all plots
TOKEN_ORDER = ["byte", "char", "BPE"]
TOKEN_COLORS = {
    "byte": "#A67C00",   # dark gold
    "char": "#8C2F39",   # deep brick red
    "BPE":  "#1A4B84",   # midnight blue
}



def safe_get(d, key, default=None):
    return d[key] if key in d and d[key] is not None else default

def normalize_tokenizer_key(name: str) -> str:
    """
    Normalize tokenizer names coming from JSON into a small fixed set:
    - byt5 / byte -> "byte"
    - char       -> "char"
    - bpe / BPE  -> "BPE"
    """
    if name is None:
        return "unknown"
    lower = name.lower()
    if "byt5" in lower or lower == "byte":
        return "byte"
    if "char" in lower:
        return "char"
    if "bpe" in lower:
        return "BPE"
    return name  # fallback, will just be plotted as-is (gray)

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

            raw_tok = safe_get(rec, "tokenizer", "unknown")
            rec["tokenizer"] = normalize_tokenizer_key(raw_tok)

            rec["checkpoint"] = safe_get(rec, "checkpoint", path)

            rec["bits_per_byte"] = safe_get(rec, "bits_per_byte", float("nan"))
            rec["compression_ratio_chars"] = safe_get(rec, "compression_ratio_chars", float("nan"))
            rec["perplexity"] = safe_get(rec, "perplexity", float("nan"))
            rec["loss"] = safe_get(rec, "loss", float("nan"))

            tokenization_eff = safe_get(rec, "tokenization_efficiency", {})
            rec["tokens_per_char"] = safe_get(tokenization_eff, "tokens_per_char", float("nan"))
            rec["chars_per_token"] = safe_get(tokenization_eff, "chars_per_token", float("nan"))

            # Convenience field for plotting model size in millions
            rec["size_m"] = (rec["num_params"] / 1e6) if rec["num_params"] else 0.0

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

def _prepare_metric_data(records, metric_key):
    """
    Build ordered (labels, values) lists for a given metric using fixed TOKEN_ORDER.
    """
    token_to_value = {}
    for r in records:
        tok = r["tokenizer"]
        val = r.get(metric_key, float("nan"))
        token_to_value[tok] = val

    labels = []
    values = []
    for tok in TOKEN_ORDER:
        if tok in token_to_value:
            labels.append(tok)
            v = token_to_value[tok]
            if v is None or np.isnan(v):
                v = 0.0
            values.append(v)

    return labels, values

def make_bar_plot(records, metric_key, ylabel, outfile, higher_better=False):
    labels, values = _prepare_metric_data(records, metric_key)

    if len(values) == 0:
        print(f"No values for {metric_key}, skipping plot {outfile}")
        return

    ymax = max(values)
    top = ymax * 1.25 if ymax != 0 else 1.0
    pad = top * 0.03

    colors = [TOKEN_COLORS.get(label, "gray") for label in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.85, edgecolor='black')
    plt.ylim(0, top)

    # Labels above bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + pad,
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    plt.ylabel(ylabel, fontsize=12)
    #plt.xlabel("Tokenizer", fontsize=12)
    plt.title(f"Tokenizer Comparison: {ylabel}", fontsize=14, fontweight='bold')
    #plt.grid(axis='y', alpha=0.3, linestyle='--')
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
        labels, values = _prepare_metric_data(records, metric_key)
        if len(values) == 0:
            continue

        ymax = max(values)
        top = ymax * 1.25 if ymax != 0 else 1.0
        pad = top * 0.03

        colors = [TOKEN_COLORS.get(label, "gray") for label in labels]

        bars = axes[idx].bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_ylim(0, top)

        for bar, value in zip(bars, values):
            axes[idx].text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + pad,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    output_path = os.path.join(out_dir, "comprehensive_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive chart → {output_path}")
    plt.close()

def make_two_panel_plot(records, metrics, outfile):
    """
    Create a 1x2 (one row, two columns) figure for LaTeX:
    metrics: list of (metric_key, ylabel, higher_better)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (metric_key, ylabel, higher_better) in zip(axes, metrics):
        labels, values = _prepare_metric_data(records, metric_key)
        if len(values) == 0:
            continue

        ymax = max(values)
        top = ymax * 1.25 if ymax != 0 else 1.0
        pad = top * 0.03

        colors = [TOKEN_COLORS.get(label, "gray") for label in labels]

        bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black')
        ax.set_ylim(0, top)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        #ax.set_xlabel("Tokenizer", fontsize=10)
        #ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + pad,
                f'{value:.4f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    fig.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved 1x2 plot → {outfile}")
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

    # Single-metric bar plots
    make_bar_plot(
        records,
        "bits_per_char",
        "Bits per Character",
        os.path.join(args.out_dir, "bits_per_char.png")
    )

    make_bar_plot(
        records,
        "bits_per_byte",
        "Bits per Byte",
        os.path.join(args.out_dir, "bits_per_byte.png")
    )

    make_bar_plot(
        records,
        "compression_ratio_chars",
        "Compression Ratio (higher better)",
        os.path.join(args.out_dir, "compression_ratio.png"),
        higher_better=True
    )

    make_bar_plot(
        records,
        "perplexity",
        "Perplexity (lower better)",
        os.path.join(args.out_dir, "perplexity.png")
    )

    make_bar_plot(
        records,
        "train_steps_per_sec",
        "Training Speed (higher better)",
        os.path.join(args.out_dir, "training_speed.png"),
        higher_better=True
    )

    # Model size plot (uses precomputed size_m in records)
    make_bar_plot(
        records,
        "size_m",
        "Model Size (M parameters, lower better)",
        os.path.join(args.out_dir, "model_size.png")
    )

    # 1x2 plots for LaTeX
    # Figure 1: bits per char + bits per byte
    bpc_bpb_metrics = [
        ("bits_per_char", "Bits per Character (lower better)", False),
        ("bits_per_byte", "Bits per Byte (lower better)", False),
    ]
    make_two_panel_plot(
        records,
        bpc_bpb_metrics,
        os.path.join(args.out_dir, "bpc_bpb_1x2.png")
    )

    # Figure 2: model size + training steps per second
    size_speed_metrics = [
        ("size_m", "Model Size (M parameters, lower better)", False),
        ("train_steps_per_sec", "Training Speed (higher better)", True),
        ]
    make_two_panel_plot(
        records,
        size_speed_metrics,
        os.path.join(args.out_dir, "size_speed_1x2.png")
    )

    print(f"\nAll plots saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()
