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

            # Original metrics
            rec["bits_per_char"] = safe_get(rec, "bits_per_char", float("nan"))
            rec["train_steps_per_sec"] = safe_get(rec, "train_steps_per_sec", float("nan"))
            rec["num_params"] = safe_get(rec, "num_params", 0)
            rec["tokenizer"] = safe_get(rec, "tokenizer", "unknown")
            rec["checkpoint"] = safe_get(rec, "checkpoint", path)
            
            # NEW: Enhanced metrics support
            rec["bits_per_byte"] = safe_get(rec, "bits_per_byte", float("nan"))
            rec["compression_ratio_chars"] = safe_get(rec, "compression_ratio_chars", float("nan"))
            rec["perplexity"] = safe_get(rec, "perplexity", float("nan"))
            rec["loss"] = safe_get(rec, "loss", float("nan"))
            
            # Tokenization efficiency
            tokenization_eff = safe_get(rec, "tokenization_efficiency", {})
            rec["tokens_per_char"] = safe_get(tokenization_eff, "tokens_per_char", float("nan"))
            rec["chars_per_token"] = safe_get(tokenization_eff, "chars_per_token", float("nan"))

            records.append(rec)
    return records

def print_table(records):
    print("="*70)
    print(" TOKENIZER COMPARISON TABLE")
    print("="*70)
    
    # Enhanced header
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
    """NEW: Detailed comparison with explanations"""
    print("\n" + "="*70)
    print(" DETAILED ANALYSIS")
    print("="*70)
    
    # Find best values (lower is better for most metrics)
    best_bpc = min(r["bits_per_char"] for r in records)
    best_bpb = min(r["bits_per_byte"] for r in records)
    best_ppl = min(r["perplexity"] for r in records)
    best_comp = max(r["compression_ratio_chars"] for r in records)
    best_speed = max(r["train_steps_per_sec"] for r in records)
    
    for r in records:
        print(f"\n--- {r['tokenizer'].upper()} TOKENIZER ---")
        print(f"  Efficiency:  {r['bits_per_char']:.4f} bits/char {'(best)' if r['bits_per_char'] == best_bpc else ''}")
        print(f"  Byte eff:    {r['bits_per_byte']:.4f} bits/byte {'(best)' if r['bits_per_byte'] == best_bpb else ''}")
        print(f"  Compression: {r['compression_ratio_chars']:.2f} chars/token {'(best)' if r['compression_ratio_chars'] == best_comp else ''}")
        print(f"  Quality:     {r['perplexity']:.2f} perplexity {'(best)' if r['perplexity'] == best_ppl else ''}")
        print(f"  Speed:       {r['train_steps_per_sec']:.2f} steps/sec { '(best)' if r['train_steps_per_sec'] == best_speed else ''}")
        print(f"  Tokenization: {r['tokens_per_char']:.3f} tokens/char")

def make_bar_plot(records, metric_key, ylabel, outfile, higher_better=False):
    """Enhanced plotting with better visuals"""
    labels = [r["tokenizer"] for r in records]
    values = [r[metric_key] for r in records]
    
    # Choose colors based on performance
    if higher_better:
        colors = ['lightgreen' if v == max(values) else 'lightcoral' for v in values]
    else:
        colors = ['lightgreen' if v == min(values) else 'lightcoral' for v in values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Tokenizer", fontsize=12)
    plt.title(f"Tokenizer Comparison: {ylabel}", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved enhanced plot → {outfile}")
    plt.close()

def create_comprehensive_chart(records, out_dir):
    """NEW: Create a comprehensive comparison chart"""
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
        if idx >= len(axes):
            break
            
        labels = [r["tokenizer"] for r in records]
        values = [r[metric_key] for r in records]
        
        # Color coding
        if higher_better:
            colors = ['lightgreen' if v == max(values) else 'lightcoral' for v in values]
        else:
            colors = ['lightgreen' if v == min(values) else 'lightcoral' for v in values]
        
        bars = axes[idx].bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_ylabel(title.split('(')[0] if '(' in title else title)
        
        # Add value labels
        for bar, value in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                          f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplots
    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    comprehensive_path = os.path.join(out_dir, "comprehensive_comparison.png")
    plt.savefig(comprehensive_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive chart → {comprehensive_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Enhanced tokenizer comparison tool")
    parser.add_argument("metrics_files", nargs="+", help="JSON metrics files to compare")
    parser.add_argument("--out_dir", default="results/plots", help="Output directory for plots")
    parser.add_argument("--comprehensive", action="store_true", help="Generate comprehensive analysis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_metrics(args.metrics_files)

    # Print comparisons
    print_table(records)
    
    if args.comprehensive:
        print_detailed_comparison(records)
        create_comprehensive_chart(records, args.out_dir)

    # Generate individual plots
    make_bar_plot(records, "bits_per_char", "Bits per Character (lower better)",
                  os.path.join(args.out_dir, "bits_per_char.png"), higher_better=False)
    
    make_bar_plot(records, "bits_per_byte", "Bits per Byte (lower better)",
                  os.path.join(args.out_dir, "bits_per_byte.png"), higher_better=False)
    
    make_bar_plot(records, "compression_ratio_chars", "Compression Ratio (higher better)",
                  os.path.join(args.out_dir, "compression_ratio.png"), higher_better=True)
    
    make_bar_plot(records, "perplexity", "Perplexity (lower better)",
                  os.path.join(args.out_dir, "perplexity.png"), higher_better=False)
    
    make_bar_plot(records, "train_steps_per_sec", "Training Speed (higher better)",
                  os.path.join(args.out_dir, "training_speed.png"), higher_better=True)
    
    # Model size plot
    pm_vals = [{"tokenizer": r["tokenizer"], "size_m": r["num_params"]/1e6} for r in records]
    make_bar_plot(pm_vals, "size_m", "Model Size (M parameters)",
                  os.path.join(args.out_dir, "model_size.png"), higher_better=False)

    print(f"\nAll plots saved to: {args.out_dir}/")

if __name__ == "__main__":
    main()