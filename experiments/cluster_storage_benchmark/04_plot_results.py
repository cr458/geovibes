#!/usr/bin/env python3
"""
Step 4: Generate bar plots with error bars.

Creates visualizations comparing storage orderings and fetch methods.

Usage:
    uv run python 04_plot_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load results
config_dir = Path(__file__).parent
results_file = config_dir / "results" / "benchmark_results.json"

if not results_file.exists():
    print(f"Error: Results file not found: {results_file}")
    print("Run 03_run_benchmark.py first.")
    exit(1)

with open(results_file) as f:
    results = json.load(f)

# Create output directory
figures_dir = config_dir / "results" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


# Aggregate results
def aggregate_results(benchmarks, group_by):
    """Aggregate benchmark results by specified keys."""
    groups = defaultdict(list)
    for r in benchmarks:
        key = tuple(r[k] for k in group_by)
        groups[key].append(r)

    aggregated = {}
    for key, runs in groups.items():
        times = [r["elapsed_seconds"] for r in runs]
        rgs = [r["row_groups_touched"] for r in runs]
        aggregated[key] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "mean_rg": np.mean(rgs),
            "n_trials": len(runs),
        }
    return aggregated


def plot_by_storage_ordering(benchmarks, output_path):
    """Bar plot comparing storage orderings for each fetch method."""
    # Filter to best method (threadpool_32) and aggregate by variant
    best_method = "threadpool_32"
    filtered = [b for b in benchmarks if b["method"] == best_method]

    agg = aggregate_results(filtered, ["variant", "n_embeddings"])

    variants = sorted(set(b["variant"] for b in filtered))
    n_embeddings_list = sorted(set(b["n_embeddings"] for b in filtered))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fetch time by variant
    ax = axes[0]
    x = np.arange(len(n_embeddings_list))
    width = 0.25
    colors = {"baseline": "#94a3b8", "hilbert": "#3b82f6", "two_level": "#22c55e"}

    for i, variant in enumerate(variants):
        means = [agg[(variant, n)]["mean_time"] for n in n_embeddings_list]
        stds = [agg[(variant, n)]["std_time"] for n in n_embeddings_list]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=variant.replace("_", " ").title(),
            color=colors.get(variant, "#666"),
            capsize=3,
            alpha=0.9,
        )

    ax.set_xlabel("Number of Embeddings", fontsize=12)
    ax.set_ylabel("Fetch Time (seconds)", fontsize=12)
    ax.set_title(
        f"Embedding Fetch Time by Storage Ordering\n(Method: {best_method})",
        fontsize=14,
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(n_embeddings_list)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Row groups touched
    ax = axes[1]
    for i, variant in enumerate(variants):
        rgs = [agg[(variant, n)]["mean_rg"] for n in n_embeddings_list]
        ax.bar(
            x + i * width,
            rgs,
            width,
            label=variant.replace("_", " ").title(),
            color=colors.get(variant, "#666"),
            alpha=0.9,
        )

    ax.set_xlabel("Number of Embeddings", fontsize=12)
    ax.set_ylabel("Row Groups Touched", fontsize=12)
    ax.set_title("Row Groups Accessed by Storage Ordering", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(n_embeddings_list)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_by_fetch_method(benchmarks, output_path):
    """Bar plot comparing fetch methods for each storage ordering."""
    # Aggregate by method and variant for n=500
    filtered = [b for b in benchmarks if b["n_embeddings"] == 500]

    agg = aggregate_results(filtered, ["variant", "method"])

    variants = sorted(set(b["variant"] for b in filtered))
    methods = sorted(set(b["method"] for b in filtered))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(variants))
    width = 0.2
    colors = ["#3b82f6", "#22c55e", "#f59e0b"]

    for i, method in enumerate(methods):
        means = [agg.get((v, method), {"mean_time": 0})["mean_time"] for v in variants]
        stds = [agg.get((v, method), {"std_time": 0})["std_time"] for v in variants]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=method,
            color=colors[i % len(colors)],
            capsize=3,
            alpha=0.9,
        )

    ax.set_xlabel("Storage Ordering", fontsize=12)
    ax.set_ylabel("Fetch Time (seconds)", fontsize=12)
    ax.set_title(
        "Fetch Time by Method and Storage Ordering\n(n=500 embeddings)", fontsize=14
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels([v.replace("_", " ").title() for v in variants])
    ax.legend(title="Fetch Method")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_speedup_heatmap(benchmarks, output_path):
    """Heatmap showing speedup vs baseline for each configuration."""
    agg = aggregate_results(benchmarks, ["variant", "method", "n_embeddings"])

    variants = sorted(set(b["variant"] for b in benchmarks))
    methods = sorted(set(b["method"] for b in benchmarks))
    n_embeddings_list = sorted(set(b["n_embeddings"] for b in benchmarks))

    # Calculate speedup vs baseline for each method
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        speedups = np.zeros((len(variants), len(n_embeddings_list)))

        for i, variant in enumerate(variants):
            for j, n_emb in enumerate(n_embeddings_list):
                baseline_time = agg.get(("baseline", method, n_emb), {"mean_time": 1})[
                    "mean_time"
                ]
                variant_time = agg.get((variant, method, n_emb), {"mean_time": 1})[
                    "mean_time"
                ]
                speedups[i, j] = baseline_time / variant_time if variant_time > 0 else 1

        im = ax.imshow(speedups, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=6)

        # Add text annotations
        for i in range(len(variants)):
            for j in range(len(n_embeddings_list)):
                text = f"{speedups[i, j]:.1f}x"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if speedups[i, j] > 3 else "black",
                )

        ax.set_xticks(range(len(n_embeddings_list)))
        ax.set_xticklabels(n_embeddings_list)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([v.replace("_", " ").title() for v in variants])
        ax.set_xlabel("Number of Embeddings")
        ax.set_title(f"{method}")

    fig.suptitle("Speedup vs Baseline (higher = better)", fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Speedup")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def plot_combined_summary(benchmarks, output_path):
    """Single comprehensive plot with all key comparisons."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get aggregated data
    agg = aggregate_results(benchmarks, ["variant", "method", "n_embeddings"])

    variants = sorted(set(b["variant"] for b in benchmarks))
    methods = sorted(set(b["method"] for b in benchmarks))
    n_embeddings_list = sorted(set(b["n_embeddings"] for b in benchmarks))
    colors_variant = {
        "baseline": "#94a3b8",
        "hilbert": "#3b82f6",
        "two_level": "#22c55e",
    }

    # Plot 1: Time by variant (best method, n=500)
    ax = axes[0, 0]
    best_method = "threadpool_32"
    x = np.arange(len(variants))

    for n_emb in n_embeddings_list:
        means = [
            agg.get((v, best_method, n_emb), {"mean_time": 0})["mean_time"]
            for v in variants
        ]
        stds = [
            agg.get((v, best_method, n_emb), {"std_time": 0})["std_time"]
            for v in variants
        ]
        ax.bar(
            x + n_embeddings_list.index(n_emb) * 0.25,
            means,
            0.25,
            yerr=stds,
            label=f"n={n_emb}",
            capsize=3,
            alpha=0.9,
        )

    ax.set_ylabel("Fetch Time (seconds)")
    ax.set_title(f"Fetch Time by Storage Ordering ({best_method})")
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels([v.replace("_", " ").title() for v in variants])
    ax.legend(title="Embeddings")
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: Method comparison for n=500
    ax = axes[0, 1]
    n_emb = 500
    x = np.arange(len(methods))
    width = 0.25

    for i, variant in enumerate(variants):
        means = [
            agg.get((variant, m, n_emb), {"mean_time": 0})["mean_time"] for m in methods
        ]
        stds = [
            agg.get((variant, m, n_emb), {"std_time": 0})["std_time"] for m in methods
        ]
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=variant.replace("_", " ").title(),
            color=colors_variant.get(variant, "#666"),
            capsize=3,
            alpha=0.9,
        )

    ax.set_ylabel("Fetch Time (seconds)")
    ax.set_title(f"Method Comparison (n={n_emb})")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", "\n") for m in methods], fontsize=9)
    ax.legend(title="Storage")
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: Row groups touched
    ax = axes[1, 0]
    x = np.arange(len(n_embeddings_list))
    width = 0.25

    for i, variant in enumerate(variants):
        rgs = [
            agg.get((variant, best_method, n), {"mean_rg": 0})["mean_rg"]
            for n in n_embeddings_list
        ]
        ax.bar(
            x + i * width,
            rgs,
            width,
            label=variant.replace("_", " ").title(),
            color=colors_variant.get(variant, "#666"),
            alpha=0.9,
        )

    ax.set_xlabel("Number of Embeddings")
    ax.set_ylabel("Row Groups Touched")
    ax.set_title("Row Groups by Storage Ordering")
    ax.set_xticks(x + width)
    ax.set_xticklabels(n_embeddings_list)
    ax.legend(title="Storage")
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: Speedup vs baseline
    ax = axes[1, 1]
    x = np.arange(len(n_embeddings_list))
    width = 0.35

    for i, variant in enumerate([v for v in variants if v != "baseline"]):
        speedups = []
        for n_emb in n_embeddings_list:
            baseline_time = agg.get(("baseline", best_method, n_emb), {"mean_time": 1})[
                "mean_time"
            ]
            variant_time = agg.get((variant, best_method, n_emb), {"mean_time": 1})[
                "mean_time"
            ]
            speedups.append(baseline_time / variant_time if variant_time > 0 else 1)

        ax.bar(
            x + i * width,
            speedups,
            width,
            label=variant.replace("_", " ").title(),
            color=colors_variant.get(variant, "#666"),
            alpha=0.9,
        )

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of Embeddings")
    ax.set_ylabel("Speedup vs Baseline")
    ax.set_title(f"Speedup by Storage Ordering ({best_method})")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(n_embeddings_list)
    ax.legend(title="Storage")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cluster-Aligned Storage Benchmark Results", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()


def print_summary_table(benchmarks):
    """Print a summary table of results."""
    agg = aggregate_results(benchmarks, ["variant", "method", "n_embeddings"])

    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    print(
        f"\n{'Variant':<12} | {'Method':<18} | {'N':>6} | {'Mean (s)':>10} | {'Std (s)':>8} | {'Speedup':>8}"
    )
    print("-" * 90)

    # Get baseline times for speedup calculation
    baseline_times = {}
    for (variant, method, n_emb), data in agg.items():
        if variant == "baseline":
            baseline_times[(method, n_emb)] = data["mean_time"]

    for (variant, method, n_emb), data in sorted(agg.items()):
        baseline = baseline_times.get((method, n_emb), data["mean_time"])
        speedup = baseline / data["mean_time"] if data["mean_time"] > 0 else 1

        print(
            f"{variant:<12} | {method:<18} | {n_emb:>6} | {data['mean_time']:>10.2f} | {data['std_time']:>8.2f} | {speedup:>7.2f}x"
        )


def main():
    print("=" * 70)
    print("Step 4: Generate Benchmark Visualizations")
    print("=" * 70)

    benchmarks = results["benchmarks"]
    print(f"\nLoaded {len(benchmarks)} benchmark results")

    # Print summary table
    print_summary_table(benchmarks)

    # Generate plots
    print("\nGenerating plots...")

    plot_by_storage_ordering(
        benchmarks, figures_dir / "01_storage_ordering_comparison.png"
    )

    plot_by_fetch_method(benchmarks, figures_dir / "02_fetch_method_comparison.png")

    plot_speedup_heatmap(benchmarks, figures_dir / "03_speedup_heatmap.png")

    plot_combined_summary(benchmarks, figures_dir / "04_combined_summary.png")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"\nFigures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
