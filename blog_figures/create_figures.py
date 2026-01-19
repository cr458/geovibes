#!/usr/bin/env python3
"""Create blog-ready figures for httpfs performance analysis.

Usage:
    python blog_figures/create_figures.py

Output:
    - blog_figures/performance_comparison.png
    - blog_figures/query_anatomy.png
    - blog_figures/architecture_diagram.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Design constants - clean, modern aesthetic
COLORS = {
    "primary": "#3b82f6",  # Blue
    "primary_dark": "#1d4ed8",
    "secondary": "#22c55e",  # Green
    "danger": "#ef4444",  # Red
    "warning": "#f59e0b",  # Amber
    "neutral": "#64748b",  # Gray
    "light": "#f1f5f9",
    "dark": "#1e293b",
    "white": "#ffffff",
    # Specific colors for components
    "faiss": "#8b5cf6",  # Purple
    "duckdb": "#f97316",  # Orange
    "s3": "#06b6d4",  # Cyan
    "cache": "#22c55e",  # Green
}

# Modern font settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "600",
        "axes.labelsize": 11,
        "axes.labelweight": "500",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_profile_results() -> dict:
    """Load profiling results from JSON."""
    results_path = Path(__file__).parent / "profile_results.json"
    with open(results_path) as f:
        return json.load(f)


def create_performance_comparison(results: dict):
    """Create bar chart comparing httpfs vs local cache performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    ids_counts = [100, 500, 1000]
    httpfs_times = [
        results["scattered_100_ids_ms"],
        results["scattered_500_ids_ms"],
        results["scattered_1000_ids_ms"],
    ]
    local_times = [
        results["local_cache_100_ids_ms"],
        results["local_cache_500_ids_ms"],
        results["local_cache_1000_ids_ms"],
    ]

    x = np.arange(len(ids_counts))
    width = 0.35

    # Bars
    bars1 = ax.bar(
        x - width / 2,
        httpfs_times,
        width,
        label="Remote (httpfs)",
        color=COLORS["danger"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        local_times,
        width,
        label="Local Cache",
        color=COLORS["secondary"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1,
    )

    # Labels and formatting
    ax.set_xlabel("Number of IDs fetched (FAISS search results)", fontweight="500")
    ax.set_ylabel("Query time (milliseconds)", fontweight="500")
    ax.set_title(
        "Fetching Scattered IDs: httpfs vs Local Geometry Cache",
        fontweight="600",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in ids_counts])
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=False,
        edgecolor=COLORS["light"],
    )

    # Use log scale for y-axis to show both clearly
    ax.set_yscale("log")
    ax.set_ylim(10, 50000)

    # Add value labels on bars
    for bar, val in zip(bars1, httpfs_times):
        ax.annotate(
            f"{val:,.0f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="500",
            color=COLORS["danger"],
        )

    for bar, val in zip(bars2, local_times):
        ax.annotate(
            f"{val:.0f}ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="500",
            color=COLORS["secondary"],
        )

    # Add speedup annotations
    for i, (http_t, local_t) in enumerate(zip(httpfs_times, local_times)):
        speedup = http_t / local_t
        ax.annotate(
            f"{speedup:.0f}x faster",
            xy=(x[i], 15),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="600",
            color=COLORS["primary_dark"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=COLORS["light"],
                edgecolor=COLORS["primary"],
                alpha=0.8,
            ),
        )

    # Subtle grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Note
    ax.text(
        0.5,
        -0.12,
        "FAISS returns IDs by embedding similarity, not storage order.\n"
        "Results are scattered across many row groups, each requiring ~500ms HTTP fetch.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=COLORS["neutral"],
        style="italic",
    )

    plt.tight_layout()
    output_path = Path(__file__).parent / "performance_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_query_anatomy(results: dict):
    """Create waterfall diagram showing the anatomy of a query."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Query stages with timing data
    stages = [
        (
            "Extension Load",
            results["extension_load_ms"],
            COLORS["neutral"],
            "One-time setup",
        ),
        ("Database Attach", results["cold_attach_ms"], COLORS["s3"], "Connect to S3"),
        ("Warmup Query", results["warmup_query_ms"], COLORS["duckdb"], "Load metadata"),
        ("FAISS Search", 2.0, COLORS["faiss"], "Local, ~2ms"),
        (
            "Fetch 1000 IDs\n(httpfs)",
            results["scattered_1000_ids_ms"],
            COLORS["danger"],
            "THE BOTTLENECK",
        ),
        (
            "Fetch 1000 IDs\n(local cache)",
            results["local_cache_1000_ids_ms"],
            COLORS["cache"],
            "With optimization",
        ),
    ]

    y_positions = np.arange(len(stages))[::-1]
    bar_height = 0.6

    # Calculate cumulative times for waterfall effect (first 4 stages)
    cumulative = 0
    for i, (name, duration, color, note) in enumerate(stages[:4]):
        ax.barh(
            y_positions[i],
            duration,
            height=bar_height,
            left=cumulative,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )

        # Duration label
        if duration > 100:
            label = f"{duration:,.0f}ms"
        else:
            label = f"{duration:.1f}ms"

        ax.text(
            cumulative + duration / 2,
            y_positions[i],
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="500",
            color="white",
        )

        cumulative += duration

    # Comparison bars (stages 5 and 6) - shown at same start position
    start_pos = cumulative + 500  # Gap for visual separation

    for i, (name, duration, color, note) in enumerate(stages[4:6]):
        idx = 4 + i
        ax.barh(
            y_positions[idx],
            duration,
            height=bar_height,
            left=start_pos,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1,
        )

        if duration > 100:
            label = f"{duration:,.0f}ms"
        else:
            label = f"{duration:.0f}ms"

        # For the long bar, put label inside; for short bar, put outside
        if duration > 1000:
            ax.text(
                start_pos + duration / 2,
                y_positions[idx],
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="500",
                color="white",
            )
        else:
            ax.text(
                start_pos + duration + 200,
                y_positions[idx],
                label,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="500",
                color=color,
            )

    # Add stage labels on left
    ax.set_yticks(y_positions)
    ax.set_yticklabels([s[0] for s in stages])

    # Add notes on right
    for i, (name, duration, color, note) in enumerate(stages):
        if i < 4:
            x_pos = cumulative + 100
        else:
            x_pos = start_pos + max(stages[4][1], stages[5][1]) + 500
        ax.text(
            x_pos,
            y_positions[i],
            note,
            va="center",
            fontsize=9,
            color=COLORS["neutral"],
            style="italic",
        )

    # Styling
    ax.set_xlabel("Time (milliseconds)", fontweight="500")
    ax.set_title("Anatomy of a Remote Database Query", fontweight="600", pad=20)

    # Add vertical lines for phase separation
    ax.axvline(x=cumulative + 250, color=COLORS["neutral"], linestyle="--", alpha=0.3)

    # Add phase labels
    ax.text(
        cumulative / 2,
        len(stages) + 0.5,
        "Cold Start Phase",
        ha="center",
        fontsize=11,
        fontweight="600",
        color=COLORS["dark"],
    )
    ax.text(
        start_pos + 5000,
        len(stages) + 0.5,
        "Search Phase\n(per query)",
        ha="center",
        fontsize=11,
        fontweight="600",
        color=COLORS["dark"],
    )

    # Set x-axis limit
    ax.set_xlim(-200, start_pos + 25000)

    # Subtle grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Remove y-axis spine
    ax.spines["left"].set_visible(False)

    # Add annotation arrow showing the improvement
    ax.annotate(
        "",
        xy=(start_pos + results["local_cache_1000_ids_ms"], y_positions[5]),
        xytext=(start_pos + results["scattered_1000_ids_ms"], y_positions[4]),
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["primary"],
            connectionstyle="arc3,rad=0.3",
            lw=2,
        ),
    )

    speedup = results["scattered_1000_ids_ms"] / results["local_cache_1000_ids_ms"]
    ax.text(
        start_pos + 8000,
        (y_positions[4] + y_positions[5]) / 2,
        f"{speedup:.0f}x\nfaster",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["primary"],
    )

    plt.tight_layout()
    output_path = Path(__file__).parent / "query_anatomy.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_architecture_diagram():
    """Create architecture diagram showing the hybrid local+remote approach."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Helper function for rounded boxes
    def draw_box(x, y, width, height, color, label, sublabel="", icon=""):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.9,
        )
        ax.add_patch(box)

        # Main label
        ax.text(
            x + width / 2,
            y + height / 2 + 0.15,
            label,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="600",
            color="white",
        )

        # Sublabel
        if sublabel:
            ax.text(
                x + width / 2,
                y + height / 2 - 0.25,
                sublabel,
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                alpha=0.9,
            )

    def draw_arrow(x1, y1, x2, y2, label="", curved=False, color=COLORS["neutral"]):
        style = "arc3,rad=0.2" if curved else "arc3,rad=0"
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            connectionstyle=style,
            color=color,
            lw=2,
            mutation_scale=15,
        )
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mid_x,
                mid_y + 0.3,
                label,
                ha="center",
                fontsize=9,
                color=color,
                fontweight="500",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.9,
                ),
            )

    # Title
    ax.text(
        7,
        7.5,
        "Hybrid Architecture: Local FAISS + Remote DuckDB",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["dark"],
    )

    # === USER/NOTEBOOK ===
    draw_box(0.5, 5, 2.5, 1.5, COLORS["primary"], "Jupyter Notebook", "User Query")

    # === LOCAL COMPONENTS ===
    # FAISS Index (cached locally)
    draw_box(4.5, 5.5, 2.5, 1.2, COLORS["faiss"], "FAISS Index", "~150MB, cached")
    ax.text(
        5.75,
        5.2,
        "LOCAL",
        fontsize=8,
        ha="center",
        color=COLORS["faiss"],
        fontweight="bold",
    )

    # Geometry Cache
    draw_box(4.5, 3.5, 2.5, 1.2, COLORS["cache"], "Geometry Cache", "~30MB, cached")
    ax.text(
        5.75,
        3.2,
        "LOCAL",
        fontsize=8,
        ha="center",
        color=COLORS["cache"],
        fontweight="bold",
    )

    # === REMOTE (S3) ===
    # S3 cloud box
    cloud_box = FancyBboxPatch(
        (8.5, 2),
        5,
        5.5,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=COLORS["light"],
        edgecolor=COLORS["s3"],
        linewidth=2,
        linestyle="--",
        alpha=0.5,
    )
    ax.add_patch(cloud_box)
    ax.text(
        11,
        7.2,
        "AWS S3 (Source Cooperative)",
        ha="center",
        fontsize=11,
        fontweight="600",
        color=COLORS["s3"],
    )

    # DuckDB on S3
    draw_box(9, 4.5, 4, 1.5, COLORS["duckdb"], "DuckDB Database", "~600MB on S3")

    # Full FAISS on S3 (grayed out - not used at runtime)
    draw_box(9, 2.5, 4, 1.2, COLORS["neutral"], "FAISS Index", "Source copy")

    # === ARROWS ===
    # User -> FAISS
    draw_arrow(3, 5.75, 4.5, 5.9, "1. Search", color=COLORS["faiss"])

    # FAISS -> returns IDs
    draw_arrow(7, 5.9, 7.5, 5.9, color=COLORS["faiss"])
    ax.text(7.25, 6.25, "IDs", fontsize=9, ha="center", color=COLORS["faiss"])

    # Two paths from IDs:
    # Path A: IDs -> Geometry Cache (fast)
    draw_arrow(5.75, 5.5, 5.75, 4.7, "2a. Geometries", color=COLORS["cache"])

    # Path B: IDs -> DuckDB (for embeddings only)
    draw_arrow(7.5, 5.5, 9, 5.25, "2b. Embeddings", curved=True, color=COLORS["duckdb"])

    # Results back to user
    draw_arrow(4.5, 4.1, 2.5, 5, "3. Results", curved=True, color=COLORS["primary"])

    # === LEGEND / NOTES ===
    notes_y = 1.2
    ax.text(
        0.5,
        notes_y,
        "Key Insight:",
        fontsize=11,
        fontweight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        0.5,
        notes_y - 0.5,
        "FAISS returns IDs by similarity, not storage order. Fetching scattered geometries\n"
        "from S3 requires loading many row groups (~500ms each). Local caching solves this.",
        fontsize=10,
        color=COLORS["neutral"],
    )

    # Timing annotations
    ax.text(
        5.75,
        2.8,
        "~30ms",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color=COLORS["cache"],
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["cache"]
        ),
    )

    ax.text(
        8.5,
        4.2,
        "~17s without cache!",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color=COLORS["danger"],
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["danger"]
        ),
    )

    plt.tight_layout()
    output_path = Path(__file__).parent / "architecture_diagram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_cold_start_breakdown(results: dict):
    """Create a pie/donut chart showing cold start breakdown."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data
    labels = ["Extension Load", "Database Attach", "Warmup Query"]
    sizes = [
        results["extension_load_ms"],
        results["cold_attach_ms"],
        results["warmup_query_ms"],
    ]
    colors = [COLORS["neutral"], COLORS["s3"], COLORS["duckdb"]]
    total = sum(sizes)

    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({pct/100*total:,.0f}ms)",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("500")

    # Center text
    ax.text(
        0,
        0,
        f"Total\n{total:,.0f}ms",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["dark"],
    )

    ax.set_title(
        "Cold Start Breakdown\n(first connection to remote database)",
        fontweight="600",
        pad=20,
    )

    plt.tight_layout()
    output_path = Path(__file__).parent / "cold_start_breakdown.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def create_row_group_diagram():
    """Create diagram explaining row group scatter problem."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # === Left: Unsorted data ===
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    ax1.set_title(
        "Unsorted Data: IDs Scattered Across Row Groups",
        fontweight="600",
        pad=10,
        color=COLORS["danger"],
    )

    # Draw row groups
    for i in range(4):
        box = FancyBboxPatch(
            (0.5, 4.5 - i * 1.2),
            6,
            0.9,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=COLORS["light"],
            edgecolor=COLORS["neutral"],
            linewidth=1,
        )
        ax1.add_patch(box)
        ax1.text(
            0.3,
            5 - i * 1.2,
            f"RG{i}",
            fontsize=9,
            ha="right",
            va="center",
            color=COLORS["neutral"],
        )

        # Show scattered IDs (red dots)
        np.random.seed(i + 42)
        for j in range(5):
            x = 0.8 + j * 1.1 + np.random.uniform(-0.2, 0.2)
            if np.random.random() < 0.7:  # 70% chance of highlight
                circle = plt.Circle(
                    (x, 5 - i * 1.2), 0.15, color=COLORS["danger"], alpha=0.7
                )
                ax1.add_patch(circle)

    # Arrow showing HTTP requests
    ax1.annotate(
        "",
        xy=(8, 5),
        xytext=(8, 1.5),
        arrowprops=dict(arrowstyle="<->", color=COLORS["danger"], lw=2),
    )
    ax1.text(
        8.5,
        3.25,
        "4 HTTP\nrequests\n~500ms each",
        fontsize=9,
        va="center",
        color=COLORS["danger"],
        fontweight="500",
    )

    # Total time
    ax1.text(
        5,
        0.5,
        "Total: ~2000ms per query",
        fontsize=11,
        ha="center",
        fontweight="bold",
        color=COLORS["danger"],
    )

    # === Right: With local cache ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title(
        "Local Geometry Cache: All Data Pre-loaded",
        fontweight="600",
        pad=10,
        color=COLORS["cache"],
    )

    # Single local cache block
    box = FancyBboxPatch(
        (1, 2),
        5,
        2.5,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=COLORS["cache"],
        edgecolor="white",
        linewidth=2,
        alpha=0.85,
    )
    ax2.add_patch(box)
    ax2.text(
        3.5,
        3.25,
        "Local Parquet\n(~30MB)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="600",
        color="white",
    )

    # Show all IDs available locally
    ax2.text(
        3.5,
        2.3,
        "All geometries indexed locally",
        ha="center",
        fontsize=10,
        color="white",
        alpha=0.9,
    )

    # Arrow showing single lookup
    ax2.annotate(
        "",
        xy=(7.5, 3.25),
        xytext=(6, 3.25),
        arrowprops=dict(arrowstyle="->", color=COLORS["cache"], lw=2),
    )
    ax2.text(
        8,
        3.25,
        "Direct\nlookup",
        fontsize=9,
        va="center",
        color=COLORS["cache"],
        fontweight="500",
    )

    # Total time
    ax2.text(
        5,
        0.5,
        "Total: ~30ms per query",
        fontsize=11,
        ha="center",
        fontweight="bold",
        color=COLORS["cache"],
    )

    plt.tight_layout()
    output_path = Path(__file__).parent / "row_group_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all figures."""
    print("Creating blog figures...")

    results = load_profile_results()

    create_performance_comparison(results)
    create_query_anatomy(results)
    create_cold_start_breakdown(results)
    create_row_group_diagram()
    create_architecture_diagram()

    print("\nAll figures saved to blog_figures/")


if __name__ == "__main__":
    main()
