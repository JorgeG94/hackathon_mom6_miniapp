#!/usr/bin/env python3
"""Plot results from strong_scaling.sh (larger_scaling_results.csv)."""

import sys
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -- Consistent styling --
BACKEND_COLORS = {
    "host": "#1f77b4",
    "openacc-gpu": "#2ca02c",
    "cuda-gpu": "#d62728",
}
BACKEND_LABELS = {
    "host": "Host (CPU)",
    "openacc-gpu": "OpenACC GPU",
    "cuda-gpu": "CUDA GPU",
}
COMPONENT_COLORS = {
    "coriolis": "#4e79a7",
    "hor_visc": "#f28e2b",
    "vert_visc": "#e15759",
    "barotropic": "#76b7b2",
    "continuity": "#59a14f",
}
COMPONENT_LABELS = {
    "coriolis": "Coriolis",
    "hor_visc": "Hor Visc",
    "vert_visc": "Vert Visc",
    "barotropic": "Barotropic",
    "continuity": "Continuity",
}
COMPONENTS = list(COMPONENT_COLORS.keys())
BACKENDS_ORDER = ["host", "openacc-gpu", "cuda-gpu"]


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[~df["wall_clock"].astype(str).isin(["FAILED", "PARSE_ERROR"])]
    numeric_cols = ["threads", "ni", "nj", "nk", "niter",
                    "wall_clock", "coriolis", "hor_visc", "vert_visc",
                    "barotropic", "continuity", "time_per_step"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wall_clock"])
    # Total grid points for x-axis sorting
    df["gridpoints"] = df["ni"] * df["nj"] * df["nk"]
    return df


def sorted_grids(df):
    """Return grid labels sorted by ni."""
    grids = df["grid"].unique()
    return sorted(grids, key=lambda g: int(g.split("x")[0]))


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "larger_scaling_results.csv"
    if not os.path.isfile(csv_path):
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("benchmark_plots")
    out_dir.mkdir(exist_ok=True)

    df = load_data(csv_path)
    grids = sorted_grids(df)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Grids: {grids}")

    # =====================================================================
    # Figure 1: 2x2 overview
    # =====================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Larger Grid Scaling — Host vs OpenACC vs CUDA",
                 fontsize=15, fontweight="bold")

    # -- Top-left: Wall clock vs grid size (log scale) --
    ax = axes[0, 0]
    for backend in BACKENDS_ORDER:
        sub = df[df["backend"] == backend].sort_values("ni")
        if sub.empty:
            continue
        ax.plot(sub["grid"].values, sub["wall_clock"].values, "o-",
                color=BACKEND_COLORS[backend],
                label=BACKEND_LABELS[backend], linewidth=2, markersize=7)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Wall Clock (s)")
    ax.set_yscale("log")
    ax.set_title("Wall Clock vs Grid Size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # -- Top-right: Speedup over host --
    ax = axes[0, 1]
    host_times = df[df["backend"] == "host"].set_index("grid")["wall_clock"]
    for backend in ["openacc-gpu", "cuda-gpu"]:
        sub = df[df["backend"] == backend].sort_values("ni")
        if sub.empty:
            continue
        speedups = []
        grid_labels = []
        for _, row in sub.iterrows():
            if row["grid"] in host_times.index:
                speedups.append(host_times[row["grid"]] / row["wall_clock"])
                grid_labels.append(row["grid"])
        if speedups:
            ax.plot(grid_labels, speedups, "o-",
                    color=BACKEND_COLORS[backend],
                    label=BACKEND_LABELS[backend], linewidth=2, markersize=7)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Speedup vs Host")
    ax.set_title("GPU Speedup over Host")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="1x (host)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # -- Bottom-left: Time per step vs grid size --
    ax = axes[1, 0]
    for backend in BACKENDS_ORDER:
        sub = df[df["backend"] == backend].sort_values("ni")
        if sub.empty:
            continue
        ax.plot(sub["grid"].values, sub["time_per_step"].values, "s-",
                color=BACKEND_COLORS[backend],
                label=BACKEND_LABELS[backend], linewidth=2, markersize=7)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Time per RK2 Step (s)")
    ax.set_yscale("log")
    ax.set_title("Time per Step vs Grid Size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # -- Bottom-right: Component breakdown for largest grid --
    ax = axes[1, 1]
    largest_grid = grids[-1]
    dg = df[df["grid"] == largest_grid]
    backends_present = [b for b in BACKENDS_ORDER if b in dg["backend"].values]
    labels = [BACKEND_LABELS[b] for b in backends_present]
    y_pos = np.arange(len(labels))
    left = np.zeros(len(labels))
    for comp_name in COMPONENTS:
        vals = np.array([dg[dg["backend"] == b][comp_name].values[0]
                         for b in backends_present])
        ax.barh(y_pos, vals, left=left, height=0.6,
                color=COMPONENT_COLORS[comp_name],
                label=COMPONENT_LABELS[comp_name], edgecolor="white")
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Component Breakdown — {largest_grid}")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    overview_path = out_dir / "larger_scaling_overview.png"
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {overview_path}")

    # =====================================================================
    # Figure 2: Component breakdown per grid size
    # =====================================================================
    n_grids = len(grids)
    fig, axes = plt.subplots(1, n_grids, figsize=(5 * n_grids, 5), sharey=True)
    if n_grids == 1:
        axes = [axes]
    fig.suptitle("Component Breakdown by Grid Size",
                 fontsize=14, fontweight="bold")

    for ax, grid in zip(axes, grids):
        dg = df[df["grid"] == grid]
        backends_present = [b for b in BACKENDS_ORDER if b in dg["backend"].values]
        labels = [BACKEND_LABELS[b] for b in backends_present]
        y_pos = np.arange(len(labels))
        left = np.zeros(len(labels))
        for comp_name in COMPONENTS:
            vals = np.array([dg[dg["backend"] == b][comp_name].values[0]
                             for b in backends_present])
            ax.barh(y_pos, vals, left=left, height=0.6,
                    color=COMPONENT_COLORS[comp_name],
                    label=COMPONENT_LABELS[comp_name], edgecolor="white")
            left += vals
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time (s)")
        ax.set_title(grid)
        ax.grid(True, axis="x", alpha=0.3)

    # Single legend for the whole figure
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center",
               ncol=len(COMPONENTS), fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    breakdown_path = out_dir / "larger_scaling_breakdown.png"
    fig.savefig(breakdown_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {breakdown_path}")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
