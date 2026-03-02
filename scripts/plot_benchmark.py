#!/usr/bin/env python3
"""Plot benchmark results from benchmark_scaling.sh CSV output."""

import sys
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# -- Consistent styling --
BACKEND_COLORS = {
    "host": "#1f77b4",
    "multicore": "#ff7f0e",
    "openacc-gpu": "#2ca02c",
    "cuda-gpu": "#d62728",
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


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # Drop failed / unparseable rows
    df = df[~df["wall_clock"].astype(str).isin(["FAILED", "PARSE_ERROR"])]
    numeric_cols = ["threads", "ni", "nj", "nk", "niter",
                    "wall_clock", "coriolis", "hor_visc", "vert_visc",
                    "barotropic", "continuity", "time_per_step"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["wall_clock"])
    return df


def pick_backends_for_comparison(df_grid):
    """Return one row per key backend: host, best multicore, openacc-gpu, cuda-gpu."""
    rows = []
    host = df_grid[df_grid["backend"] == "host"]
    if not host.empty:
        rows.append(host.iloc[0])

    mc = df_grid[df_grid["backend"] == "multicore"]
    if not mc.empty:
        best = mc.loc[mc["wall_clock"].idxmin()]
        rows.append(best)

    for gpu in ["openacc-gpu", "cuda-gpu"]:
        g = df_grid[df_grid["backend"] == gpu]
        if not g.empty:
            rows.append(g.iloc[0])

    return pd.DataFrame(rows)


def backend_label(row):
    if row["backend"] == "multicore":
        return f"multicore ({int(row['threads'])}t)"
    return row["backend"]


def plot_grid(df_grid, grid_label, out_dir):
    """Create a 2x2 figure for one grid size."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Benchmark — Grid {grid_label}", fontsize=15, fontweight="bold")

    mc = df_grid[df_grid["backend"] == "multicore"].sort_values("threads")
    host_row = df_grid[df_grid["backend"] == "host"]
    host_time = host_row["wall_clock"].values[0] if not host_row.empty else None

    # ---- Top-left: Multicore scaling (log-log) ----
    ax = axes[0, 0]
    if not mc.empty:
        threads = mc["threads"].values
        times = mc["wall_clock"].values
        ax.loglog(threads, times, "o-", color=BACKEND_COLORS["multicore"],
                  label="multicore", linewidth=2, markersize=6)
        # Ideal scaling reference from single-thread multicore time
        t1 = times[0]
        ideal = t1 / threads
        ax.loglog(threads, ideal, "--", color="gray", alpha=0.6, label="ideal scaling")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Wall clock (s)")
    ax.set_title("Multicore Scaling")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # ---- Top-right: Speedup ----
    ax = axes[0, 1]
    if host_time and not mc.empty:
        threads = mc["threads"].values
        speedup = host_time / mc["wall_clock"].values
        ax.plot(threads, speedup, "o-", color=BACKEND_COLORS["multicore"],
                label="multicore", linewidth=2, markersize=6)
        # Ideal linear speedup from host baseline
        ax.plot(threads, threads * (host_time / mc["wall_clock"].values[0]),
                "--", color="gray", alpha=0.6, label="ideal (from 1-thread)")
        # GPU backends as horizontal lines
        for gpu_backend in ["openacc-gpu", "cuda-gpu"]:
            gpu = df_grid[df_grid["backend"] == gpu_backend]
            if not gpu.empty:
                gpu_speedup = host_time / gpu["wall_clock"].values[0]
                ax.axhline(gpu_speedup, linestyle=":", linewidth=2,
                           color=BACKEND_COLORS[gpu_backend],
                           label=f"{gpu_backend} ({gpu_speedup:.1f}x)")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup vs host serial")
    ax.set_title("Speedup")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ---- Bottom-left: Backend comparison (horizontal bar) ----
    ax = axes[1, 0]
    comp = pick_backends_for_comparison(df_grid)
    if not comp.empty:
        labels = [backend_label(r) for _, r in comp.iterrows()]
        times = comp["wall_clock"].values
        colors = [BACKEND_COLORS.get(r["backend"], "#999999") for _, r in comp.iterrows()]
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, times, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Wall clock (s)")
        ax.set_title("Backend Comparison")
        ax.grid(True, axis="x", alpha=0.3)
        # Value labels on bars
        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + max(times) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{t:.3f}s", va="center", fontsize=9)
        ax.set_xlim(right=max(times) * 1.18)

    # ---- Bottom-right: Component breakdown (stacked horizontal bar) ----
    ax = axes[1, 1]
    if not comp.empty:
        labels = [backend_label(r) for _, r in comp.iterrows()]
        y_pos = np.arange(len(labels))
        left = np.zeros(len(labels))
        for comp_name in COMPONENTS:
            vals = comp[comp_name].values.astype(float)
            ax.barh(y_pos, vals, left=left, height=0.6,
                    color=COMPONENT_COLORS[comp_name],
                    label=COMPONENT_LABELS[comp_name], edgecolor="white")
            left += vals
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time (s)")
        ax.set_title("Component Breakdown")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"scaling_{grid_label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_summary(df, out_dir):
    """Wall clock vs grid size for key backends."""
    grids = df["grid"].unique()
    # Sort grids by ni (first dimension)
    grids = sorted(grids, key=lambda g: int(g.split("x")[0]))

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Wall Clock vs Grid Size", fontsize=14, fontweight="bold")

    # Collect one line per key backend
    entries = []
    for grid in grids:
        dg = df[df["grid"] == grid]
        host = dg[dg["backend"] == "host"]
        if not host.empty:
            entries.append(("host", grid, host["wall_clock"].values[0]))
        mc = dg[dg["backend"] == "multicore"]
        if not mc.empty:
            best = mc.loc[mc["wall_clock"].idxmin()]
            entries.append((f"multicore (best)", grid, best["wall_clock"]))
        for gpu in ["openacc-gpu", "cuda-gpu"]:
            g = dg[dg["backend"] == gpu]
            if not g.empty:
                entries.append((gpu, grid, g["wall_clock"].values[0]))

    summary_df = pd.DataFrame(entries, columns=["label", "grid", "wall_clock"])

    color_map = {
        "host": BACKEND_COLORS["host"],
        "multicore (best)": BACKEND_COLORS["multicore"],
        "openacc-gpu": BACKEND_COLORS["openacc-gpu"],
        "cuda-gpu": BACKEND_COLORS["cuda-gpu"],
    }

    for label in summary_df["label"].unique():
        sub = summary_df[summary_df["label"] == label]
        ax.plot(sub["grid"].values, sub["wall_clock"].values, "o-",
                color=color_map.get(label, "#999"), label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Wall Clock (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_dir / "summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
    if not os.path.isfile(csv_path):
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("benchmark_plots")
    out_dir.mkdir(exist_ok=True)

    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    grids = sorted(df["grid"].unique(), key=lambda g: int(g.split("x")[0]))
    for grid in grids:
        print(f"\nPlotting grid {grid}...")
        plot_grid(df[df["grid"] == grid].copy(), grid, out_dir)

    print("\nPlotting summary...")
    plot_summary(df, out_dir)
    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
