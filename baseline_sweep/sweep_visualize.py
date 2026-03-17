"""Visualize baseline sweep results from sweep_summary.csv.

Produces three outputs:
  1. Heatmap grids — for each metric, sliced by inject_every
  2. Top-N table  — best combos ranked by reward
  3. Sensitivity   — marginal effect of each variable at the optimum
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Metrics to plot in heatmaps
HEATMAP_METRICS = [
    ("reward_mean", "Mean Reward"),
    ("triple_product_mean", "Triple Product (n·T·τ)"),
    ("fgw_n_e_volume_avg_mean", "Greenwald Fraction (fGW)"),
    ("T_e_volume_avg_mean", "Te Volume Avg (keV)"),
    ("T_i_volume_avg_mean", "Ti Volume Avg (keV)"),
    ("n_e_volume_avg_mean", "ne Volume Avg (m⁻³)"),
    ("tau_E_mean", "τ_E (s)"),
]

# Metrics shown in sensitivity plot
SENSITIVITY_METRICS = [
    ("reward_mean", "Reward", "tab:blue"),
    ("triple_product_mean", "Triple Product", "tab:orange"),
    ("fgw_n_e_volume_avg_mean", "fGW", "tab:green"),
    ("T_e_volume_avg_mean", "Te (keV)", "tab:red"),
    ("T_i_volume_avg_mean", "Ti (keV)", "tab:purple"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize baseline parameter sweep results.")
    p.add_argument("--csv", default="baseline_sweep/eval_logs/sweep_summary.csv",
                    help="Path to sweep_summary.csv")
    p.add_argument("--output-dir", default="baseline_sweep/eval_logs",
                    help="Directory for output figures")
    p.add_argument("--top-n", type=int, default=20, help="Number of top combos to show")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ═══════════════════════════════════════════════════════════
#  Layer 1: Heatmap grids
# ═══════════════════════════════════════════════════════════
def plot_heatmaps(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    ie_values = sorted(df["inject_every"].unique())
    vel_values = sorted(df["velocity_mps"].unique())
    thk_values = sorted(df["thickness_mm"].unique())

    # Pick up to 6 representative inject_every slices
    if len(ie_values) > 6:
        ie_slices = [ie_values[i] for i in np.linspace(0, len(ie_values) - 1, 6, dtype=int)]
    else:
        ie_slices = ie_values

    for metric_key, metric_label in HEATMAP_METRICS:
        if metric_key not in df.columns:
            continue

        n_slices = len(ie_slices)
        fig, axes = plt.subplots(1, n_slices, figsize=(4.5 * n_slices, 4), squeeze=False)
        axes = axes[0]

        # Global color scale
        vmin = df[metric_key].min()
        vmax = df[metric_key].max()
        if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
            vmin, vmax = 0, 1
        norm = Normalize(vmin=vmin, vmax=vmax)

        for ax_idx, ie in enumerate(ie_slices):
            ax = axes[ax_idx]
            subset = df[df["inject_every"] == ie]

            # Build 2D grid: rows=thickness, cols=velocity
            grid = np.full((len(thk_values), len(vel_values)), np.nan)
            for _, row in subset.iterrows():
                vi = vel_values.index(row["velocity_mps"])
                ti = thk_values.index(row["thickness_mm"])
                grid[ti, vi] = row[metric_key]

            im = ax.imshow(
                grid, origin="lower", aspect="auto", norm=norm,
                extent=[0, len(vel_values), 0, len(thk_values)],
                cmap="viridis",
            )

            ax.set_xticks(np.arange(len(vel_values)) + 0.5)
            ax.set_xticklabels([f"{v:.0f}" for v in vel_values], fontsize=7, rotation=45)
            ax.set_yticks(np.arange(len(thk_values)) + 0.5)
            ax.set_yticklabels([f"{t:.1f}" for t in thk_values], fontsize=7)
            ax.set_title(f"ie={ie}", fontsize=9)
            if ax_idx == 0:
                ax.set_ylabel("Thickness (mm)")
            ax.set_xlabel("Velocity (m/s)")

            # Annotate cells
            for ti_idx in range(len(thk_values)):
                for vi_idx in range(len(vel_values)):
                    val = grid[ti_idx, vi_idx]
                    if np.isfinite(val):
                        txt = f"{val:.3f}" if abs(val) < 100 else f"{val:.1e}"
                        ax.text(
                            vi_idx + 0.5, ti_idx + 0.5, txt,
                            ha="center", va="center", fontsize=5,
                            color="white" if (val - vmin) / (vmax - vmin + 1e-9) < 0.5 else "black",
                        )

        # Shared colorbar
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=metric_label)

        fig.suptitle(f"{metric_label}  (by inject_every slice)", fontsize=11, y=1.02)
        fig.tight_layout(rect=[0, 0, 0.92, 1.0])

        safe_name = metric_key.replace("/", "_")
        fig.savefig(output_dir / f"heatmap_{safe_name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved heatmap_{safe_name}.png")


# ═══════════════════════════════════════════════════════════
#  Layer 2: Top-N table
# ═══════════════════════════════════════════════════════════
def print_and_save_top_n(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    display_cols = [
        "inject_every", "velocity_mps", "thickness_mm",
        "reward_mean", "triple_product_mean",
        "fgw_n_e_volume_avg_mean", "T_e_volume_avg_mean", "T_i_volume_avg_mean",
        "n_e_volume_avg_mean", "tau_E_mean", "unsafe_step_count",
    ]
    cols = [c for c in display_cols if c in df.columns]

    ranked = df.sort_values("reward_mean", ascending=False).head(top_n).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "rank"

    print(f"\n{'=' * 120}")
    print(f"  TOP {top_n} BASELINE CONFIGURATIONS (by reward_mean)")
    print(f"{'=' * 120}")
    print(ranked[cols].to_string(float_format="%.4f"))
    print()

    csv_path = output_dir / f"sweep_top{top_n}.csv"
    ranked[cols].to_csv(csv_path)
    print(f"  Saved {csv_path.name}")


# ═══════════════════════════════════════════════════════════
#  Layer 3: Marginal sensitivity
# ═══════════════════════════════════════════════════════════
def plot_sensitivity(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if df.empty:
        return

    best_row = df.loc[df["reward_mean"].idxmax()]
    best_ie = best_row["inject_every"]
    best_vel = best_row["velocity_mps"]
    best_thk = best_row["thickness_mm"]

    variables = [
        ("inject_every", "Inject Every (steps)", best_vel, best_thk,
         lambda d, v, t: d[(d["velocity_mps"] == v) & (d["thickness_mm"] == t)]),
        ("velocity_mps", "Velocity (m/s)", best_ie, best_thk,
         lambda d, ie, t: d[(d["inject_every"] == ie) & (d["thickness_mm"] == t)]),
        ("thickness_mm", "Thickness (mm)", best_ie, best_vel,
         lambda d, ie, v: d[(d["inject_every"] == ie) & (d["velocity_mps"] == v)]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (var_key, var_label, fix1, fix2, filter_fn) in zip(axes, variables):
        subset = filter_fn(df, fix1, fix2).sort_values(var_key)
        if subset.empty:
            continue

        x = subset[var_key].values

        # Left y-axis: reward
        color_reward = "tab:blue"
        ax.set_xlabel(var_label)
        ax.set_ylabel("Reward", color=color_reward)
        ax.plot(x, subset["reward_mean"].values, "o-", color=color_reward, linewidth=2, label="Reward")
        ax.tick_params(axis="y", labelcolor=color_reward)

        # Right y-axis: other metrics (normalized to [0,1] range for comparison)
        ax2 = ax.twinx()
        for metric_key, metric_label, color in SENSITIVITY_METRICS[1:]:
            if metric_key not in subset.columns:
                continue
            vals = subset[metric_key].values
            if np.all(np.isnan(vals)):
                continue
            ax2.plot(x, vals, "s--", color=color, linewidth=1.2, markersize=4, label=metric_label)
        ax2.set_ylabel("Diagnostic Value")
        ax2.tick_params(axis="y")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

        # Title with fixed params
        if var_key == "inject_every":
            ax.set_title(f"vel={best_vel:.0f}, thk={best_thk:.1f}mm (fixed)")
        elif var_key == "velocity_mps":
            ax.set_title(f"ie={int(best_ie)}, thk={best_thk:.1f}mm (fixed)")
        else:
            ax.set_title(f"ie={int(best_ie)}, vel={best_vel:.0f} (fixed)")

    fig.suptitle(
        f"Marginal Sensitivity (best: ie={int(best_ie)}, vel={best_vel:.0f}, thk={best_thk:.1f}mm)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "sweep_sensitivity.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sweep_sensitivity.png")


def main() -> None:
    args = parse_args()
    csv_path = resolve(args.csv)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run sweep_eval.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    print("\n[1/3] Generating heatmaps ...")
    plot_heatmaps(df, output_dir, args.dpi)

    print("\n[2/3] Generating top-N table ...")
    print_and_save_top_n(df, output_dir, args.top_n)

    print("\n[3/3] Generating sensitivity plots ...")
    plot_sensitivity(df, output_dir, args.dpi)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
