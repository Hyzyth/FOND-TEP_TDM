#!/usr/bin/env python3
"""
inference/plot_postprocessing.py
==================================
Generates analytics plots from postprocessing_logs.csv produced by
inference/infer_npz.py, visualising how much artifactual volume was removed
during post-processing.

Plots generated
---------------
  PP01_removal_distributions.png  - violin + strip, volume removed per type
  PP02_removal_log_scale.png      - boxplot on symlog scale
  PP03_removal_by_timepoint.png   - per-timepoint breakdown (TemPoRAL only)
  PP04_removal_counts.png         - number of removed components per type
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

PALETTE = {
    "GTVp Border":    "#FFB300",
    "GTVn Border":    "#FF9800",
    "GTVp Shell":     "#EE2C2C",
    "GTVp Small Obj": "#2196F3",
    "GTVn Small Obj": "#FF5722",
    "GTVp Total":     "#1565C0",
    "GTVn Total":     "#D84315",
}
BOX_KW   = dict(showfliers=False, width=0.4)
STRIP_KW = dict(color="black", alpha=0.5, size=4, jitter=True)


def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {os.path.basename(path)}")


def generate_plots(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  [SKIP] File not found: {csv_path}")
        return

    if df.empty:
        print("  [SKIP] Dataframe is empty.")
        return

    sns.set_theme(style="whitegrid", palette="muted")
    print(f"Generating Post-Processing plots → {output_dir}  ({len(df)} cases)")

    # ── Volume columns ────────────────────────────────────────────────────────
    vol_cols = [
        "border_removed_GTVp_mm3",
        "border_removed_GTVn_mm3",
        "shell_removed_GTVp_mm3",
        "small_obj_removed_GTVp_mm3",
        "small_obj_removed_GTVn_mm3",
    ]
    # Only melt columns that exist (graceful degradation)
    vol_cols = [c for c in vol_cols if c in df.columns]

    label_map = {
        "border_removed_GTVp_mm3":   "GTVp Border",
        "border_removed_GTVn_mm3":   "GTVn Border",
        "shell_removed_GTVp_mm3":    "GTVp Shell",
        "small_obj_removed_GTVp_mm3": "GTVp Small Obj",
        "small_obj_removed_GTVn_mm3": "GTVn Small Obj",
    }

    df_melt = df.melt(
        id_vars=[c for c in ("case_id", "timepoint") if c in df.columns],
        value_vars=vol_cols,
        var_name="Removal_Type",
        value_name="Volume_mm3",
    )
    df_melt["Removal_Type"] = df_melt["Removal_Type"].map(label_map)
    df_melt = df_melt.dropna(subset=["Removal_Type"])

    order_vol = [label_map[c] for c in vol_cols if c in label_map]

    # ── PP01: Distribution of removed volumes ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(x="Removal_Type", y="Volume_mm3", data=df_melt,
                   palette=PALETTE, inner=None, ax=ax, alpha=0.6, order=order_vol)
    sns.stripplot(x="Removal_Type", y="Volume_mm3", data=df_melt,
                  palette=PALETTE, ax=ax, order=order_vol, **STRIP_KW)
    ax.set_title("Distribution of Removed Volumes (mm³) per Case",
                 fontweight="bold", fontsize=13)
    ax.set_ylabel("Removed Volume (mm³)")
    ax.set_xlabel("")
    medians = df_melt.groupby("Removal_Type")["Volume_mm3"].median()
    for i, label in enumerate(order_vol):
        if label in medians.index:
            ax.text(i, medians[label],
                    f" Med: {medians[label]:.1f} mm³",
                    color="black", ha="left", va="bottom",
                    fontweight="bold", fontsize=9)
    _save(fig, os.path.join(output_dir, "PP01_removal_distributions.png"))

    # ── PP02: Log-scale boxplot ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(x="Removal_Type", y="Volume_mm3", data=df_melt,
                palette=PALETTE, ax=ax, order=order_vol, **BOX_KW)
    sns.stripplot(x="Removal_Type", y="Volume_mm3", data=df_melt,
                  ax=ax, order=order_vol, **STRIP_KW)
    ax.set_yscale("symlog", linthresh=10.0)
    ax.set_title("Removed Volumes (Symlog Scale)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Removed Volume (mm³, symlog scale)")
    ax.set_xlabel("")
    _save(fig, os.path.join(output_dir, "PP02_removal_log_scale.png"))

    # ── PP03: Timepoint progression (TemPoRAL data) ───────────────────────────
    tp_col = "timepoint" if "timepoint" in df.columns else None
    if tp_col and df[tp_col].nunique() > 1:
        tp_order = sorted(df[tp_col].dropna().unique())
        n_types  = len(order_vol)
        fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 5), sharey=False)
        if n_types == 1:
            axes = [axes]
        for ax, removal_type in zip(axes, order_vol):
            sub = df_melt[df_melt["Removal_Type"] == removal_type]
            if sub.empty:
                continue
            color = PALETTE.get(removal_type, "#888888")
            sns.boxplot(x=tp_col, y="Volume_mm3", data=sub,
                        order=tp_order, color=color, ax=ax, **BOX_KW)
            sns.stripplot(x=tp_col, y="Volume_mm3", data=sub,
                          order=tp_order, ax=ax, **STRIP_KW)
            ax.set_title(f"{removal_type}", fontweight="bold", fontsize=10)
            ax.set_ylabel("Volume (mm³)")
            ax.set_xlabel("")
        fig.suptitle("Post-Processing Impact Across Timepoints",
                     fontsize=14, fontweight="bold")
        _save(fig, os.path.join(output_dir, "PP03_removal_by_timepoint.png"))

    # ── PP04: Number of components removed ────────────────────────────────────
    count_cols_map = {
        "border_removed_GTVp_count":   "GTVp Border",
        "border_removed_GTVn_count":   "GTVn Border",
        "shell_removed_GTVp_count":    "GTVp Shell",
        "small_obj_removed_GTVp_count": "GTVp Small Obj",
        "small_obj_removed_GTVn_count": "GTVn Small Obj",
        "total_removed_GTVp_count":    "GTVp Total",
        "total_removed_GTVn_count":    "GTVn Total",
    }
    avail_cnt = {k: v for k, v in count_cols_map.items() if k in df.columns}
    if avail_cnt:
        id_vars = [c for c in ("case_id", "timepoint") if c in df.columns]
        df_counts = df.melt(
            id_vars=id_vars,
            value_vars=list(avail_cnt.keys()),
            var_name="Removal_Type",
            value_name="Count",
        )
        df_counts["Removal_Type"] = df_counts["Removal_Type"].map(avail_cnt)
        df_counts = df_counts.dropna(subset=["Removal_Type"])
        order_cnt = [v for v in avail_cnt.values()]

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.violinplot(x="Removal_Type", y="Count", data=df_counts,
                       palette=PALETTE, inner=None, ax=ax, alpha=0.6, order=order_cnt)
        sns.stripplot(x="Removal_Type", y="Count", data=df_counts,
                      palette=PALETTE, ax=ax, order=order_cnt, **STRIP_KW)
        ax.set_title("Number of Artifacts Removed per Case",
                     fontweight="bold", fontsize=13)
        ax.set_ylabel("Count of Removed Components")
        ax.set_xlabel("")
        plt.xticks(rotation=30, ha="right")
        medians_cnt = df_counts.groupby("Removal_Type")["Count"].median()
        for i, label in enumerate(order_cnt):
            if label in medians_cnt.index:
                ax.text(i, medians_cnt[label],
                        f" Med: {medians_cnt[label]:.0f}",
                        color="black", ha="left", va="bottom",
                        fontweight="bold", fontsize=9)
        _save(fig, os.path.join(output_dir, "PP04_removal_counts.png"))

    print(f"\n Post-processing plots saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MedSAM2 post-processing analytics from postprocessing_logs.csv"
    )
    parser.add_argument("--csv_path",   required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    generate_plots(args.csv_path, args.output_dir)
