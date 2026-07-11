#!/usr/bin/env python3
"""
plot_postprocessing.py
========================
Generates analytics plots from postprocessing_logs.csv to visualize 
how much artifactual volume was removed during inference.
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
    "GTVp Border": "#FFB300", "GTVn Border": "#FF9800",
    "GTVp Shell": "#EE2C2C", 
    "GTVp Small Obj": "#2196F3", "GTVn Small Obj": "#FF5722",
    "GTVp Total": "#1565C0", "GTVn Total": "#D84315"
}
BOX_KW = dict(showfliers=False, width=0.4)
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
        print(f" File {csv_path} not found.")
        return

    if df.empty:
        print("Dataframe is empty. Skipping plots.")
        return

    sns.set_theme(style="whitegrid", palette="muted")
    print(f"Generating Post-Processing plots → {output_dir} ({len(df)} cases)")

    vol_cols = [
        "border_removed_GTVp_mm3", "border_removed_GTVn_mm3",
        "shell_removed_GTVp_mm3", 
        "small_obj_removed_GTVp_mm3", "small_obj_removed_GTVn_mm3"
    ]

    df_melt = df.melt(
        id_vars=["case_id", "timepoint"], 
        value_vars=vol_cols,
        var_name="Removal_Type", value_name="Volume_mm3"
    )
    
    label_map = {
        "border_removed_GTVp_mm3": "GTVp Border",
        "border_removed_GTVn_mm3": "GTVn Border",
        "shell_removed_GTVp_mm3": "GTVp Shell",
        "small_obj_removed_GTVp_mm3": "GTVp Small Obj",
        "small_obj_removed_GTVn_mm3": "GTVn Small Obj"
    }
    df_melt["Removal_Type"] = df_melt["Removal_Type"].map(label_map)

    # ── 1. Global Distribution of Removed Volumes ──
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Define explicit order to keep plots and labels perfectly synced
    order_vol = df_melt["Removal_Type"].unique()
    
    # 2. Pass 'order=order_vol' to Seaborn
    sns.violinplot(x="Removal_Type", y="Volume_mm3", data=df_melt, palette=PALETTE, inner=None, ax=ax, alpha=0.6, order=order_vol)
    sns.stripplot(x="Removal_Type", y="Volume_mm3", data=df_melt, palette=PALETTE, **STRIP_KW, ax=ax, order=order_vol)
    
    ax.set_title("Distribution of Removed Volumes (mm³) per Case", fontweight="bold", fontsize=13)
    ax.set_ylabel("Removed Volume (mm³)")
    ax.set_xlabel("")
    
    # Add medians as text
    medians = df_melt.groupby("Removal_Type")["Volume_mm3"].median()
    
    # 3. Iterate over 'order_vol' instead of 'medians.index'
    for i, label in enumerate(order_vol):
        ax.text(i, medians[label], f" Med: {medians[label]:.1f} mm³", 
                color="black", ha="left", va="bottom", fontweight="bold", fontsize=9)
    
    _save(fig, os.path.join(output_dir, "PP01_removal_distributions.png"))

    # ── 2. Boxplot (Log Scale for better visibility of outliers) ──
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="Removal_Type", y="Volume_mm3", data=df_melt, palette=PALETTE, **BOX_KW, ax=ax)
    sns.stripplot(x="Removal_Type", y="Volume_mm3", data=df_melt, **STRIP_KW, ax=ax)
    
    ax.set_yscale("symlog", linthresh=10.0) # Handle 0s gracefully
    ax.set_title("Removed Volumes (Log Scale)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Removed Volume (mm³, symlog scale)")
    ax.set_xlabel("")
    _save(fig, os.path.join(output_dir, "PP02_removal_log_scale.png"))

    # ── 3. Timepoint Progression (if temporal data exists) ──
    if "timepoint" in df.columns and df["timepoint"].nunique() > 1:
        tp_order = sorted(df["timepoint"].dropna().unique())
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        for ax, removal_type in zip(axes, label_map.values()):
            sub_df = df_melt[df_melt["Removal_Type"] == removal_type]
            if sub_df.empty: continue
            
            sns.boxplot(x="timepoint", y="Volume_mm3", data=sub_df, order=tp_order, color=PALETTE[removal_type], ax=ax, **BOX_KW)
            sns.stripplot(x="timepoint", y="Volume_mm3", data=sub_df, order=tp_order, **STRIP_KW, ax=ax)
            
            ax.set_title(f"{removal_type} Removed by Timepoint", fontweight="bold")
            ax.set_ylabel("Volume (mm³)")
            ax.set_xlabel("")
            
        fig.suptitle("Post-Processing Impact Across Timepoints", fontsize=14, fontweight="bold")
        _save(fig, os.path.join(output_dir, "PP03_removal_by_timepoint.png"))
    
    # ── 4. Number of Objects Removed ──
    count_cols = [
        "border_removed_GTVp_count", "border_removed_GTVn_count",
        "shell_removed_GTVp_count", 
        "small_obj_removed_GTVp_count", "small_obj_removed_GTVn_count",
        "total_removed_GTVp_count", "total_removed_GTVn_count"
    ]
    if all(c in df.columns for c in count_cols):
        df_counts = df.melt(
            id_vars=["case_id", "timepoint"],
            value_vars=count_cols,
            var_name="Removal_Type", value_name="Count"
        )
        
        count_map = {
            "border_removed_GTVp_count": "GTVp Border",
            "border_removed_GTVn_count": "GTVn Border",
            "shell_removed_GTVp_count": "GTVp Shell",
            "small_obj_removed_GTVp_count": "GTVp Small Obj",
            "small_obj_removed_GTVn_count": "GTVn Small Obj",
            "total_removed_GTVp_count": "GTVp Total",
            "total_removed_GTVn_count": "GTVn Total"
        }
        df_counts["Removal_Type"] = df_counts["Removal_Type"].map(count_map)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 1. Define explicit order for counts
        order_cnt = df_counts["Removal_Type"].unique()
        
        # 2. Pass 'order=order_cnt' to Seaborn
        sns.violinplot(x="Removal_Type", y="Count", data=df_counts, palette=PALETTE, inner=None, ax=ax, alpha=0.6, order=order_cnt)
        sns.stripplot(x="Removal_Type", y="Count", data=df_counts, palette=PALETTE, **STRIP_KW, ax=ax, order=order_cnt)

        ax.set_title("Number of Artifacts Removed per Case", fontweight="bold", fontsize=13)
        ax.set_ylabel("Count of Removed Components")
        ax.set_xlabel("")
        plt.xticks(rotation=30, ha="right")

        # Add medians as text
        medians_cnt = df_counts.groupby("Removal_Type")["Count"].median()
        
        # 3. Iterate over your explicit order
        for i, label in enumerate(order_cnt):
            ax.text(i, medians_cnt[label], f" Med: {medians_cnt[label]:.0f}",
                    color="black", ha="left", va="bottom", fontweight="bold", fontsize=9)

        _save(fig, os.path.join(output_dir, "PP04_removal_counts.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    generate_plots(args.csv_path, args.output_dir)
