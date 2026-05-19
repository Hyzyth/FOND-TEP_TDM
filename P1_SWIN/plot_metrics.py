#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plots(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File {csv_path} not found. Skipping plotting.")
        return

    # Filter out cases with no ground truth
    df = df[df['gt_available'] == True]
    if df.empty:
        print(f"No valid cases with ground truth in {csv_path}. Skipping.")
        return

    # Convert numeric columns safely
    numeric_cols = [
        'GTVp_dice', 'GTVn_dice', 'mean_dice',
        'gt_vol_GTVp_mm3', 'gt_vol_GTVn_mm3', 
        'pred_vol_GTVp_mm3', 'pred_vol_GTVn_mm3',
        'GTVp_sensitivity', 'GTVp_precision',
        'GTVn_sensitivity', 'GTVn_precision',
        'gt_count_GTVp', 'pred_count_GTVp',
        'gt_count_GTVn', 'pred_count_GTVn',
        'vol_sim_GTVp', 'vol_sim_GTVn'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    sns.set_theme(style="whitegrid", palette="muted")
    
    print(f"Generating plots for {csv_path} -> {output_dir}")

    # --- Plot 1: Dice Score Distributions (Box + Stripplot) ---
    plt.figure(figsize=(10, 6))
    dice_df = df[['GTVp_dice', 'GTVn_dice', 'mean_dice']].dropna(how='all')
    if not dice_df.empty:
        dice_melted = dice_df.melt(var_name='Metric', value_name='Dice Score')
        ax = sns.boxplot(x='Metric', y='Dice Score', data=dice_melted, showfliers=False, width=0.5, boxprops={'alpha': 0.6})
        sns.stripplot(x='Metric', y='Dice Score', data=dice_melted, color='black', alpha=0.5, jitter=True, ax=ax)
        plt.title('Dice Score Distributions (GTVp, GTVn, Mean)')
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dice_distributions.png'), dpi=300)
    plt.close()

    # --- Plot 2: Volume Correlation (Scatter plot with y=x line) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cls_name in zip(axes, ['GTVp', 'GTVn']):
        gt_col, pred_col = f'gt_vol_{cls_name}_mm3', f'pred_vol_{cls_name}_mm3'
        if gt_col in df.columns and pred_col in df.columns:
            valid_vols = df[[gt_col, pred_col]].dropna()
            if not valid_vols.empty:
                sns.scatterplot(x=gt_col, y=pred_col, data=valid_vols, ax=ax, alpha=0.7)
                # Identity line
                max_val = max(valid_vols[gt_col].max(), valid_vols[pred_col].max())
                if pd.notna(max_val) and max_val > 0:
                    ax.plot([0, max_val], [0, max_val], 'r--', label='Ideal (y=x)')
                ax.set_title(f'{cls_name} Volume Correlation (mm³)')
                ax.set_xlabel('Ground Truth Volume (mm³)')
                ax.set_ylabel('Predicted Volume (mm³)')
                ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_correlation.png'), dpi=300)
    plt.close()

    # --- Plot 3: Sensitivity vs Precision (Joint/Scatter Plot) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cls_name in zip(axes, ['GTVp', 'GTVn']):
        sens_col, prec_col = f'{cls_name}_sensitivity', f'{cls_name}_precision'
        if sens_col in df.columns and prec_col in df.columns:
            sns.scatterplot(x=sens_col, y=prec_col, data=df, ax=ax, hue='mean_dice', palette='viridis', size='mean_dice', sizes=(20, 200))
            ax.set_title(f'{cls_name}: Sensitivity vs Precision')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Sensitivity (Recall)')
            ax.set_ylabel('Precision (PPV)')
            ax.legend(title='Mean Dice', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_vs_precision.png'), dpi=300)
    plt.close()

    # --- Plot 4: Object Counts Error Distribution (Violin/Swarm) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cls_name in zip(axes, ['GTVp', 'GTVn']):
        gt_count_col, pred_count_col = f'gt_count_{cls_name}', f'pred_count_{cls_name}'
        if gt_count_col in df.columns and pred_count_col in df.columns:
            df[f'{cls_name}_count_diff'] = df[pred_count_col] - df[gt_count_col]
            sns.violinplot(y=df[f'{cls_name}_count_diff'].dropna(), ax=ax, inner=None, color=".8")
            sns.swarmplot(y=df[f'{cls_name}_count_diff'].dropna(), ax=ax, size=5)
            ax.set_title(f'{cls_name} Object Count Difference (Pred - GT)')
            ax.set_ylabel('Difference in number of connected objects')
            ax.axhline(0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'object_counts_difference.png'), dpi=300)
    plt.close()

    # --- Plot 5: Timepoint progression (if available) ---
    if 'timepoint' in df.columns and df['timepoint'].nunique() > 1:
        plt.figure(figsize=(12, 6))
        # Ensure it's treated as categorical if it's text/discrete
        sns.boxplot(x='timepoint', y='mean_dice', data=df, order=sorted(df['timepoint'].dropna().unique()))
        sns.stripplot(x='timepoint', y='mean_dice', data=df, color='black', alpha=0.5, order=sorted(df['timepoint'].dropna().unique()))
        plt.title('Mean Dice Scores across Timepoints')
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dice_by_timepoint.png'), dpi=300)
        plt.close()

    print(f"✅ Plots successfully saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rich plots from evaluation CSV")
    parser.add_argument("--csv_path", required=True, help="Path to per_case_evaluation_rich.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots")
    args = parser.parse_args()
    
    generate_plots(args.csv_path, args.output_dir)
