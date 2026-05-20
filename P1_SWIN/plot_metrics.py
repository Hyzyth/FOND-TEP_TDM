#!/usr/bin/env python3
"""
plot_metrics.py
===============
Generates a rich plot suite from per_case_evaluation_rich.csv.

Plots generated:
  1.  Dice per label              (box + strip)
  2.  Jaccard per label           (box + strip)
  3.  Hausdorff per label         (box + strip, mm)
  4.  MHD per label               (box + strip, mm)
  5.  Overlap on GT per label     (box + strip)
  6.  Overlap on Pred per label   (box + strip)
  7.  Volume correlation          (scatter, GT vs Pred, per label)
  8.  Volume difference           (GT − Pred, violin + swarm, per label)
  9.  Volume similarity           (bar + strip, per label)
  10. Object count difference     (violin + swarm, per label)
  11. Dice vs Volume GT           (scatter, per label, coloured by Jaccard)
  12. Timepoint progression       (box + strip, if ≥2 timepoints)
  13. Summary metrics heatmap     (mean per timepoint, if available)
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Palette / style ───────────────────────────────────────────────────────────
PALETTE  = {"GTVp": "#2196F3", "GTVn": "#FF5722"}
STRIP_KW = dict(color="black", alpha=0.45, size=3, jitter=True)
BOX_KW   = dict(showfliers=False, width=0.45)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ {os.path.basename(path)}")


def _per_label_boxplot(df: pd.DataFrame, cols: dict, title: str,
                       ylabel: str, output_path: str,
                       ylim=None, log_scale=False):
    """
    Generic helper: one subplot per label with box + strip.

    cols = {"GTVp": "GTVp_dice", "GTVn": "GTVn_dice"}
    """
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5), sharey=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, (label, col) in zip(axes, cols.items()):
        data = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
        if data.empty:
            ax.set_title(f"{label} (no data)")
            continue
        color = PALETTE.get(label, "#888888")
        sns.boxplot(y=data, ax=ax, color=color, **BOX_KW)
        sns.stripplot(y=data, ax=ax, **STRIP_KW)
        ax.set_title(label)
        ax.set_ylabel(ylabel if ax == axes[0] else "")
        ax.set_xlabel("")
        if ylim:
            ax.set_ylim(*ylim)
        if log_scale:
            ax.set_yscale("log")
        # Annotate median
        med = data.median()
        ax.axhline(med, color=color, lw=1.5, ls="--", alpha=0.7,
                   label=f"Median: {med:.3f}")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    _save(fig, output_path)


# ── Main plotting function ────────────────────────────────────────────────────

def generate_plots(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File {csv_path} not found. Skipping plotting.")
        return

    # Keep only rows with GT annotation
    if "gt_available" in df.columns:
        df = df[df["gt_available"].astype(str).str.lower().isin(["true", "1"])]

    if df.empty:
        print(f"No valid cases with ground truth in {csv_path}. Skipping.")
        return

    # ── Numeric coercion ──────────────────────────────────────────────────────
    numeric_cols = [
        "GTVp_dice", "GTVn_dice", "mean_dice",
        "GTVp_jaccard", "GTVn_jaccard",
        "GTVp_hausdorff_mm", "GTVn_hausdorff_mm",
        "GTVp_mhd_mm", "GTVn_mhd_mm",
        "GTVp_overlap_gt", "GTVn_overlap_gt",
        "GTVp_overlap_pred", "GTVn_overlap_pred",
        "gt_vol_GTVp_mm3", "gt_vol_GTVn_mm3",
        "pred_vol_GTVp_mm3", "pred_vol_GTVn_mm3",
        "gt_count_GTVp", "pred_count_GTVp",
        "gt_count_GTVn", "pred_count_GTVn",
        "vol_sim_GTVp", "vol_sim_GTVn",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sns.set_theme(style="whitegrid", palette="muted")
    print(f"Generating plots → {output_dir}  ({len(df)} cases)")

    # ── 00. Overview: all scalar metrics, GTVp vs GTVn side-by-side ───────────
    # _combined_label_boxplot puts both labels on the same axes (melted),
    # giving a quick single-glance comparison across the key metrics.
    overview_metrics = [
        ("Dice Score",          "GTVp_dice",         "GTVn_dice",         (-0.05, 1.05)),
        ("Jaccard (IoU)",       "GTVp_jaccard",      "GTVn_jaccard",      (-0.05, 1.05)),
        ("Overlap on GT",       "GTVp_overlap_gt",   "GTVn_overlap_gt",   (-0.05, 1.05)),
        ("Overlap on Pred",     "GTVp_overlap_pred", "GTVn_overlap_pred", (-0.05, 1.05)),
        ("Volume Similarity",   "vol_sim_GTVp",      "vol_sim_GTVn",      (-0.05, 1.05)),
    ]
    n_ov = len(overview_metrics)
    fig_ov, axes_ov = plt.subplots(1, n_ov, figsize=(4.5 * n_ov, 5), sharey=False)
    for ax_ov, (ylabel, col_p, col_n, ylim) in zip(axes_ov, overview_metrics):
        frames = []
        for label, col in (("GTVp", col_p), ("GTVn", col_n)):
            if col in df.columns:
                tmp = df[[col]].rename(columns={col: "value"}).copy()
                tmp["Label"] = label
                frames.append(tmp)
        if not frames:
            continue
        melted = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
        sns.boxplot(x="Label", y="value", data=melted, ax=ax_ov,
                    palette=PALETTE, **BOX_KW)
        sns.stripplot(x="Label", y="value", data=melted, ax=ax_ov, **STRIP_KW)
        ax_ov.set_title(ylabel, fontweight="bold", fontsize=10)
        ax_ov.set_xlabel("")
        ax_ov.set_ylabel("")
        if ylim:
            ax_ov.set_ylim(*ylim)
    fig_ov.suptitle("Scalar Metrics Overview — GTVp vs GTVn",
                    fontsize=13, fontweight="bold")
    _save(fig_ov, os.path.join(output_dir, "00_overview_scalar_metrics.png"))

    # ── 00b. Overview: surface-distance metrics (separate y-scales) ───────────
    dist_metrics = [
        ("Hausdorff (mm)", "GTVp_hausdorff_mm", "GTVn_hausdorff_mm"),
        ("MHD / ASSD (mm)", "GTVp_mhd_mm",       "GTVn_mhd_mm"),
    ]
    fig_d, axes_d = plt.subplots(1, len(dist_metrics),
                                  figsize=(6 * len(dist_metrics), 5))
    for ax_d, (ylabel, col_p, col_n) in zip(axes_d, dist_metrics):
        frames = []
        for label, col in (("GTVp", col_p), ("GTVn", col_n)):
            if col in df.columns:
                tmp = df[[col]].rename(columns={col: "value"}).copy()
                tmp["Label"] = label
                frames.append(tmp)
        if not frames:
            continue
        melted = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
        sns.boxplot(x="Label", y="value", data=melted, ax=ax_d,
                    palette=PALETTE, **BOX_KW)
        sns.stripplot(x="Label", y="value", data=melted, ax=ax_d, **STRIP_KW)
        ax_d.set_title(ylabel, fontweight="bold")
        ax_d.set_xlabel("")
        ax_d.set_ylabel(ylabel)
    fig_d.suptitle("Surface Distance Metrics Overview — GTVp vs GTVn",
                   fontsize=13, fontweight="bold")
    _save(fig_d, os.path.join(output_dir, "00b_overview_surface_distances.png"))

    # ── 1. Dice per label ─────────────────────────────────────────────────────
    _per_label_boxplot(
        df,
        cols={"GTVp": "GTVp_dice", "GTVn": "GTVn_dice"},
        title="Dice Score per Label",
        ylabel="Dice Score",
        output_path=os.path.join(output_dir, "01_dice_per_label.png"),
        ylim=(-0.05, 1.05),
    )

    # ── 2. Jaccard per label ──────────────────────────────────────────────────
    _per_label_boxplot(
        df,
        cols={"GTVp": "GTVp_jaccard", "GTVn": "GTVn_jaccard"},
        title="Jaccard Index (IoU) per Label",
        ylabel="Jaccard",
        output_path=os.path.join(output_dir, "02_jaccard_per_label.png"),
        ylim=(-0.05, 1.05),
    )

    # ── 3. Hausdorff per label ────────────────────────────────────────────────
    _per_label_boxplot(
        df,
        cols={"GTVp": "GTVp_hausdorff_mm", "GTVn": "GTVn_hausdorff_mm"},
        title="Hausdorff Distance per Label (mm)",
        ylabel="Hausdorff Distance (mm)",
        output_path=os.path.join(output_dir, "03_hausdorff_per_label.png"),
    )

    # ── 4. MHD per label ─────────────────────────────────────────────────────
    _per_label_boxplot(
        df,
        cols={"GTVp": "GTVp_mhd_mm", "GTVn": "GTVn_mhd_mm"},
        title="Mean Hausdorff Distance (ASSD) per Label (mm)",
        ylabel="MHD / ASSD (mm)",
        output_path=os.path.join(output_dir, "04_mhd_per_label.png"),
    )

    # ── 5. Overlap on GT per label ────────────────────────────────────────────
    _per_label_boxplot(
        df,
        cols={"GTVp": "GTVp_overlap_gt", "GTVn": "GTVn_overlap_gt"},
        title="Overlap on GT per Label  (TP / |GT|  = Recall)",
        ylabel="Overlap on GT",
        output_path=os.path.join(output_dir, "05_overlap_gt_per_label.png"),
        ylim=(-0.05, 1.05),
    )

    # ── 6. Overlap on Pred per label ──────────────────────────────────────────
    _per_label_boxplot(
        df,
        cols={"GTVp": "GTVp_overlap_pred", "GTVn": "GTVn_overlap_pred"},
        title="Overlap on Prediction per Label  (TP / |Pred|  = Precision)",
        ylabel="Overlap on Pred",
        output_path=os.path.join(output_dir, "06_overlap_pred_per_label.png"),
        ylim=(-0.05, 1.05),
    )

    # ── 7. Volume correlation (scatter, GT vs Pred) ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (cls_name, color) in zip(axes, PALETTE.items()):
        gt_col   = f"gt_vol_{cls_name}_mm3"
        pred_col = f"pred_vol_{cls_name}_mm3"
        if gt_col not in df.columns or pred_col not in df.columns:
            continue
        valid = df[[gt_col, pred_col]].dropna()
        if valid.empty:
            continue
        sns.scatterplot(x=gt_col, y=pred_col, data=valid, ax=ax,
                        color=color, alpha=0.7, edgecolor="white", s=60)
        max_val = max(valid[gt_col].max(), valid[pred_col].max())
        ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="y = x (ideal)")
        ax.set_title(f"{cls_name} — Volume Correlation (mm³)", fontweight="bold")
        ax.set_xlabel("GT Volume (mm³)")
        ax.set_ylabel("Predicted Volume (mm³)")
        ax.legend()
    fig.suptitle("Predicted vs Ground-Truth Volume", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(output_dir, "07_volume_correlation.png"))

    # ── 8. Volume difference (GT − Pred) ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (cls_name, color) in zip(axes, PALETTE.items()):
        gt_col   = f"gt_vol_{cls_name}_mm3"
        pred_col = f"pred_vol_{cls_name}_mm3"
        if gt_col not in df.columns or pred_col not in df.columns:
            continue
        diff = (df[gt_col] - df[pred_col]).dropna()
        if diff.empty:
            continue
        sns.violinplot(y=diff, ax=ax, color=color, inner=None, alpha=0.5)
        sns.swarmplot(y=diff, ax=ax, color="black", size=3, alpha=0.6)
        ax.axhline(0, color="red", lw=1.5, ls="--", label="No difference")
        ax.set_title(f"{cls_name} — Volume Difference (GT − Pred)",
                     fontweight="bold")
        ax.set_ylabel("Volume Difference (mm³)")
        ax.legend()
    fig.suptitle("Volume Difference per Label (GT − Predicted)",
                 fontsize=13, fontweight="bold")
    _save(fig, os.path.join(output_dir, "08_volume_difference.png"))

    # ── 9. Volume similarity per label ────────────────────────────────────────
    _per_label_boxplot(
        df,
        cols={"GTVp": "vol_sim_GTVp", "GTVn": "vol_sim_GTVn"},
        title="Volume Similarity per Label  [1 − |Va−Vb| / (Va+Vb)]",
        ylabel="Volume Similarity",
        output_path=os.path.join(output_dir, "09_volume_similarity.png"),
        ylim=(-0.05, 1.05),
    )

    # ── 10. Object count difference (Pred − GT) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (cls_name, color) in zip(axes, PALETTE.items()):
        gt_col   = f"gt_count_{cls_name}"
        pred_col = f"pred_count_{cls_name}"
        if gt_col not in df.columns or pred_col not in df.columns:
            continue
        diff = (df[pred_col] - df[gt_col]).dropna()
        if diff.empty:
            continue
        sns.violinplot(y=diff, ax=ax, color=color, inner=None, alpha=0.5)
        sns.swarmplot(y=diff, ax=ax, color="black", size=4)
        ax.axhline(0, color="red", lw=1.5, ls="--")
        ax.set_title(f"{cls_name} — Object Count Diff (Pred − GT)",
                     fontweight="bold")
        ax.set_ylabel("Δ Connected Components")
    fig.suptitle("Object Count Difference per Label",
                 fontsize=13, fontweight="bold")
    _save(fig, os.path.join(output_dir, "10_object_count_difference.png"))

    # ── 11. Dice vs GT volume (scatter, coloured by Jaccard) ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cls_name in zip(axes, ("GTVp", "GTVn")):
        dice_col   = f"{cls_name}_dice"
        vol_col    = f"gt_vol_{cls_name}_mm3"
        jac_col    = f"{cls_name}_jaccard"
        if not all(c in df.columns for c in (dice_col, vol_col, jac_col)):
            continue
        sub = df[[dice_col, vol_col, jac_col]].dropna()
        if sub.empty:
            continue
        sc = ax.scatter(sub[vol_col], sub[dice_col], c=sub[jac_col],
                        cmap="viridis", alpha=0.75, edgecolors="white", s=60,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Jaccard")
        ax.set_title(f"{cls_name} — Dice vs GT Volume", fontweight="bold")
        ax.set_xlabel("GT Volume (mm³)")
        ax.set_ylabel("Dice Score")
        ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Dice Score vs Ground-Truth Volume (coloured by Jaccard)",
                 fontsize=13, fontweight="bold")
    _save(fig, os.path.join(output_dir, "11_dice_vs_volume.png"))

    # ── 12. Recall vs Precision per label ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cls_name in zip(axes, ("GTVp", "GTVn")):
        ov_gt_col   = f"{cls_name}_overlap_gt"
        ov_pred_col = f"{cls_name}_overlap_pred"
        dice_col    = f"{cls_name}_dice"
        if not all(c in df.columns for c in (ov_gt_col, ov_pred_col)):
            continue
        sub = df[[ov_gt_col, ov_pred_col, dice_col]].dropna()
        if sub.empty:
            continue
        sc = ax.scatter(sub[ov_gt_col], sub[ov_pred_col],
                        c=sub[dice_col] if dice_col in sub else "steelblue",
                        cmap="plasma", alpha=0.75, edgecolors="white", s=60,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="Dice")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{cls_name} — Overlap on GT vs Overlap on Pred",
                     fontweight="bold")
        ax.set_xlabel("Overlap on GT  (Recall)")
        ax.set_ylabel("Overlap on Pred  (Precision)")
        # F1 iso-lines
        for f1 in (0.3, 0.5, 0.7, 0.9):
            r_vals = np.linspace(1e-6, 1.0, 200)
            p_vals = f1 * r_vals / (2 * r_vals - f1)
            valid  = (p_vals > 0) & (p_vals <= 1) & (r_vals > 0) & (r_vals <= 1)
            ax.plot(r_vals[valid], p_vals[valid], "k--", lw=0.7, alpha=0.4)
            ax.text(0.97, f1 / (2 - f1) + 0.02, f"F1={f1}", fontsize=7,
                    color="gray", ha="right")
    fig.suptitle("Overlap on GT vs Overlap on Pred  (F1 iso-curves dashed)",
                 fontsize=13, fontweight="bold")
    _save(fig, os.path.join(output_dir, "12_overlap_gt_vs_pred.png"))

    # ── 13. Timepoint progression ─────────────────────────────────────────────
    if "timepoint" in df.columns and df["timepoint"].nunique() > 1:
        tp_order = sorted(df["timepoint"].dropna().unique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (cls_name, col) in zip(
                axes, (("GTVp", "GTVp_dice"), ("GTVn", "GTVn_dice"))):
            if col not in df.columns:
                continue
            sns.boxplot(x="timepoint", y=col, data=df, order=tp_order,
                        ax=ax, color=PALETTE[cls_name], **BOX_KW)
            sns.stripplot(x="timepoint", y=col, data=df, order=tp_order,
                          ax=ax, **STRIP_KW)
            ax.set_title(f"{cls_name} Dice across Timepoints", fontweight="bold")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel("Dice")
        fig.suptitle("Dice Score by Timepoint", fontsize=13, fontweight="bold")
        _save(fig, os.path.join(output_dir, "13_dice_by_timepoint.png"))

        # ── 14. Summary heatmap (mean per timepoint) ──────────────────────────
        heat_cols = {
            "Dice GTVp": "GTVp_dice", "Dice GTVn": "GTVn_dice",
            "Jaccard GTVp": "GTVp_jaccard", "Jaccard GTVn": "GTVn_jaccard",
            "MHD GTVp": "GTVp_mhd_mm", "MHD GTVn": "GTVn_mhd_mm",
            "OvGT GTVp": "GTVp_overlap_gt", "OvGT GTVn": "GTVn_overlap_gt",
            "OvPred GTVp": "GTVp_overlap_pred", "OvPred GTVn": "GTVn_overlap_pred",
        }
        available = {k: v for k, v in heat_cols.items() if v in df.columns}
        if available:
            heat_df = (
                df.groupby("timepoint")[[*available.values()]]
                .mean()
                .reindex(tp_order)
                .rename(columns={v: k for k, v in available.items()})
            )
            fig, ax = plt.subplots(figsize=(max(10, len(available) * 1.2),
                                            max(4, len(tp_order) * 0.9)))
            sns.heatmap(heat_df, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                        vmin=0, vmax=1, linewidths=0.5,
                        cbar_kws={"label": "Mean metric value"})
            ax.set_title("Mean Metrics per Timepoint", fontsize=13, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Timepoint")
            _save(fig, os.path.join(output_dir, "14_heatmap_by_timepoint.png"))

    print(f"\n✅ All plots saved in {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate rich metric plots from evaluation CSV")
    parser.add_argument("--csv_path",   required=True,
                        help="Path to per_case_evaluation_rich.csv")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save plots")
    args = parser.parse_args()
    generate_plots(args.csv_path, args.output_dir)
