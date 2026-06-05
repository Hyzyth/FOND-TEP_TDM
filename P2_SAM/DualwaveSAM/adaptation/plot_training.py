"""
plot_training.py  —  DualwaveSAM training log plotter
======================================================

Thin wrapper that re-uses the SwinCross plot_training.py parser verbatim.
The log format written by trainer.py is identical:
  "[Epoch N/M] ... Train loss: X.XXXX  LR: Y.YYeZZ  Epoch Time: T.Ts"
  "Epoch N Validation Dice: X.XXXX ... Val loss: Y.YYYY"
  "Initial Validation Complete | Dice: X.XXXX ... Val loss: Y.YYYY"

Run:
  python {folder}/plot_training.py \\
      --log_dirs ./runs/DualwaveSAM3c_classic \\
      --output_dir ./runs/DualwaveSAM3c_classic/plots \\
      --title "DualwaveSAM 3-class Classic (300 Epochs)"

For k-fold (pass all fold dirs):
  python {folder}/plot_training.py \\
      --log_dirs ./runs/DualwaveSAM3c_kfold_fold0 ... \\
      --output_dir ./runs/DualwaveSAM3c_kfold_ensemble/plots \\
      --title "DualwaveSAM 3-class K-Fold"
"""

import os
import argparse
import re
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", context="paper")


def parse_logs(log_directory: str):
    """
    Scan all .log files in a directory and extract per-epoch metrics.
    Uses a dict so restarted runs overwrite old epochs cleanly.
    """
    # Regex Patterns
    train_pat = re.compile(
        r"\[Epoch (\d+)/\d+\] Loss=([0-9.]+)\s+LR=([0-9.eE+\-]+)\s+Time=([0-9.]+)s"
    )
    
    # Validation start marker
    val_start_pat = re.compile(r"Validation at epoch (\d+)")
    
    # Validation primary metrics
    val_dice_pat = re.compile(r"Val Dice=([0-9.]+).*?Loss=([0-9.]+)")
    
    # Validation per-class metrics
    per_class_pat = re.compile(
        r"↳ Per-Class:\s*BG=([0-9.]+)\s*\|\s*Tumor=([0-9.]+)\s*\|\s*Nodule=([0-9.]+)"
    )
    
    # Epoch 0 (Init) single-line marker
    init_val_pat = re.compile(
        r"Init.*?Dice=([0-9.]+).*?Loss=([0-9.]+).*?BG=([0-9.]+).*?Tumor=([0-9.]+).*?Nodule=([0-9.]+)", re.IGNORECASE
    )

    train_data: dict = {}
    val_data:   dict = {}

    log_files = sorted(
        glob.glob(os.path.join(log_directory, "*.log")),
        key=os.path.getmtime,
    )
    
    if not log_files:
        print(f"  ⚠  No .log files in {log_directory}")
        return pd.DataFrame(), pd.DataFrame()

    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            current_val_epoch = None
            
            for line in f:
                # 1. Parse Training
                m = train_pat.search(line)
                if m:
                    train_data[int(m.group(1))] = {
                        "Train_Loss":    float(m.group(2)),
                        "Learning_Rate": float(m.group(3)),
                        "Time_Seconds":  float(m.group(4)),
                    }
                    continue

                # 2. Parse Initial Validation (Epoch 0)
                m = init_val_pat.search(line)
                if m:
                    val_data[0] = {
                        "Dice":        float(m.group(1)),
                        "Val_Loss":    float(m.group(2)),
                        "BG_Dice":     float(m.group(3)),
                        "Tumor_Dice":  float(m.group(4)),
                        "Nodule_Dice": float(m.group(5)),
                    }
                    continue

                # 3. Parse Regular Validation Start Marker
                m = val_start_pat.search(line)
                if m:
                    current_val_epoch = int(m.group(1))
                    if current_val_epoch not in val_data:
                        val_data[current_val_epoch] = {}
                    continue

                # 4. Parse Regular Validation Metrics (Requires current_val_epoch)
                m = val_dice_pat.search(line)
                if m and current_val_epoch is not None:
                    val_data[current_val_epoch]["Dice"]     = float(m.group(1))
                    val_data[current_val_epoch]["Val_Loss"] = float(m.group(2))
                    continue

                # 5. Parse Regular Validation Per-Class Metrics
                m = per_class_pat.search(line)
                if m and current_val_epoch is not None:
                    val_data[current_val_epoch]["BG_Dice"]     = float(m.group(1))
                    val_data[current_val_epoch]["Tumor_Dice"]  = float(m.group(2))
                    val_data[current_val_epoch]["Nodule_Dice"] = float(m.group(3))
                    continue

    df_train = (
        pd.DataFrame.from_dict(train_data, orient="index")
        .sort_index()
        .reset_index(names="Epoch")
    )
    df_val = (
        pd.DataFrame.from_dict(val_data, orient="index")
        .sort_index()
        .reset_index(names="Epoch")
    )
    return df_train, df_val


def plot_single_run(df_train, df_val, output_dir: str, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    df_train["Smoothed_Loss"] = (
        df_train["Train_Loss"].rolling(window=5, min_periods=1).mean()
    )

    # --- Training & Validation Loss ---
    axes[0, 0].plot(df_train["Epoch"], df_train["Train_Loss"],
                    color="lightblue", alpha=0.55, label="Train Loss (Raw)")
    axes[0, 0].plot(df_train["Epoch"], df_train["Smoothed_Loss"],
                    color="blue", linewidth=2, label="Train Loss (Smoothed)")
    if not df_val.empty and "Val_Loss" in df_val.columns:
        axes[0, 0].plot(df_val["Epoch"], df_val["Val_Loss"],
                        color="red", marker="o", markersize=3, label="Val Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=9)

    # --- Validation Dice Score ---
    if not df_val.empty and "Dice" in df_val.columns:
        # Mean FG
        axes[0, 1].plot(df_val["Epoch"], df_val["Dice"],
                        color="purple", marker="o", markersize=4, linewidth=2,
                        label="Mean Dice (FG)")
        
        # Per-Class Lines
        if "Tumor_Dice" in df_val.columns:
            axes[0, 1].plot(df_val["Epoch"], df_val["Tumor_Dice"],
                            color="red", marker="x", markersize=4, linewidth=1.2, linestyle="--",
                            label="Tumor")
        if "Nodule_Dice" in df_val.columns:
            axes[0, 1].plot(df_val["Epoch"], df_val["Nodule_Dice"],
                            color="orange", marker="s", markersize=3, linewidth=1.2, linestyle="--",
                            label="Nodule")
        if "BG_Dice" in df_val.columns:
            axes[0, 1].plot(df_val["Epoch"], df_val["BG_Dice"],
                            color="gray", alpha=0.6, linewidth=1, linestyle=":",
                            label="Background")

        # Highlight Best Epoch (based on Mean FG Dice)
        best_epoch = df_val.loc[df_val["Dice"].idxmax()]
        axes[0, 1].axvline(best_epoch["Epoch"], color="green", ls="--", lw=1.2,
                           label=f"Best FG: {best_epoch['Dice']:.4f} @ ep{int(best_epoch['Epoch'])}")
        
        axes[0, 1].set_title("Validation Dice Score")
        axes[0, 1].set_ylabel("Dice")
        axes[0, 1].set_ylim([-0.05, 1.05])
        axes[0, 1].legend(fontsize=9)

    # --- Learning Rate ---
    axes[1, 0].plot(df_train["Epoch"], df_train["Learning_Rate"],
                    color="green", linewidth=2)
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_ylabel("Learning Rate")

    # --- Epoch Duration ---
    axes[1, 1].plot(df_train["Epoch"], df_train["Time_Seconds"],
                    color="orange", alpha=0.8)
    mean_t = df_train["Time_Seconds"].mean()
    axes[1, 1].axhline(mean_t, color="red", ls="--",
                       label=f"Mean: {mean_t:.1f}s")
    axes[1, 1].set_title("Epoch Duration")
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    max_ep = int(df_train["Epoch"].max()) if not df_train.empty else 0
    out    = os.path.join(output_dir, f"training_plots_ep{max_ep}.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  Classic plot → {out}")


def plot_kfold_run(train_dfs, val_dfs, output_dir: str, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title} ({len(train_dfs)} Folds)", fontsize=16, fontweight="bold")

    combined_train  = pd.concat(train_dfs)
    mean_train      = combined_train.groupby("Epoch").mean().reset_index()
    mean_train["Smoothed_Loss"] = (
        mean_train["Train_Loss"].rolling(window=5, min_periods=1).mean()
    )

    # --- Training Loss (superimposed) ---
    for df in train_dfs:
        axes[0, 0].plot(df["Epoch"], df["Train_Loss"],
                        color="lightblue", alpha=0.25, linewidth=1)
    axes[0, 0].plot(mean_train["Epoch"], mean_train["Smoothed_Loss"],
                    color="blue", linewidth=2.5, label="Mean Smoothed")
    axes[0, 0].set_title("Training Loss (All Folds)")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=9)

    # --- Validation Dice (superimposed & means) ---
    valid_val = [df for df in val_dfs if not df.empty]
    if valid_val:
        combined_val = pd.concat(valid_val)
        mean_val     = combined_val.groupby("Epoch").mean().reset_index()
        
        # Plot individual fold dots for Mean FG
        for df in valid_val:
            axes[0, 1].plot(df["Epoch"], df["Dice"],
                            color="violet", alpha=0.35, linewidth=0,
                            marker=".", markersize=4)
        
        # Plot Mean FG Line
        axes[0, 1].plot(mean_val["Epoch"], mean_val["Dice"],
                        color="purple", linewidth=2.5, marker="o", markersize=4,
                        label="Mean Dice (FG)")
        
        # Plot Mean Per-Class Lines
        if "Tumor_Dice" in mean_val.columns:
            axes[0, 1].plot(mean_val["Epoch"], mean_val["Tumor_Dice"],
                            color="red", linewidth=1.5, marker="x", markersize=4, linestyle="--",
                            label="Mean Tumor")
        if "Nodule_Dice" in mean_val.columns:
            axes[0, 1].plot(mean_val["Epoch"], mean_val["Nodule_Dice"],
                            color="orange", linewidth=1.5, marker="s", markersize=3, linestyle="--",
                            label="Mean Nodule")
        if "BG_Dice" in mean_val.columns:
            axes[0, 1].plot(mean_val["Epoch"], mean_val["BG_Dice"],
                            color="gray", alpha=0.6, linewidth=1, linestyle=":",
                            label="Mean Background")

        axes[0, 1].set_title("Validation Dice (All Folds)")
        axes[0, 1].set_ylabel("Dice")
        axes[0, 1].set_ylim([-0.05, 1.05])
        axes[0, 1].legend(fontsize=9)

    # --- Learning Rate ---
    axes[1, 0].plot(mean_train["Epoch"], mean_train["Learning_Rate"],
                    color="green", linewidth=2)
    axes[1, 0].set_title("Learning Rate Schedule")
    axes[1, 0].set_ylabel("Learning Rate")

    # --- Epoch Duration ---
    for df in train_dfs:
        axes[1, 1].plot(df["Epoch"], df["Time_Seconds"],
                        color="orange", alpha=0.2, linewidth=1)
    axes[1, 1].plot(mean_train["Epoch"], mean_train["Time_Seconds"],
                    color="darkorange", linewidth=2, label="Mean Time")
    
    global_mean = mean_train["Time_Seconds"].mean()
    axes[1, 1].axhline(global_mean, color="red", ls="--",
                       label=f"Overall mean: {global_mean:.1f}s")
    axes[1, 1].set_title("Epoch Duration")
    axes[1, 1].set_ylabel("Time (s)")
    axes[1, 1].legend(fontsize=9)

    plt.tight_layout()
    max_ep = int(mean_train["Epoch"].max()) if not mean_train.empty else 0
    out    = os.path.join(output_dir, f"kfold_training_plots_ep{max_ep}.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  K-Fold plot → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot DualwaveSAM training logs")
    parser.add_argument("--log_dirs",   nargs="+", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--title",      default="DualwaveSAM 3-class Training")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dfs, val_dfs = [], []
    for d in args.log_dirs:
        if os.path.isdir(d):
            df_t, df_v = parse_logs(d)
            if not df_t.empty:
                train_dfs.append(df_t)
                val_dfs.append(df_v)
        else:
            print(f"  ⚠  Directory not found: {d}")

    if not train_dfs:
        print("No valid training data found.")
    elif len(train_dfs) == 1:
        plot_single_run(train_dfs[0], val_dfs[0], args.output_dir, args.title)
    else:
        plot_kfold_run(train_dfs, val_dfs, args.output_dir, args.title)
