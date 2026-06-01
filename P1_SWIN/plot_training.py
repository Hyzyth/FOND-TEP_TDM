import argparse
import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a cleaner, modern aesthetic
sns.set_theme(style="darkgrid", context="paper")

def parse_logs(log_directory):
    """
    Scans a directory for all .log files, sorts them chronologically, 
    and parses them. Uses a dictionary to automatically overwrite 
    epochs that were interrupted and restarted.
    """
    train_pattern = re.compile(r"\[Epoch (\d+)/\d+\] Train loss:\s+([0-9.]+)\s+LR:\s+([0-9.eE+-]+)\s+Epoch Time:\s+([0-9.]+)s")
    val_pattern = re.compile(r"Epoch (\d+) Validation Dice:\s+([0-9.]+).*?Val loss:\s+([0-9.]+)")
    init_val_pattern = re.compile(r"Initial Validation Complete \| Dice:\s+([0-9.]+).*?Val loss:\s+([0-9.]+)")

    train_data = {}
    val_data = {}

    log_files = sorted(glob.glob(os.path.join(log_directory, "*.log")), key=os.path.getmtime)
    
    if not log_files:
        print(f"  ⚠ No .log files found in {log_directory}")
        return pd.DataFrame(), pd.DataFrame()

    for log_path in log_files:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                tm = train_pattern.search(line)
                if tm:
                    train_data[int(tm.group(1))] = {
                        'Train_Loss': float(tm.group(2)),
                        'Learning_Rate': float(tm.group(3)),
                        'Time_Seconds': float(tm.group(4))
                    }
                    continue
                vm = val_pattern.search(line)
                if vm:
                    val_data[int(vm.group(1))] = {
                        'Dice': float(vm.group(2)),
                        'Val_Loss': float(vm.group(3))
                    }
                    continue
                ivm = init_val_pattern.search(line)
                if ivm:
                    val_data[0] = {'Dice': float(ivm.group(1)), 'Val_Loss': float(ivm.group(2))}

    df_train = pd.DataFrame.from_dict(train_data, orient='index').sort_index().reset_index(names='Epoch')
    df_val = pd.DataFrame.from_dict(val_data, orient='index').sort_index().reset_index(names='Epoch')
    
    return df_train, df_val

def plot_single_run(df_train, df_val, output_dir, title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    df_train['Smoothed_Loss'] = df_train['Train_Loss'].rolling(window=5, min_periods=1).mean()

    # 1. Loss
    axes[0,0].plot(df_train['Epoch'], df_train['Train_Loss'], color='lightblue', alpha=0.6, label='Train Loss (Raw)')
    axes[0,0].plot(df_train['Epoch'], df_train['Smoothed_Loss'], color='blue', linewidth=2, label='Train Loss (Smoothed)')
    if not df_val.empty:
        axes[0,0].plot(df_val['Epoch'], df_val['Val_Loss'], color='red', marker='o', label='Val Loss')
    axes[0,0].set_title('Training & Validation Loss')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend(fontsize=10)

    # 2. Dice
    if not df_val.empty:
        axes[0,1].plot(df_val['Epoch'], df_val['Dice'], color='purple', marker='o', linewidth=2, label='Val Dice')
        axes[0,1].set_title('Validation Dice Score')
        axes[0,1].set_ylabel('Dice')
        axes[0,1].legend(fontsize=10)
        axes[0,1].set_ylim([-0.05, 1.05])

    # 3. LR
    axes[1,0].plot(df_train['Epoch'], df_train['Learning_Rate'], color='green', linewidth=2)
    axes[1,0].set_title('Learning Rate Schedule')
    axes[1,0].set_ylabel('Learning Rate')

    # 4. Duration
    axes[1,1].plot(df_train['Epoch'], df_train['Time_Seconds'], color='orange', alpha=0.8)
    mean_time = df_train['Time_Seconds'].mean()
    axes[1,1].axhline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.1f}s')
    axes[1,1].set_title('Epoch Duration')
    axes[1,1].set_ylabel('Time (s)')
    axes[1,1].legend(fontsize=10)

    plt.tight_layout()
    max_epoch = int(df_train['Epoch'].max()) if not df_train.empty else 0
    out_filename = os.path.join(output_dir, f'training_plots_ep{max_epoch}.png')
    plt.savefig(out_filename, dpi=300)
    print(f"   Classic plot generated: {out_filename}")

def plot_kfold_run(train_dfs, val_dfs, output_dir, title):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title} ({len(train_dfs)} Folds)", fontsize=16, fontweight='bold')

    # Combine data for mean calculations
    combined_train = pd.concat(train_dfs)
    mean_train = combined_train.groupby('Epoch').mean().reset_index()
    mean_train['Smoothed_Loss'] = mean_train['Train_Loss'].rolling(window=5, min_periods=1).mean()

    # 1. Train Loss (Superimposed)
    for df in train_dfs:
        axes[0,0].plot(df['Epoch'], df['Train_Loss'], color='lightblue', alpha=0.3, linewidth=1)
    axes[0,0].plot(mean_train['Epoch'], mean_train['Smoothed_Loss'], color='blue', linewidth=2.5, label='Mean Smoothed Loss')
    axes[0,0].set_title('Training Loss (All Folds)')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend(fontsize=10)

    # 2. Val Dice (Superimposed)
    valid_val_dfs = [df for df in val_dfs if not df.empty]
    if valid_val_dfs:
        combined_val = pd.concat(valid_val_dfs)
        mean_val = combined_val.groupby('Epoch').mean().reset_index()
        for df in valid_val_dfs:
            axes[0,1].plot(df['Epoch'], df['Dice'], color='violet', alpha=0.4, linewidth=1, marker='.', markersize=4)
        axes[0,1].plot(mean_val['Epoch'], mean_val['Dice'], color='purple', linewidth=2.5, marker='o', label='Mean Val Dice')
        axes[0,1].set_title('Validation Dice Score (All Folds)')
        axes[0,1].set_ylabel('Dice')
        axes[0,1].legend(fontsize=10)
        axes[0,1].set_ylim([-0.05, 1.05])

    # 3. LR (Mean)
    axes[1,0].plot(mean_train['Epoch'], mean_train['Learning_Rate'], color='green', linewidth=2)
    axes[1,0].set_title('Learning Rate Schedule')
    axes[1,0].set_ylabel('Learning Rate')

    # 4. Duration (Superimposed)
    for df in train_dfs:
        axes[1,1].plot(df['Epoch'], df['Time_Seconds'], color='orange', alpha=0.2, linewidth=1)
    axes[1,1].plot(mean_train['Epoch'], mean_train['Time_Seconds'], color='darkorange', linewidth=2, label='Mean Duration')
    mean_time_global = mean_train['Time_Seconds'].mean()
    axes[1,1].axhline(mean_time_global, color='red', linestyle='--', label=f'Overall Mean: {mean_time_global:.1f}s')
    axes[1,1].set_title('Epoch Duration')
    axes[1,1].set_ylabel('Time (s)')
    axes[1,1].legend(fontsize=10)

    plt.tight_layout()
    max_epoch = int(mean_train['Epoch'].max()) if not mean_train.empty else 0
    out_filename = os.path.join(output_dir, f'kfold_training_plots_ep{max_epoch}.png')
    plt.savefig(out_filename, dpi=300)
    print(f"   K-Fold plot generated: {out_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training logs handling restarts and k-folds.")
    parser.add_argument("--log_dirs", nargs='+', required=True, help="One or more directories containing .log files")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save the plot")
    parser.add_argument("--title", type=str, default="Training Metrics", help="Title for the plot")
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
            print(f"  ⚠ Directory not found: {d}")

    if not train_dfs:
        print(" No valid training data found across provided directories.")
    elif len(train_dfs) == 1:
        plot_single_run(train_dfs[0], val_dfs[0], args.output_dir, args.title)
    else:
        plot_kfold_run(train_dfs, val_dfs, args.output_dir, args.title)
