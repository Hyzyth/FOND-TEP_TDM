import re
import os
import pandas as pd
import matplotlib.pyplot as plt

log_path = "/data/ethan/SwinCross/HECKTOR_run_1000_epoch/training_from_scratch.log"

# Regex patterns to parse the logs (using \s+ to safely handle terminal spaces/formatting)
train_pattern = re.compile(r"\[Epoch (\d+)/\d+\] Train loss:\s+([0-9.]+)\s+LR:\s+([0-9.eE+-]+)\s+Epoch Time:\s+([0-9.]+)s")
val_pattern = re.compile(r"Epoch (\d+) Validation Dice:\s+([0-9.]+).*?Val loss:\s+([0-9.]+)")
init_val_pattern = re.compile(r"Initial Validation Complete \| Dice:\s+([0-9.]+).*?Val loss:\s+([0-9.]+)")

history = []
val_history = []

# Parse the log file
if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Check for Train metrics
            tm = train_pattern.search(line)
            if tm:
                history.append([int(tm.group(1)), float(tm.group(2)), float(tm.group(3)), float(tm.group(4))])
                continue
            
            # 2. Check for Validation metrics
            vm = val_pattern.search(line)
            if vm:
                val_history.append([int(vm.group(1)), float(vm.group(2)), float(vm.group(3))])
                continue
            
            # 3. Check for Initial Zero-Shot Validation (mapped to Epoch 0)
            ivm = init_val_pattern.search(line)
            if ivm:
                val_history.append([0, float(ivm.group(1)), float(ivm.group(2))])
                continue
else:
    print(f"Error: Log file not found at {log_path}")
    exit(1)

# Create DataFrames
df = pd.DataFrame(history, columns=['Epoch', 'Train_Loss', 'Learning_Rate', 'Time_Seconds'])
val_df = pd.DataFrame(val_history, columns=['Epoch', 'Dice', 'Val_Loss'])

if df.empty:
    print("Warning: No training data parsed. Check log file contents.")
    exit(1)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
df['Smoothed_Loss'] = df['Train_Loss'].rolling(window=5, min_periods=1).mean()

# Loss Plot
axes[0,0].plot(df['Epoch'], df['Train_Loss'], 'b-', marker='.', alpha=0.3, label='Train Loss (Raw)')
axes[0,0].plot(df['Epoch'], df['Smoothed_Loss'], 'b-', linewidth=2, label='Train Loss (Smoothed)')
if not val_df.empty:
    axes[0,0].plot(val_df['Epoch'], val_df['Val_Loss'], 'r-o', markersize=8, label='Val Loss')
axes[0,0].set_title('Training & Validation Loss', fontsize=14)
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True)

# Dice Score Plot
if not val_df.empty:
    axes[0,1].plot(val_df['Epoch'], val_df['Dice'], 'm-o', markersize=8, label='Val Dice')
    axes[0,1].set_title('Validation Dice Score', fontsize=14)
    axes[0,1].set_ylabel('Dice')
    axes[0,1].legend()
    axes[0,1].grid(True)
    # Autoscale x-axis based on max epoch found in val
    axes[0,1].set_xlim(left=0, right=max(100, df['Epoch'].max() + 10))

# Learning Rate Plot
axes[1,0].plot(df['Epoch'], df['Learning_Rate'], 'g-', marker='.')
axes[1,0].set_title('Learning Rate Schedule', fontsize=14)
axes[1,0].set_ylabel('Learning Rate')
axes[1,0].legend()
axes[1,0].grid(True)

# Epoch Duration Plot
axes[1,1].plot(df['Epoch'], df['Time_Seconds'], 'purple', marker='.', linestyle='-')
mean_time = df['Time_Seconds'].mean()
axes[1,1].axhline(mean_time, color='orange', linestyle='--', label=f'Mean: {mean_time:.1f}s')
axes[1,1].set_title('Epoch Duration', fontsize=14)
axes[1,1].set_ylabel('Time (s)')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()

# Save dynamically based on current max epoch parsed
max_epoch = int(df['Epoch'].max())
out_filename = f'training_plots_ep{max_epoch}.png'
plt.savefig(out_filename, dpi=300)
print(f"✅ Successfully generated {out_filename}")
# plt.show() # Commented out for headless server environments. Uncomment if running locally.
