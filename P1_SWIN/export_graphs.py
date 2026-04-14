import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def export_graphs(log_dir, output_folder):
    # Find event files
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "tfevents" in file:
                event_files.append(os.path.join(root, file))

    if not event_files:
        print(f"No event files found in {log_dir}")
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Found {len(event_files)} event files. Processing...")

    for event_file in event_files:
        print(f"Loading {event_file}...")
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # Get all scalar tags (metrics like loss, accuracy, dice)
            tags = ea.Tags()['scalars']
            
            for tag in tags:
                events = ea.Scalars(tag)
                steps = [e.step for e in events]
                values = [e.value for e in events]

                plt.figure(figsize=(10, 6))
                plt.plot(steps, values, label=tag)
                plt.title(f"{tag} over epochs")
                plt.xlabel("Step/Epoch")
                plt.ylabel("Value")
                plt.grid(True)
                plt.legend()
                
                # Sanitize filename
                safe_tag = tag.replace("/", "_").replace(" ", "_")
                filename = f"{safe_tag}.png"
                save_path = os.path.join(output_folder, filename)
                
                plt.savefig(save_path)
                plt.close()
                print(f"Saved plot: {save_path}")

        except Exception as e:
            print(f"Failed to process {event_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TensorBoard logs to PNG images")
    parser.add_argument("--logdir", required=True, help="Path to the folder containing tfevents files")
    parser.add_argument("--output", default="exported_graphs", help="Folder to save PNGs")
    
    args = parser.parse_args()
    export_graphs(args.logdir, args.output)
