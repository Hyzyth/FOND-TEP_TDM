import os
import argparse
import matplotlib.pyplot as plt
import numpy.ma as M
from numpy import nan  
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_unique_filename(base_name, extension):
    """
    Generates a unique filename by appending a number if the file already exists.

    Args:
        base_name (str): Desired file name without extension.
        extension (str): File extension (e.g., ".nii.gz", ".csv", ".png").

    Returns:
        str: A unique file name.
    """
    filename = f"{base_name}{extension}"
    counter = 1

    while os.path.exists(filename):
        filename = f"{base_name}({counter}){extension}"
        counter += 1

    return filename

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
                save_path = get_unique_filename(os.path.join(output_folder, f"{safe_tag}"), ".png")

                
                plt.savefig(save_path)
                plt.close()
                print(f"Saved plot: {save_path}")

        except Exception as e:
            print(f"Failed to process {event_file}: {e}")


def export_continuous_graphs(log_dir, output_folder):
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

    dico = {}

    for event_file in event_files:
        print(f"Loading {event_file}...")
        try:
            ea = EventAccumulator(event_file)
            ea.Reload()
            
            # Get all scalar tags (metrics like loss, accuracy, dice)
            tags = ea.Tags()['scalars']
            
            for tag in tags: 
                events = ea.Scalars(tag)
                
                # Sanitize filename
                safe_tag = tag.replace("/", "_").replace(" ", "_")
                if not dico.get(safe_tag):
                    dico[safe_tag] = [e.value for e in events]
                    dico[f"{safe_tag}_steps"] = [e.step for e in events]
                else:
                    # Insert NaN to break the line between segments,
                    # avoiding the connecting line between runs
                    dico[safe_tag].append(nan)
                    dico[f"{safe_tag}_steps"].append(nan)

                    dico[safe_tag].extend([e.value for e in events])
                    dico[f"{safe_tag}_steps"].extend([e.step for e in events])

                    
        except Exception as e:
            print(f"Failed to process {event_file}: {e}")
    
    for tag, values in dico.items():
        if not tag.endswith("_steps"):
            plt.figure(figsize=(10, 6))
            plt.plot(dico[f"{tag}_steps"], values, label=tag)
            plt.title(f"{tag} over epochs")
            plt.xlabel("Step/Epoch")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
        
            save_path = get_unique_filename(os.path.join(output_folder, f"{tag}"), ".png")

            plt.savefig(save_path)
            plt.close()
            print(f"Saved plot: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TensorBoard logs to PNG images")
    parser.add_argument("--logdir", required=True, help="Path to the folder containing tfevents files")
    parser.add_argument("--output", default="exported_graphs", help="Folder to save PNGs")
    parser.add_argument("--continuous", action="store_true", help="Export all graphs in single plots, use if training was stopped and restarted, to have a continuous graph")
    
    args = parser.parse_args()
    if args.continuous:
        export_continuous_graphs(args.logdir, args.output)
    else:
        export_graphs(args.logdir, args.output)
