import argparse
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="Extract model weights from a PyTorch checkpoint to reduce file size.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the original checkpoint")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to save the slimmed checkpoint")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File '{args.input}' does not exist.")
        return

    # Load the checkpoint to CPU to avoid CUDA memory issues
    try:
        ckpt = torch.load(args.input, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Smart extraction: look for standard PyTorch weight keys
    slim_ckpt = None
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in ckpt:
                slim_ckpt = ckpt[key]
                print(f" -> Found '{key}' dict. Extracting...")
                break
    
    # Fallback if standard keys aren't found
    if slim_ckpt is None:
        print(" -> Could not find standard weight keys. Saving the full dictionary.")
        slim_ckpt = ckpt

    # Save the new slim checkpoint
    torch.save(slim_ckpt, args.output)

    # Calculate and print the size difference
    orig_mb = os.path.getsize(args.input) / (1024 * 1024)
    new_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f" -> Saved to {os.path.basename(args.output)}")
    print(f" -> Size reduced from {orig_mb:.1f} MB to {new_mb:.1f} MB.\n")

if __name__ == "__main__":
    main()