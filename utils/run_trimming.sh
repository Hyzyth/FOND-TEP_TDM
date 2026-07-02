#!/bin/bash

# Ensure PyTorch is available by sourcing the MedSAM2 environment
if [ -d "medsam2_env" ]; then
    source medsam2_env/bin/activate
fi

# Target Directories
DIR_X1= #Your path here
DIR_X2= #Your path here


# Function to loop over a directory and trim files
trim_directory() {
    local target_dir=$1
    echo "========================================================="
    echo " Scanning: $target_dir"
    echo "========================================================="
    
    if [ ! -d "$target_dir" ]; then
        echo "Directory not found! Skipping..."
        echo ""
        return
    fi

    # Enable nullglob so loop doesn't fail if no .pth files exist
    shopt -s nullglob
    for file in "$target_dir"/*.pth "$target_dir"/*.pt; do
        
        # Skip files that already have '_slim' in their name
        if [[ "$file" == *"_slim"* ]]; then
            continue
        fi
        
        # Extract filename, extension, and basename
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        basename="${filename%.*}"
        
        # Construct the new output path
        output_file="$target_dir/${basename}_slim.${extension}"
        
        echo "Processing: $filename"
        python trim_checkpoint.py --input "$file" --output "$output_file"
    done
    shopt -u nullglob
}

# Interactive Menu
echo "Which checkpoints would you like to trim?"
echo "  1) X1"
echo "  2) X2"
echo "  3) Unbound"
echo "  4) unbound2"
echo "  5) All"
echo "  6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        trim_directory "$DIR_X1"
        ;;
    2)
        trim_directory "$DIR_X2"
        ;;
    3)
        trim_directory "$NODIR"
        ;;
    4)

        trim_directory "$NODIR"
        ;;
    5)
        trim_directory "$DIR_X1"
        trim_directory "$DIR_X2"
        ;;
    6)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "All trimming tasks complete!"