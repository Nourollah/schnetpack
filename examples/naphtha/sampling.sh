#!/bin/bash

# Source directory where your 50k files are located
SOURCE_DIR="/Users/masoud/Projects/schnetpack/examples/naphtha/split_xyz_files"

# Destination directory where selected files will be copied
DEST_DIR="/Users/masoud/Projects/schnetpack/examples/naphtha/split_xyz_files_skimmed"

# Counter to track every 50th file
counter=0

# Iterate over all files in the source directory
for file in "$SOURCE_DIR"/*; do
  # Increment the counter
  ((counter++))

  # Check if the counter is 1 (i.e., every 50th file)
  if ((counter % 50 == 1)); then
    # Copy the file to the destination directory
    cp "$file" "$DEST_DIR"
  fi
done

echo "Done! Every 50th file has been copied to $DEST_DIR."
