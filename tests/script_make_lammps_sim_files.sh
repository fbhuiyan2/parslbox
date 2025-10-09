#!/bin/bash

# Usage: ./copy_friction.sh /path/to/lammps/examples N

# Check arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 /path/to/lammps/examples N"
    exit 1
fi

EXAMPLES_PATH=$1
N=$2
SRC_DIR="${EXAMPLES_PATH}/friction"

# Check if source exists
if [[ ! -d "$SRC_DIR" ]]; then
    echo "❌ Error: $SRC_DIR not found!"
    exit 1
fi

# Copy N times
for i in $(seq 1 $N); do
    DEST_DIR="friction_$i"
    cp -r "$SRC_DIR" "$DEST_DIR"
    echo "✅ Copied $SRC_DIR → $DEST_DIR"
done

echo "🎉 Done! Created $N copies in current directory."