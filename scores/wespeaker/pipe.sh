#!/usr/bin/env bash

# Args: path_to_be_cleaned output.csv /full/path/to/files
cleanpath="$1"
outfile="$2"
filepath="$3"

if [ ! -d "$filepath" ]; then
    echo "Usage: $0 path/to/be/cleaned output.csv /full/path/to/files"
    exit 1
fi

# Find all reference.wav files and process them
find "$filepath" -type f -name "reference.wav" | while read -r file; do
    base=$(dirname "$file")
    echo "Processing: $file"
    python3 compare.py "$cleanpath" "$outfile" "$file" \
        "$base/audio0.wav" "$base/audio1.wav" "$base/audio2.wav"
done

