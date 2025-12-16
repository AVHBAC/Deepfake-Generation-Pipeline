#!/usr/bin/env bash

# Usage: ./pipe.sh path/to/be/cleaned output.csv /full/path/to/files [--model /path/to/model]
# Args:
#   $1 - cleanpath: Base path to strip from output paths
#   $2 - outfile: Output CSV file
#   $3 - filepath: Directory containing audio files
#   --model: Optional local model path (default: "english" from hub)

cleanpath="$1"
outfile="$2"
filepath="$3"
shift 3

# Parse optional --model argument
model_arg=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            model_arg="--model $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ ! -d "$filepath" ]; then
    echo "Usage: $0 path/to/be/cleaned output.csv /full/path/to/files [--model /path/to/model]"
    exit 1
fi

# Find all reference.wav files and process them
find "$filepath" -type f -name "reference.wav" | while read -r file; do
    base=$(dirname "$file")
    echo "Processing: $file"
    python3 compare.py "$cleanpath" "$outfile" "$file" \
        "$base/audio0.wav" "$base/audio1.wav" "$base/audio2.wav" $model_arg
done
