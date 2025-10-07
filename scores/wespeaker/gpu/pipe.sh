#!/usr/bin/env bash

# Usage: ./script.sh output.csv /full/path/to/files
outfile="$1"
filepath="$2"
logfile="log.txt"

if [ ! -d "$filepath" ]; then
    echo "Usage: $0 output.csv /full/path/to/files"
    exit 1
fi

# Ensure log file exists
touch "$logfile"
touch "$outfile"

# Find all reference.wav files and process them
find "$filepath" -type f -name "reference.wav" | while read -r file; do
    base=$(dirname "$file")
    echo "Processing: $file"

    retries=0
    success=false

    # Get line count before running
    prev_lines=$(wc -l < "$outfile" 2>/dev/null || echo 0)

    while [ $retries -lt 10 ]; do
        python3 compute.py "$outfile" "$base/resampled.wav" "$base/audio0.wav" "$base/audio1.wav" "$base/audio2.wav"
        exit_code=$?

        # Get new line count
        new_lines=$(wc -l < "$outfile" 2>/dev/null || echo 0)
        diff=$((new_lines - prev_lines))

        if [ $exit_code -eq 0 ] && [ $diff -eq 3 ]; then
            success=true
            break
        else
            retries=$((retries + 1))
            echo "[WARN] $file failed $retries times." >> "$logfile"
            sleep 1  # Optional small delay before retry
        fi
    done

    if [ "$success" = false ]; then
        echo "[ERROR] $file failed." >> "$logfile"
    fi
done

