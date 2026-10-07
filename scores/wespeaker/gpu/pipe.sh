#!/usr/bin/env bash
# Score every <dir>/reference.wav against <dir>/audio{0,1,2}.wav below a root directory.
#
# Usage: ./pipe.sh <cleanpath> <output.csv> <root-dir> [--model /path/to/model]
#   cleanpath   base path stripped from the paths written to the CSV
#   output.csv  CSV file to append to (reference, audio, score; no header)
#   root-dir    directory searched recursively for reference.wav files
#   --model     WeSpeaker model name or local directory (default: "english" from the hub)
#
# Exit status is non-zero if any reference could not be fully scored.
set -u
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ $# -lt 3 ]; then
    echo "Usage: $0 <cleanpath> <output.csv> <root-dir> [--model /path/to/model]" >&2
    exit 1
fi
cleanpath=$1
outfile=$2
filepath=$3
shift 3

model_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --model)
            [ $# -ge 2 ] || { echo "--model needs a value" >&2; exit 1; }
            model_args=(--model "$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ ! -d "$filepath" ]; then
    echo "Not a directory: $filepath" >&2
    exit 1
fi

status=0
while IFS= read -r -d '' file; do
    base=$(dirname "$file")
    echo "Processing: $file"
    python "$script_dir/compute.py" "$cleanpath" "$outfile" "$file" \
        "$base/audio0.wav" "$base/audio1.wav" "$base/audio2.wav" \
        ${model_args[@]+"${model_args[@]}"} || status=1
done < <(find "$filepath" -type f -name reference.wav -print0 | sort -z)
exit $status
