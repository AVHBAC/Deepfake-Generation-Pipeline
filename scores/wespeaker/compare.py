#!/usr/bin/env python3
import sys
import csv
import argparse
import wespeaker


def clean_path(path, base):
    """Remove the given base path prefix from the file path if present."""
    if path.startswith(base):
        return path[len(base):].lstrip("/\\")
    return path


def main():
    parser = argparse.ArgumentParser(description="WeSpeaker similarity scoring (CPU)")
    parser.add_argument("cleanpath", help="Base path to strip from output paths")
    parser.add_argument("outfile", help="Output CSV file")
    parser.add_argument("reference", help="Reference audio file")
    parser.add_argument("audios", nargs="+", help="Audio files to compare")
    parser.add_argument("--model", default="english",
                        help="Model name or local path (default: english)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = wespeaker.load_model(args.model)

    rows = []

    print(f"Reference file: {args.reference}")
    for path in args.audios:
        try:
            similarity = model.compute_similarity(args.reference, path)
            print(f"Similarity with {path}: {similarity}")
            rows.append([
                clean_path(args.reference, args.cleanpath),
                clean_path(path, args.cleanpath),
                similarity
            ])
        except Exception as e:
            print(f"Error processing {path}: {e}")

    with open(args.outfile, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nResults written to {args.outfile}")


if __name__ == "__main__":
    main()
