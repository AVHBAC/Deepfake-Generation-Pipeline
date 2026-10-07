#!/usr/bin/env python3
"""WeSpeaker speaker-similarity scoring (GPU when available).

Same interface and output format as ../compare.py; the model runs on CUDA when
a GPU is present. The score is WeSpeaker's normalized cosine similarity,
(cos + 1) / 2, in [0, 1]. Rows are appended to the CSV as: reference, audio,
score (no header).
"""
import argparse
import csv
import sys

import torch
import wespeaker


def clean_path(path, base):
    """Remove the given base path prefix from the file path if present."""
    if path.startswith(base):
        return path[len(base):].lstrip("/\\")
    return path


def score_files(model, reference, audios, cleanpath):
    ref_emb = model.extract_embedding(reference)
    if ref_emb is None:
        raise RuntimeError(f"Could not extract an embedding from reference {reference}")

    rows = []
    for path in audios:
        try:
            emb = model.extract_embedding(path)
            if emb is None:
                raise RuntimeError("no speech found after VAD")
            similarity = model.cosine_similarity(ref_emb, emb)
            print(f"Similarity with {path}: {similarity}")
            rows.append([clean_path(reference, cleanpath), clean_path(path, cleanpath), similarity])
        except Exception as e:  # noqa: BLE001 - report and continue with the next file
            print(f"Error processing {path}: {e}", file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser(description="WeSpeaker similarity scoring (GPU)")
    parser.add_argument("cleanpath", help="Base path to strip from output paths")
    parser.add_argument("outfile", help="Output CSV file (rows are appended)")
    parser.add_argument("reference", help="Reference audio file")
    parser.add_argument("audios", nargs="+", help="Audio files to compare")
    parser.add_argument("--model", default="english",
                        help="Model name or local path (default: english)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = wespeaker.load_model(args.model)
    model.set_device(device)

    print(f"Reference file: {args.reference}")
    rows = score_files(model, args.reference, args.audios, args.cleanpath)

    with open(args.outfile, "a", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n{len(rows)} of {len(args.audios)} scores written to {args.outfile}")
    sys.exit(0 if len(rows) == len(args.audios) else 1)


if __name__ == "__main__":
    main()
