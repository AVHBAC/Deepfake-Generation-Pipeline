#!/usr/bin/env python3
"""Zero-shot voice cloning with YourTTS (Coqui TTS).

Normalizes the reference recording, then synthesizes the same text three times.
The three samples differ only through the model's stochastic sampling.
"""
import argparse
import os
import random
import shutil
import subprocess
import sys

import soundfile as sf
import torch
from TTS.api import TTS

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEFAULT_TEXT = (
    "Tom sat back down, this time not under the tree but closer to the edge of the lake. "
    "He looked across the water, then up at the sky. One star blinked brighter than the rest. "
    "It blinked red. Tom took out the map again. Now it was blank. The star was gone. "
    "The path was gone. Just white space."
)
DEFAULT_MODEL = "tts_models/multilingual/multi-dataset/your_tts"
NUM_SAMPLES = 3


def normalize_audio(infile, outfile, target_db=-27, sample_rate=16000):
    """RMS-normalize and resample with ffmpeg-normalize (needs ffmpeg on PATH)."""
    cmd = [
        sys.executable, "-m", "ffmpeg_normalize",
        infile,
        "-nt", "rms",
        f"-t={target_db}",
        "-ar", str(sample_rate),
        "-o", outfile,
        "-f",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg-normalize failed (rc={result.returncode}): {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description="YourTTS zero-shot voice cloning")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--ref", type=str, nargs="+", required=True,
                        help="Reference audio file(s); only the first one is used for cloning")
    parser.add_argument("--outdir", type=str, default="out", help="Output directory")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Coqui TTS model name")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    print(f"Using device: {'cuda' if use_cuda else 'cpu'}")

    tts = TTS(model_name=args.model_name, progress_bar=True, gpu=use_cuda)
    sample_rate = tts.synthesizer.output_sample_rate

    first_ref = args.ref[0]
    reference_copy_path = os.path.join(args.outdir, "reference.wav")
    shutil.copy(first_ref, reference_copy_path)

    resampled_path = os.path.join(args.outdir, "resampled.wav")
    normalize_audio(first_ref, resampled_path, sample_rate=16000)

    for i in range(NUM_SAMPLES):
        wav = tts.tts(text=args.text, speaker_wav=resampled_path, language="en")
        out_path = os.path.join(args.outdir, f"audio{i}.wav")
        sf.write(out_path, wav, sample_rate)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
