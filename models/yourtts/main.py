#!/usr/bin/env python3
import os
import sys
import argparse
import string
import torch
import random
import tempfile
import shutil
import soundfile as sf
from TTS.api import TTS
from TTS.utils.audio import AudioProcessor

random.seed(42)

def normalize_audio(infile, outfile, target_db=-27, sample_rate=16000):
    """Normalize and resample audio with ffmpeg-normalize"""
    import subprocess
    cmd = [
        "ffmpeg-normalize",
        infile,
        "-nt", "rms",
        f"-t={target_db}",
        "-ar", str(sample_rate),
        "-o", outfile,
        "-f"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    parser = argparse.ArgumentParser(description="TTS CLI Inference")
    parser.add_argument("--text", type=str, default="Tom sat back down, this time not under the tree but closer to the edge of the lake. "
                "He looked across the water, then up at the sky. One star blinked brighter than the rest. "
                "It blinked red. Tom took out the map again. Now it was blank. The star was gone. "
                "The path was gone. Just white space.", help="Text to synthesize")
    parser.add_argument("--ref", type=str, nargs="+", required=True, help="Reference audio files for voice cloning")
    parser.add_argument("--outdir", type=str, default="out", help="Output directory")
    parser.add_argument("--model_name", type=str, default="tts_models/multilingual/multi-dataset/your_tts", help="HuggingFace TTS model")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    use_cuda = torch.cuda.is_available()

    # Initialize TTS (from pip-installed package)
    tts = TTS(model_name=args.model_name, progress_bar=True, gpu=use_cuda)

    # Temporary directory for normalized reference audio
    temp_dir = tempfile.mkdtemp(prefix="yourtts_ref_")
    try:
        # Normalize first reference audio and copy to out/
        first_ref = args.ref[0]
        reference_copy_path = os.path.join(args.outdir, "reference.wav")
        shutil.copy(first_ref, reference_copy_path)

        resampled_path = os.path.join(args.outdir, "resampled.wav")
        normalize_audio(first_ref, resampled_path, sample_rate=16000)

        # Load reference embedding for cloning (if supported by model)
        try:
            reference_emb = tts.tts_speaker_embedding(resampled_path)
        except Exception:
            reference_emb = None
            print("Warning: model may not support speaker embeddings.")

        # Generate audio files
        base_name = args.text.replace(" ", "_").translate(str.maketrans('', '', string.punctuation.replace('_', '')))

        for i in range(3):
            noise_scale = round(random.uniform(0.2, 0.4), 2)
            noise_scale_dp = round(random.uniform(0.2, 0.6), 2)

            wav = tts.tts(
                text=args.text,
                speaker_wav=resampled_path if os.path.exists(resampled_path) else None,
                language="en"
            )

            out_path = os.path.join(args.outdir, f"audio{i}.wav")
            sf.write(out_path, wav, 16000)
            print(f"Saved: {out_path} (noise={noise_scale}, dp={noise_scale_dp})")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
