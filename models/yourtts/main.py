#!/usr/bin/env python3
import os
import sys
import argparse
import string
import torch
import librosa
import subprocess
import tempfile
import shutil
import random
from TTS.tts.utils.synthesis import synthesis
from TTS.utils.audio import AudioProcessor
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.utils.speakers import SpeakerManager

random.seed(42)


def normalize_audio(infile, outfile, target_db=-27, sample_rate=16000):
    """Normalize and resample audio with ffmpeg-normalize"""
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


def compute_reference_embedding(ref_files, ap, se_manager, temp_dir):
    """Compute d-vector embedding from reference files using normalized temp copies."""
    norm_files = []
    for ref in ref_files:
        base = os.path.basename(ref)
        norm_file = os.path.join(temp_dir, f"{base}_norm.wav")
        normalize_audio(ref, norm_file)
        norm_files.append(norm_file)
    return se_manager.compute_d_vector_from_clip(norm_files)


def main():
    parser = argparse.ArgumentParser(description="YourTTS CLI Inference")
    parser.add_argument("--text", type=str, default="Tom sat back down, this time not under the tree but closer to the edge of the lake. He looked across the water, then up at the sky. One star blinked brighter than the rest. It blinked red. Tom took out the map again. Now it was blank. The star was gone. The path was gone. Just white space.", help="Text to synthesize")
    parser.add_argument("--ref", type=str, nargs="+", required=True, help="Reference audio files")
    parser.add_argument("--outdir", type=str, default="out", help="Output directory")
    parser.add_argument("--model_path", type=str, default="./yourtts_models/model_file.pth.tar")
    parser.add_argument("--config_path", type=str, default="./yourtts_models/config.json")
    parser.add_argument("--languages", type=str, default="./yourtts_models/language_ids.json")
    parser.add_argument("--speakers", type=str, default="./yourtts_models/speakers.json")
    parser.add_argument("--se_config", type=str, default="./yourtts_models/config_se.json")
    parser.add_argument("--se_checkpoint", type=str, default="./yourtts_models/model_se.pth.tar")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    use_cuda = torch.cuda.is_available()

    # Load TTS config + model
    C = load_config(args.config_path)
    ap = AudioProcessor(**C.audio)

    C.model_args['d_vector_file'] = args.speakers
    C.model_args['use_speaker_encoder_as_loss'] = False
    model = setup_model(C)
    model.language_manager.set_language_ids_from_file(args.languages)

    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    model_weights = checkpoint['model'].copy()
    for key in list(model_weights.keys()):
        if "speaker_encoder" in key:
            del model_weights[key]
    model.load_state_dict(model_weights)
    model.eval()
    if use_cuda:
        model = model.cuda()

    # Load speaker encoder
    se_manager = SpeakerManager(
        encoder_model_path=args.se_checkpoint,
        encoder_config_path=args.se_config,
        use_cuda=use_cuda
    )

    # Temporary directory for normalized refs
    temp_dir = tempfile.mkdtemp(prefix="yourtts_ref_")
    try:
        # Copy original reference audio to output as 'reference.wav'
        first_ref = args.ref[0]
        reference_copy_path = os.path.join(args.outdir, "reference.wav")
        shutil.copy(first_ref, reference_copy_path)

        # Resample first reference audio to 24000 Hz and save
        resampled_path = os.path.join(args.outdir, "resampled.wav")
        normalize_audio(first_ref, resampled_path, sample_rate=16000)

        # Compute speaker embedding from the resampled file only
        reference_emb = compute_reference_embedding([resampled_path], ap, se_manager, temp_dir)

        # Save normalized version of first reference audio
        first_ref = args.ref[0]
        resample_path = os.path.join(args.outdir, "resample.wav")
        normalize_audio(first_ref, resample_path)

        # Copy original reference audio as 'reference.wav'
        reference_copy_path = os.path.join(args.outdir, "reference.wav")
        shutil.copy(first_ref, reference_copy_path)

        # Start generating synthesized outputs
        text = args.text
        base_name = text.replace(" ", "_")
        base_name = base_name.translate(str.maketrans('', '', string.punctuation.replace('_', '')))

        with open(f"{args.outdir}/info", "a") as f:
            f.write("")

        for i in range(3):
            model.length_scale = 1.0
            model.inference_noise_scale = round(random.uniform(0.2, 0.4), 2)
            model.inference_noise_scale_dp = round(random.uniform(0.2, 0.6), 2)

            wav, _, _, _ = synthesis(
                model,
                text,
                C,
                "cuda" in str(next(model.parameters()).device),
                ap,
                speaker_id=None,
                d_vector=reference_emb,
                style_wav=None,
                language_id=0,
                enable_eos_bos_chars=C.enable_eos_bos_chars,
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()

            out_path = os.path.join(args.outdir, f"audio{i}.wav")
            ap.save_wav(wav, out_path)
            print(f"Saved: {out_path} (noise={model.inference_noise_scale}, dp={model.inference_noise_scale_dp})")
            with open(f"{args.outdir}/info", "a") as f:
                f.write(f"Saved: {out_path} (noise={model.inference_noise_scale}, dp={model.inference_noise_scale_dp})\n")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()