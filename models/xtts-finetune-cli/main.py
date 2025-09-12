import os
import shutil
import torch
import tempfile
import torchaudio
from pathlib import Path
import argparse

# Import functions from utils
from utils.xtts_header import (
    preprocess_dataset,
    train_model,
    optimize_model,
    load_model,
    run_tts,
    get_model_zip,
)

def main():
    parser = argparse.ArgumentParser(
        description="XTTS Training + Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_audio", help="Path to input audio file (WAV)")
    parser.add_argument(
        "--model_dir",
        default="finetune_models",
        help="Directory to store model checkpoints",
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Directory to store final outputs",
    )
    parser.add_argument(
        "--text",
        default="Tom sat back down, this time not under the tree but closer to the edge of the lake. "
                "He looked across the water, then up at the sky. One star blinked brighter than the rest. "
                "It blinked red. Tom took out the map again. Now it was blank. The star was gone. "
                "The path was gone. Just white space.",
        help="Text to synthesize with the trained model",
    )
    parser.add_argument(
        "--help_only",
        action="store_true",
        help="Print help message and exit",
    )
    args = parser.parse_args()

    if args.help_only:
        parser.print_help()
        return

    # Convert to absolute paths
    input_audio = str(Path(args.input_audio).resolve())
    model_dir = str(Path(args.model_dir).resolve())
    out_dir = str(Path(args.out_dir).resolve())
    results_dir = Path(out_dir).resolve()
    results_dir.mkdir(exist_ok=True, parents=True)

    # --- Step 0: Ensure input audio is 22050 Hz mono ---
    print("Checking and resampling input audio...")
    waveform, sr = torchaudio.load(input_audio)

    # Resample if needed
    if sr != 22050:
        print(f"Resampling from {sr} Hz -> 22050 Hz")
        waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)

    # Force mono (average channels if stereo)
    if waveform.shape[0] > 1:
        print("Converting to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Save resampled audio into out_dir
    resampled_audio_path = str(Path(out_dir) / "resampled.wav")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torchaudio.save(resampled_audio_path, waveform, 22050)

    # Use resampled audio as input
    input_audio = resampled_audio_path

    # Reset model dir each run
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Also create ready/ and copy reference.wav
    ready_dir = Path(model_dir) / "ready"
    ready_dir.mkdir(parents=True, exist_ok=True)
    reference_audio_path = ready_dir / "reference.wav"
    shutil.copy(resampled_audio_path, reference_audio_path)
    print(f"Reference audio saved at: {reference_audio_path}")

    print("Step 1 - Creating dataset...")
    train_csv, eval_csv = preprocess_dataset([input_audio], "", "en", "small", model_dir)

    print("Step 2 - Training...")
    train_model(
        "",
        "v2.0.2",
        "en",
        train_csv,
        eval_csv,
        num_epochs=6,
        batch_size=2,
        grad_acumm=1,
        output_path=model_dir,
        max_audio_length=11,
    )

    print("Step 2.5 - Optimizing model...")
    optimized_path = optimize_model(model_dir)
    print("Optimized model at:", optimized_path)

    print("Step 3 - Loading model...")
    model = load_model(
        f"{model_dir}/ready/model.pth",
        f"{model_dir}/ready/config.json",
        f"{model_dir}/ready/vocab.json",
        f"{model_dir}/ready/speakers_xtts.pth",
    )

    print("Step 4 - Running inference...")
    reference_audio = f"{model_dir}/ready/reference.wav"
    for i in range(3):
        output_audio = run_tts(
            "en",
            args.text,
            reference_audio,
        )
        final_path = results_dir / f"audio{i}.wav"
        waveform, sr = torchaudio.load(output_audio)
        if sr != 22050:
            waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
            torchaudio.save(final_path, waveform, 22050)
        #shutil.move(output_audio, final_path)
        print("Generated speech at:", final_path)

    # --- Copy reference and resampled audio into results ---
    if os.path.exists(reference_audio):
        shutil.copy(reference_audio, results_dir / "reference.wav")
        print("Copied reference.wav to results directory.")

if __name__ == "__main__":
    main()
