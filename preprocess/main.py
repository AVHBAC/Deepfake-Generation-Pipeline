#!/usr/bin/env python3
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

from pyannote_diarize import diarization, analyze

def process_directory(input_dir, output_dir):
    # Collect all files first
    all_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            all_files.append(Path(root) / file)

    # Progress bar
    for file_path in tqdm(all_files, desc="Processing files", unit="file"):
        try:
            rel_path = file_path.relative_to(input_dir)
            rel_path_no_ext = rel_path.with_suffix('')
            out_path = Path(output_dir) / rel_path_no_ext

            # Make sure output subdir exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.mkdir(exist_ok=True)

            # 1. Convert to wav
            try:
                tqdm.write(f"[INFO] Converting {file_path.name} to wav...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(file_path), str(out_path / f"{rel_path_no_ext}.wav")],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                tqdm.write(f"[WARN] ffmpeg failed for {file_path}: {e}")

            # 2. Diarization
            try:
                tqdm.write(f"[INFO] Running diarization on {file_path.name}...")
                results, diarization_obj = diarization(f"{out_path}/{rel_path_no_ext}.wav")
                analysis, results_df = analyze(results)

                results_df.to_csv(f"{out_path}/{rel_path_no_ext}-diarization.csv", index=False)

                with open(f"{out_path}/{rel_path_no_ext}-summary.txt", 'w') as f:
                    f.write("PYANNOTE DIARIZATION SUMMARY\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Total Speakers: {analysis['total_speakers']}\n")
                    f.write(f"Speaker Labels: {analysis['speaker_list']}\n")
                    f.write(f"Total Segments: {analysis['total_segments']}\n")
                    f.write(f"Total Duration: {analysis['total_duration']:.2f} seconds\n\n")
                    f.write("Speaker Statistics:\n")
                    f.write(str(analysis['speaker_stats']))
            except Exception as e:
                tqdm.write(f"[WARN] Diarization failed for {file_path}: {e}")

            # 3. Speaker removal
            try:
                    tqdm.write(f"[INFO] Removing non-dominant speakers for {file_path.name}...")
                    subprocess.run(
                        [
                            "python", "multispeaker_remover.py",
                            "--audio", f"{out_path}/{rel_path_no_ext}.wav",
                            "--csv", f"{out_path}/{rel_path_no_ext}-diarization.csv",
                            "--keep-top-k", "1",
                            "--out", f"{out_path}/{rel_path_no_ext}-clean.wav"
                        ],
                        #stdout=subprocess.DEVNULL,   # discard stdout
                        #stderr=subprocess.DEVNULL    # discard stderr
                        capture_output=True,
                        text=True
                    )
                    tqdm.write(f"[INFO] Removed non-dominant speakers for {out_path}/{rel_path_no_ext}-clean.wav...")
            except Exception as e:
                tqdm.write(f"[WARN] Multispeaker remover failed for {file_path}: {e}")

        except Exception as e:
            tqdm.write(f"[ERROR] Unexpected failure with file {file_path.name}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Walk input dir and process files.")
    parser.add_argument("input_dir", type=str, help="Path to input directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()

    process_directory(Path(args.input_dir), Path(args.output_dir))

if __name__ == "__main__":
    main()
