#!/usr/bin/env python3
"""Walk an input directory, convert every audio file to WAV, run pyannote
speaker diarization, and keep only the dominant speaker.

For an input file <input_dir>/<sub>/<name>.<ext> the outputs are written to
<output_dir>/<sub>/<name>/:
    <name>.wav                 converted audio
    <name>-diarization.csv     speaker segments
    <name>-summary.txt         diarization statistics
    <name>-clean.wav           dominant speaker only

Exit status is 0 only if every input file produced all four outputs.
Otherwise it is 1 and a per-file failure summary is printed on stderr.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from pyannote_diarize import analyze, diarization

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
}


def process_file(file_path: Path, input_dir: Path, output_dir: Path) -> None:
    """Process one input file. Raises on any failure."""
    rel_no_ext = file_path.relative_to(input_dir).with_suffix("")
    out_dir = output_dir / rel_no_ext
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rel_no_ext.name

    wav_path = out_dir / f"{stem}.wav"
    csv_path = out_dir / f"{stem}-diarization.csv"
    summary_path = out_dir / f"{stem}-summary.txt"
    clean_path = out_dir / f"{stem}-clean.wav"

    # 1. Convert to WAV
    tqdm.write(f"[INFO] Converting {file_path.name} to wav...")
    ffmpeg = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(file_path), str(wav_path)],
        capture_output=True, text=True,
    )
    if ffmpeg.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={ffmpeg.returncode}): {ffmpeg.stderr.strip()}")

    # 2. Diarization
    tqdm.write(f"[INFO] Running diarization on {file_path.name}...")
    segments, _ = diarization(str(wav_path))
    analysis, df = analyze(segments)
    if df is None:
        raise RuntimeError("diarization returned no speech segments")
    df.to_csv(csv_path, index=False)
    with open(summary_path, "w") as f:
        f.write("PYANNOTE DIARIZATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Speakers: {analysis['total_speakers']}\n")
        f.write(f"Speaker Labels: {analysis['speaker_list']}\n")
        f.write(f"Total Segments: {analysis['total_segments']}\n")
        f.write(f"Total Duration: {analysis['total_duration']:.2f} seconds\n\n")
        f.write("Speaker Statistics:\n")
        f.write(str(analysis["speaker_stats"]))

    # 3. Keep the dominant speaker only
    tqdm.write(f"[INFO] Removing non-dominant speakers for {file_path.name}...")
    script = Path(__file__).resolve().parent / "multispeaker_remover.py"
    remover = subprocess.run(
        [
            sys.executable, str(script),
            "--audio", str(wav_path),
            "--csv", str(csv_path),
            "--keep-top-k", "1",
            "--out", str(clean_path),
        ],
        capture_output=True, text=True,
    )
    if remover.returncode != 0:
        raise RuntimeError(
            f"multispeaker_remover.py failed (rc={remover.returncode}): "
            f"{remover.stderr.strip()[-2000:]}"
        )
    if not clean_path.is_file():
        raise RuntimeError("multispeaker_remover.py exited 0 but wrote no output")
    tqdm.write(f"[INFO] Wrote {clean_path}")


def process_directory(input_dir: Path, output_dir: Path) -> int:
    audio_files, skipped = [], []
    for root, _, files in os.walk(input_dir):
        for name in files:
            path = Path(root) / name
            (audio_files if path.suffix.lower() in AUDIO_EXTENSIONS else skipped).append(path)
    audio_files.sort()

    for path in skipped:
        tqdm.write(f"[SKIP] {path} (unsupported extension)")
    if not audio_files:
        print(f"[ERROR] No audio files found under {input_dir}", file=sys.stderr)
        return 1

    failures = []
    for file_path in tqdm(audio_files, desc="Processing files", unit="file"):
        try:
            process_file(file_path, input_dir, output_dir)
        except Exception as e:  # noqa: BLE001 - record and continue with the next file
            tqdm.write(f"[ERROR] {file_path}: {e}")
            failures.append((file_path, str(e)))

    ok = len(audio_files) - len(failures)
    print(f"\nProcessed {len(audio_files)} file(s): {ok} succeeded, {len(failures)} failed.")
    for path, msg in failures:
        first_line = msg.splitlines()[0] if msg else ""
        print(f"  FAILED {path}: {first_line}", file=sys.stderr)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert, diarize and clean every audio file under a directory.")
    parser.add_argument("input_dir", type=str, help="Path to input directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    sys.exit(process_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve()))


if __name__ == "__main__":
    main()
