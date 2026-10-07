# Data Preprocessing

Converts raw recordings to WAV, runs pyannote speaker diarization, and writes a copy that keeps only the dominant speaker.

## Setup

```bash
conda env create -f environment.yml
conda activate preprocess
huggingface-cli login
```

The diarization models are gated. Before the first run, while logged in to huggingface.co with the same account, accept the user conditions on both model pages:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

`ffmpeg` must be installed system-wide (`sudo apt-get install ffmpeg`). A GPU is used automatically when available; CPU works but is slow.

## Usage

```bash
python main.py /path/to/input/audio/ /path/to/output/
```

Every file under the input directory with a common audio or video extension is processed. Other files are listed as skipped. The exit status is 0 only if every file produced all four outputs, otherwise 1 with a failure list on stderr.

## Output

For `<input>/<sub>/<name>.<ext>` the outputs are written to `<output>/<sub>/<name>/`:

- `<name>.wav` - converted audio
- `<name>-diarization.csv` - speaker segments (start_time, end_time, duration, speaker)
- `<name>-summary.txt` - diarization statistics
- `<name>-clean.wav` - dominant speaker only (segments concatenated)

`multispeaker_remover.py` can also be run on its own; see its docstring for the `--min-share`, `--mute`, `--pad` and `--crossfade` options.

## Pinned versions

`environment.yml` installs `requirements-lock.txt`, in which every package is pinned to the newest release available on 30 September 2025 (pyannote-audio 3.4.0, torch 2.8.0, huggingface_hub 0.35.3, lightning 2.5.5). pyannote-audio 4.x requires a second gated repository, and huggingface_hub 1.0 and lightning 2.6 break 3.4.0, so do not upgrade individual packages. Direct dependencies and the reasons for their constraints are in `requirements.in`; see the top-level README for how to regenerate the lock.
