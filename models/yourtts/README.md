# YourTTS Pipeline

Zero-shot voice cloning using YourTTS.

## Setup

```bash
conda env create -f environment.yml
conda activate yourtts
```

Requires ffmpeg installed system-wide.

## Usage

```bash
python main.py --ref path/to/reference.wav --outdir path/to/output
```

## Output

- `reference.wav` - Copy of input reference
- `resampled.wav` - Normalized reference (16kHz)
- `audio0.wav`, `audio1.wav`, `audio2.wav` - Generated deepfakes
