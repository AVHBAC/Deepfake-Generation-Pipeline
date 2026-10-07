# YourTTS Pipeline

Zero-shot voice cloning with YourTTS (Coqui TTS 0.22.0). No training; the reference recording conditions the model directly.

## Setup

```bash
conda env create -f environment.yml
conda activate yourtts
```

`ffmpeg` must be installed system-wide. The YourTTS model (about 400 MB) is downloaded to `~/.local/share/tts/` on first run. A GPU is used when available.

## Usage

```bash
python main.py --ref path/to/reference.wav --outdir path/to/output [--text "..."]
```

Only the first `--ref` file is used.

## Output

- `reference.wav` - copy of the input reference
- `resampled.wav` - reference RMS-normalized to -27 dB and resampled to 16 kHz
- `audio0.wav`, `audio1.wav`, `audio2.wav` - generated samples at the model's native 16 kHz

The three samples are produced with identical settings and differ only through the model's stochastic sampling.

## Model license

The YourTTS weights shipped with Coqui TTS are licensed CC BY-NC-ND 4.0 (non-commercial, no derivatives). See the `license` field in the package's `.models.json`.
