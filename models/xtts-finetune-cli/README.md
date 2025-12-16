# XTTS Fine-tuning Pipeline

Fine-tune XTTS models on target speaker audio and generate deepfakes.

Based on [xtts-finetune-webui](https://github.com/daswer123/xtts-finetune-webui).

## Setup

```bash
conda env create -f environment.yml
conda activate xtts-env
```

## Usage

```bash
python main.py path/to/input.wav --out_dir path/to/output
```

## Pipeline Steps

1. Resample input to 22050Hz mono
2. Create dataset with Whisper transcription
3. Train XTTS (6 epochs default)
4. Optimize model
5. Generate 3 deepfake samples

## Output

- `resampled.wav` - Preprocessed input
- `reference.wav` - Reference for cloning
- `audio0.wav`, `audio1.wav`, `audio2.wav` - Generated deepfakes

## Parameters

Edit `main.py` to adjust:
- `num_epochs` (default: 6)
- `batch_size` (default: 2)
- `max_audio_length` (default: 11s)
