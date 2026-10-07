# XTTS Fine-tuning Pipeline

Fine-tunes Coqui XTTS v2 on one target speaker and generates three deepfake samples.

Based on [xtts-finetune-webui](https://github.com/daswer123/xtts-finetune-webui).

## Setup

```bash
conda env create -f environment.yml
conda activate xtts-env
```

Requirements:

- NVIDIA GPU; fine-tuning used about 8 GB of GPU memory at the default `batch_size=2` (verified on a 16 GB RTX 4080)
- about 15 GB of free disk per run: 2 GB of base models in `base_models/` under the current working directory plus about 13 GB of checkpoints in `--model_dir`
- at least 2 minutes of speech in the input file (the dataset step aborts below 120 s)
- no Hugging Face login; the base models are downloaded from the Coqui gateway on first run and reused afterwards

## Usage

```bash
python main.py path/to/input.wav --out_dir path/to/output [--model_dir finetune_models] [--text "..."]
```

`--model_dir` is deleted and recreated on every run.

## Pipeline Steps

1. Resample input to 22050 Hz mono
2. Transcribe with faster-whisper (`small`, CPU, int8) and build the training CSVs
3. Fine-tune the XTTS GPT (6 epochs, batch 2, lr 5e-6)
4. Strip optimizer and DVAE weights from the last checkpoint
5. Generate 3 samples of `--text` with the fine-tuned model

## Output

In `--out_dir`:

- `resampled.wav` - 22050 Hz mono input
- `reference.wav` - reference used for cloning (same content as `resampled.wav`)
- `audio0.wav`, `audio1.wav`, `audio2.wav` - generated samples, 22050 Hz

## Parameters

Edit `main.py` to adjust `num_epochs` (default 6), `batch_size` (default 2) and `max_audio_length` (default 11 s). Sampling parameters (temperature 0.6, top_k 50, top_p 0.85, repetition penalty 5.0) are in `utils/xtts_header.py`.

## Model license

The XTTS v2 weights are distributed by Coqui under the Coqui Public Model License (CPML), which permits non-commercial use only. See https://coqui.ai/cpml.
