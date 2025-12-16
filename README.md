# Deepfake Generation Pipeline

End-to-end pipeline for audio deepfake generation: preprocessing, TTS fine-tuning/inference, and speaker verification scoring.

## Structure

```
├── preprocess/          # Speaker diarization and audio cleanup
├── models/
│   ├── xtts-finetune-cli/  # XTTS fine-tuning
│   └── yourtts/            # YourTTS inference
└── scores/wespeaker/       # Speaker verification scoring
```

## Requirements

- Ubuntu 22.04+ (tested on 5.15.0-157 kernel)
- CUDA 11.8+ (XTTS module requires CUDA 11.8 for PyTorch compatibility)
- Conda/Miniconda
- ffmpeg

## Quick Start

Each module has its own environment. See the README in each directory for setup instructions.

```bash
# Example: Preprocessing
cd preprocess
conda env create -f environment.yml
conda activate preprocess
python main.py /path/to/input /path/to/output
```

## Modules

| Module | Purpose |
|--------|---------|
| `preprocess/` | Converts audio to WAV, runs pyannote diarization, removes non-dominant speakers |
| `models/xtts-finetune-cli/` | Fine-tunes XTTS on target speaker, generates deepfakes |
| `models/yourtts/` | Zero-shot voice cloning with YourTTS |
| `scores/wespeaker/` | Speaker verification scoring with WeSpeaker |

## License

MIT
