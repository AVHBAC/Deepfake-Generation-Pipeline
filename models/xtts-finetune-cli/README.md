# XTTS CLI Pipeline

A command-line pipeline for fine-tuning [XTTS](https://github.com/coqui-ai/TTS) models, running inference, and packaging the optimized model. Heavily written from [xtts-finetune-webui](https://github.com/daswer123/xtts-finetune-webui)

This project provides:
- **Step 0:** Resample and adjust input audio
- **Step 1:** Create a dataset from input audio
- **Step 2:** Train the XTTS model
    - Edit epochs, batch size, etc. here.
- **Step 2.5:** Optimize the trained model
- **Step 3:** Load the optimized model
- **Step 4:** Run inference (generate speech from text)

## Usage

### 1. Install dependencies

```bash
conda env create -f environment.yml
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run main.py

```bash
python3 main.py --out_dir path/to/output/dir path/to/input.wav
```

### 3. Results

This script current produces 3 deepfake audio files. It also reproduces the reference audio and provides resampled audio if it needs to be.