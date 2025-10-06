# Yourtts Pipeline

A command-line pipeline for running inference using a yourtts model.

## Usage

## 1. Install Dependencies
```bash
conda create -n yourtts python=3.9
git clone https://github.com/Edresson/Coqui-TTS
pip install -q -e TTS/
pip install -q torchaudio==0.9.0 pydub ffmpeg-normalize==1.21.0
```

## 2. Run main.py
```bash
main.py --ref path/to/refernce/audio.wav --outdir path/to/output
```

## 3. Results
This script current produces 3 deepfake audio files. It also reproduces the reference audio and provides resampled audio if it needs to be.
