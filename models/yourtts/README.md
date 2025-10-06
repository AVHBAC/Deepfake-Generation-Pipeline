# Yourtts Pipeline

A command-line pipeline for running inference using a yourtts model.

## Usage

## 1. Install Dependencies

Make sure that you have ffmpeg installed. You are encourage to solve CUDA issues on your own, online assistance does exist.

```bash
conda create -n yourtts_env python=3.10 -y
conda activate yourtts_env
python -m pip install --upgrade pip
pip install TTS soundfile ffmpeg-normalize
```

## 2. Run main.py
```bash
main.py --ref path/to/refernce/audio.wav --outdir path/to/output
```

## 3. Results
This script current produces 3 deepfake audio files. It also reproduces the reference audio and provides resampled audio if it needs to be.
