# Deepfake Generation Pipeline

An end-to-end pipeline for **audio deepfake generation and scoring**, built to produce extended-length synthetic speech of a target speaker from their real recordings:

- **Preprocessing** with pyannote speaker diarization, keeping only the dominant speaker
- **XTTS v2 fine-tuning** on the target speaker followed by synthesis
- **YourTTS zero-shot cloning** from the same reference
- **WeSpeaker speaker-verification scoring** of every generated sample against its reference

Each stage is a self-contained module with its own pinned conda environment, so stages can be run on different machines or re-run independently.

---

## Features

- Processes a whole directory tree of recordings in one command and keeps the folder structure
- Fine-tunes XTTS v2 (6 epochs, lr 5e-6) and generates three samples per speaker
- Produces a matched YourTTS baseline without training
- Scores every sample with a VoxCeleb ResNet221 speaker-verification model on CPU or GPU
- Fails loudly: every script exits non-zero when an output is missing
- Every Python package locked to an exact version (PyPI state of 30 September 2025) and verified from a clean install on Ubuntu 22.04

---

## Requirements

- Ubuntu 22.04 or newer (verified on 22.04, kernel 6.8)
- NVIDIA GPU with a driver that supports CUDA 12 (verified on an RTX 4080 16 GB, driver 580). XTTS fine-tuning used about 8 GB of GPU memory at the default batch size; the other stages need less or run on CPU.
- Conda (Miniconda or Anaconda), `git` and `ffmpeg` installed system-wide
- A Hugging Face account for the gated pyannote models (preprocessing only)
- Disk: about 30 GB for the four environments plus about 15 GB per XTTS run

---

## Installation

Each module has its own environment. Create the ones you need:

```bash
git clone https://github.com/AVHBAC/Deepfake-Generation-Pipeline.git
cd Deepfake-Generation-Pipeline

conda env create -f preprocess/environment.yml                # env: preprocess
conda env create -f models/xtts-finetune-cli/environment.yml  # env: xtts-env
conda env create -f models/yourtts/environment.yml            # env: yourtts
conda env create -f scores/wespeaker/environment.yml          # env: wespeaker
```

For preprocessing, accept the user conditions of [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) on huggingface.co, then run `huggingface-cli login` inside the `preprocess` environment. The TTS base models and the WeSpeaker model are downloaded automatically on first use.

Each `environment.yml` installs the module's `requirements-lock.txt`, in which every package, direct or transitive, is pinned to the newest release that existed on 30 September 2025, the period in which the pipeline was developed. Newer releases of setuptools, torchaudio, huggingface_hub, lightning and pyannote-audio each break one of the stages, so do not `pip install` or upgrade anything on top of these environments. The direct dependencies and the reasons for their constraints are in each module's `requirements.in`.

To change a dependency, edit `requirements.in` and regenerate the lock with [uv](https://docs.astral.sh/uv/):

```bash
uv pip compile requirements.in -o requirements-lock.txt --exclude-newer 2025-09-30T23:59:59Z --python-version 3.10
```

Move the `--exclude-newer` date forward only together with a full re-test of the module.

---

## Usage

A full run for one collection of recordings:

```bash
# 1. Clean: convert, diarize, keep the dominant speaker
conda activate preprocess
python preprocess/main.py /data/raw/ /data/clean/
#    -> /data/clean/<sub>/<name>/<name>-clean.wav

# 2a. XTTS: fine-tune on the cleaned speaker and synthesize 3 samples
conda activate xtts-env
cd models/xtts-finetune-cli
python main.py /data/clean/<sub>/<name>/<name>-clean.wav --out_dir /data/xtts/<name>
cd ../..

# 2b. YourTTS: zero-shot clone from the same reference
conda activate yourtts
python models/yourtts/main.py --ref /data/clean/<sub>/<name>/<name>-clean.wav --outdir /data/yourtts/<name>

# 3. Score every generated sample against its reference
conda activate wespeaker
scores/wespeaker/gpu/pipe.sh /data /data/xtts-scores.csv /data/xtts
scores/wespeaker/gpu/pipe.sh /data /data/yourtts-scores.csv /data/yourtts
```

Every generation output directory contains `reference.wav`, `resampled.wav` and `audio0.wav`, `audio1.wav`, `audio2.wav`, which is the layout the scoring step expects. See each module's README for options and details.

---

## Project Structure

```
├── preprocess/
│   ├── main.py                  → directory walker: ffmpeg → diarization → speaker removal
│   ├── pyannote_diarize.py      → pyannote pipeline (loaded once, GPU when available)
│   ├── multispeaker_remover.py  → keeps dominant speaker(s) from a diarization CSV
│   └── environment.yml
├── models/
│   ├── xtts-finetune-cli/
│   │   ├── main.py              → resample → dataset → fine-tune → optimize → synthesize
│   │   ├── utils/               → vendored fine-tuning code (based on xtts-finetune-webui)
│   │   └── environment.yml
│   └── yourtts/
│       ├── main.py              → normalize reference → zero-shot synthesis
│       └── environment.yml
└── scores/wespeaker/
    ├── compare.py, pipe.sh      → CPU scoring
    ├── gpu/compute.py, pipe.sh  → GPU scoring, same interface
    └── environment.yml
```

---

## Models and Licenses

This repository's code is MIT licensed. The models it downloads at run time have their own terms:

| Stage | Model | Source | License / access |
|-------|-------|--------|------------------|
| Preprocess | pyannote/speaker-diarization-3.1, pyannote/segmentation-3.0 | Hugging Face | MIT, gated: user conditions must be accepted |
| XTTS | Coqui XTTS v2 (v2.0.2) | Coqui gateway | Coqui Public Model License (CPML), non-commercial |
| YourTTS | tts_models/multilingual/multi-dataset/your_tts | Coqui TTS hub | CC BY-NC-ND 4.0 |
| Scoring | WeSpeaker `english` (voxceleb_resnet221_LM) | WeSpeaker hub (ModelScope) | see the WeSpeaker model page |

Check these terms before using generated audio outside research.

---

## Notes and Limitations

- XTTS fine-tuning requires at least 2 minutes of speech per speaker and writes about 2 GB of base models into the current working directory (`base_models/`) plus about 13 GB of checkpoints into `--model_dir`, which is recreated on every run.
- Whisper transcription in the XTTS stage runs on CPU (int8) and dominates the runtime for long recordings.
- The YourTTS samples differ from each other only through the model's stochastic sampling.
- Scoring CSVs have no header row and are appended to, so delete an old file before re-scoring into the same path.
- Only English (`language="en"`) is wired through the command-line scripts.
- The lock files were resolved for Linux x86_64 and Python 3.10. Other platforms would need their own resolution.
- GPU inference is not bit-exact across runs: repeated XTTS runs with the same seed produced samples whose speaker-similarity scores agree to five decimal places, not identical files.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Acknowledgements

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) speaker diarization
- [Coqui TTS](https://github.com/coqui-ai/TTS) XTTS v2 and YourTTS, and [xtts-finetune-webui](https://github.com/daswer123/xtts-finetune-webui) for the fine-tuning code
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription
- [WeSpeaker](https://github.com/wenet-e2e/wespeaker) speaker verification

CITeR project: Development of Extended-Length Audio Dataset for Advanced Deepfake Synthesis and Detection, Project #24F-02C, Masudul H. Imtiaz (Clarkson University), mimtiaz@clarkson.edu.
