# WeSpeaker Scoring

Speaker verification scoring using WeSpeaker embeddings.

## Setup

```bash
conda env create -f environment.yml
conda activate wespeaker
pip install git+https://github.com/wenet-e2e/wespeaker.git
chmod +x pipe.sh  # Linux only
```

## Usage

```bash
./pipe.sh path/to/clean output.csv /full/path/to/files
```

Or call directly:

```bash
python compare.py output.csv reference.wav audio1.wav audio2.wav
```

## GPU Version

See `gpu/` directory for GPU-accelerated scoring.
