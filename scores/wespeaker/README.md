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

### Batch Processing

```bash
./pipe.sh path/to/clean output.csv /full/path/to/files
```

### Direct Python Usage

```bash
python compare.py path/to/clean output.csv reference.wav audio1.wav audio2.wav
```

### Using Local Model

By default, the scripts download the "english" model from the WeSpeaker hub (cached at `~/.wespeaker/english/`). To use a local model:

```bash
# First run downloads to ~/.wespeaker/english/
python -c "import wespeaker; wespeaker.load_model('english')"

# Subsequent runs use local cache (faster, no download check)
./pipe.sh path/to/clean output.csv /full/path/to/files --model ~/.wespeaker/english

# Or directly
python compare.py path/to/clean output.csv reference.wav audio1.wav --model ~/.wespeaker/english
```

Available models: `english`, `chinese`, `campplus`, `eres2net`

## GPU Version

See `gpu/` directory for GPU-accelerated scoring. Same interface:

```bash
cd gpu
./pipe.sh path/to/clean output.csv /full/path/to/files --model /path/to/model
```
