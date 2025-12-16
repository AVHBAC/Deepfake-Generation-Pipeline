# Data Preprocessing

Converts raw audio to clean, single-speaker WAV files using pyannote speaker diarization.

## Setup

Requires a [Hugging Face token](https://huggingface.co/settings/tokens) with read access.

```bash
conda env create -f environment.yml
conda activate preprocess
pip install git+https://github.com/pyannote/pyannote-audio.git
huggingface-cli login
```

Accept model conditions at huggingface.co on first run.

## Usage

```bash
python main.py /path/to/input/audio/ /path/to/output/
```

## Output

For each input file:
- `{name}.wav` - Converted audio
- `{name}-diarization.csv` - Speaker segments
- `{name}-summary.txt` - Diarization stats
- `{name}-clean.wav` - Dominant speaker only
