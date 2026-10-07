# WeSpeaker Scoring

Speaker-verification scoring of generated samples against their reference recording with WeSpeaker embeddings.

## Setup

```bash
conda env create -f environment.yml
conda activate wespeaker
```

`environment.yml` installs `requirements-lock.txt`, which pins every package to its 30 September 2025 release and WeSpeaker to upstream commit `f59df4f`, the last one whose GPU path works. Do not run `pip install wespeaker` or install from `master` on top of it. The default model, `english` (ResNet221 trained on VoxCeleb, large-margin fine-tuned), is downloaded from the WeSpeaker hub to `~/.wespeaker/english/` on first use (about 220 MB).

## Usage

### Batch processing

```bash
./pipe.sh <cleanpath> output.csv /full/path/to/files [--model ~/.wespeaker/english]
```

Every `reference.wav` found below `/full/path/to/files` is scored against `audio0.wav`, `audio1.wav` and `audio2.wav` in the same directory. `<cleanpath>` is the prefix stripped from the paths written to the CSV, for example `/home/<user>/data` if you do not want local paths in a shared results file.

### Direct Python usage

```bash
python compare.py <cleanpath> output.csv reference.wav audio0.wav audio1.wav [--model ~/.wespeaker/english]
```

### Using a local model

```bash
# First run downloads to ~/.wespeaker/english/
python -c "import wespeaker; wespeaker.load_model('english')"

# Subsequent runs use the local copy (no download check)
./pipe.sh <cleanpath> output.csv /full/path/to/files --model ~/.wespeaker/english
```

Other hub models: `chinese`, `campplus`, `eres2net`.

## Output

Rows are appended to `output.csv` without a header:

```
reference_path,audio_path,score
```

`score` is WeSpeaker's normalized cosine similarity between the two embeddings, `(cos + 1) / 2`, so it lies in [0, 1] and higher means more similar. The reference is embedded once per call and reused for every comparison. The exit status is non-zero if any file could not be scored.

## GPU version

`gpu/compute.py` and `gpu/pipe.sh` have the same interface and output format and run the model on CUDA when available. Scores agree with the CPU version to about four decimal places.

```bash
cd gpu
./pipe.sh <cleanpath> output.csv /full/path/to/files --model ~/.wespeaker/english
```
