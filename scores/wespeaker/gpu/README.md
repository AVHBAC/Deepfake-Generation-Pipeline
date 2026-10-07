# WeSpeaker Scoring on GPU

Same interface and output format as the CPU scorer in the parent directory; the model is moved to CUDA when a GPU is available and falls back to CPU otherwise.

## Setup

Use the environment from the parent directory:

```bash
conda env create -f ../environment.yml
conda activate wespeaker
```

## Usage

```bash
./pipe.sh <cleanpath> output.csv /full/path/to/files [--model ~/.wespeaker/english]
```

If you share the results file, set `<cleanpath>` to the local prefix you want removed from the paths, for example `/home/<your-user>/data`.
