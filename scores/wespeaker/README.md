# Wespeaker for Scoring

This section allows you to score your preprocessed data, it is intended to be used only with generated data, though can be modified. This does run on CPU only, due to the difficulty of running on GPU, if you find a simple way of getting wespeaker to run on GPU please make a issue and mention it there. See `/scores/wespeaker/gpu` for GPU.

If you are on windows do not use the pipe.sh file, please call compare.py manually or write a DOS or ps1 script of your own.

## Usage

### 1. Install Dependencies

```bash
conda env create -f environment.yml
pip install git+https://github.com/wenet-e2e/wespeaker.git
chmod +x pipe.sh # Linux use only
```

### Run pipe.sh

```bash
./pipe.sh path/to/be/cleaned output.csv /full/path/to/files
```

If you are sharing your scoring data it is recommended to set `path/to/be/clean` to `/home/<your-user>/remaining/path` as to obfuscate yourself from the actually scoring.