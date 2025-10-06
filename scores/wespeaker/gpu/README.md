# Wespeaker for Scoring on GPU

This section allows you to score your preprocessed data, it is intended to be used only with generated data, though can be modified. Will primary towards running on GPU, falls back to cpu.

## Usage

### 1. Install Dependencies

```bash
conda env create -f ../environment.yml
pip install git+https://github.com/wenet-e2e/wespeaker.git
chmod +x pipe.sh # Linux use only
```

### Run pipe.sh

```bash
./pipe.sh path/to/be/cleaned output.csv /full/path/to/files
```

If you are sharing your scoring data it is recommended to set `path/to/be/clean` to `/home/<your-user>/remaining/path` as to obfuscate yourself from the actually scoring.