# Data Preprocesing

This section allows you to take unclean data and ensure that it is clean prior to input it into the models.

## Usage

### 1. Install Dependencies

You will need a [Hugging Face Token](https://huggingface.co/settings/tokens) with all permisions under `Repositories`, `Inference`, `Webhooks`, and `Collections`.
```bash
conda env create -f environment.yml
huggingface-cli login # You will be asked for your token here, put it in.
```

### 2. Run main.py

The first time your run this you will be asked to go to [Hugging Face](https://huggingface.co/) and accept the conditions of the models. You must do this or they will not work.

```bash
main.py /path/to/input/auidio/ /path/to/output/audio/
```