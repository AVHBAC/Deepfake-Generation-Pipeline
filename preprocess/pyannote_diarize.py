#!/usr/bin/env python3
"""Speaker diarization helpers built on pyannote.audio.

The pipeline is loaded once per process and moved to the GPU when one is
available. The model repositories are gated on Hugging Face: accept the user
conditions of pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0,
then authenticate with `huggingface-cli login` (or set HF_TOKEN).
"""
import os

import pandas as pd
import torch

PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"
_PIPELINE = None


def get_pipeline():
    """Load the diarization pipeline once and cache it for the process."""
    global _PIPELINE
    if _PIPELINE is None:
        from pyannote.audio import Pipeline

        print(f"Loading pipeline: {PIPELINE_MODEL}")
        pipeline = Pipeline.from_pretrained(PIPELINE_MODEL)
        if pipeline is None:
            raise RuntimeError(
                f"Could not load {PIPELINE_MODEL}. Accept the user conditions for "
                "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0 on "
                "huggingface.co and run `huggingface-cli login`."
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))
        print(f"Pipeline loaded on {device}")
        _PIPELINE = pipeline
    return _PIPELINE


def diarization(audio_path):
    """Run speaker diarization on one file.

    Returns (segments, annotation) where segments is a list of dicts with
    start_time, end_time, duration and speaker, and annotation is the raw
    pyannote.core.Annotation. Raises on any failure.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    pipeline = get_pipeline()
    annotation = pipeline(audio_path)

    segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append({
            "start_time": round(turn.start, 2),
            "end_time": round(turn.end, 2),
            "duration": round(turn.end - turn.start, 2),
            "speaker": speaker,
        })
    return segments, annotation


def analyze(segments):
    """Summarize diarization segments. Returns (analysis dict, DataFrame)."""
    if not segments:
        return None, None

    df = pd.DataFrame(segments)
    analysis = {
        "total_speakers": df["speaker"].nunique(),
        "total_segments": len(df),
        "total_duration": df["duration"].sum(),
        "speaker_list": sorted(df["speaker"].unique()),
        "speaker_stats": df.groupby("speaker").agg({
            "duration": ["sum", "count", "mean", "std"],
            "start_time": "min",
            "end_time": "max",
        }).round(3),
    }
    return analysis, df
