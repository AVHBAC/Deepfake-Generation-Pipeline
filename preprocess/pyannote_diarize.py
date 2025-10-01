#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import tqdm
from collections import defaultdict

def diarization(audio_path):
    """Perform speaker diarization using pyannote.audio"""
    
    try:
        # Import pyannote components
        from pyannote.audio import Pipeline
        from pyannote.core import Segment
        print("Successfully imported pyannote.audio")
    except Exception as e:
        print(f"Failed to import pyannote.audio: {e}")
        return None, None
    
    print(f"Processing audio file: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return None, None
    
    try:
        # Try different pipeline models in order of preference
        pipeline_models = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
            "pyannote/speaker-diarization@2022.07"
        ]
        
        pipeline = None
        for model in pipeline_models:
            try:
                print(f"Attempting to load pipeline: {model}")
                pipeline = Pipeline.from_pretrained(model)
                print(f"Successfully loaded: {model}")
                break
            except Exception as e:
                print(f"Failed to load {model}: {e}")
                continue
        
        if pipeline is None:
            print("Failed to load any pyannote pipeline")
            return None, None
        
        # Apply diarization
        print("Running diarization...")
        diarization = pipeline(audio_path)
        
        # Extract results
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                'start_time': round(turn.start, 2),
                'end_time': round(turn.end, 2),
                'duration': round(turn.end - turn.start, 2),
                'speaker': speaker
            })
        
        return results, diarization
        
    except Exception as e:
        print(f"Error during diarization: {e}")
        return None, None

def analyze(results):
    """Analyze pyannote diarization results"""
    if not results:
        return None, None
    
    df = pd.DataFrame(results)
    
    # Calculate statistics
    analysis = {
        'total_speakers': df['speaker'].nunique(),
        'total_segments': len(df),
        'total_duration': df['duration'].sum(),
        'speaker_list': sorted(df['speaker'].unique()),
        'speaker_stats': df.groupby('speaker').agg({
            'duration': ['sum', 'count', 'mean', 'std'],
            'start_time': 'min',
            'end_time': 'max'
        }).round(3)
    }
    
    return analysis, df