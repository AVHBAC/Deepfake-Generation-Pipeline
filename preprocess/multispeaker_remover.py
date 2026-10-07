#!/usr/bin/env python3
"""Remove least-dominant speakers from an audio file using a diarization CSV.

Reads the diarization CSV with pandas, computes total speaking time per
speaker, keeps only the dominant speakers (by top-K or by minimum share), and
either concatenates the kept segments or mutes the removed ones in place.
Audio I/O uses pydub, which needs ffmpeg on PATH.

-------------------------------------------
CSV FORMAT
-------------------------------------------
Expected columns (any naming variant is OK):
  - speaker: ["speaker","label","speaker_label","SPEAKER","track"]
  - start:   ["start","start_time","start_seconds","tstart","begin"]
  - end:     ["end","end_time","end_seconds","tend","stop"]

Start/end may be numeric seconds (preferred) or timestamps like "HH:MM:SS.sss" or "MM:SS.sss".

-------------------------------------------
USAGE EXAMPLES
-------------------------------------------
# Keep top-1 speaker by total speaking time; concatenate kept segments
python multispeaker_remover.py --audio input.wav --csv diar.csv --out out_top1.wav --keep-top-k 1

# Keep speakers contributing at least 25% each; mute others in-place (same duration)
python multispeaker_remover.py --audio input.wav --csv diar.csv --out out_keep25.wav --min-share 0.25 --mute

# Keep top-2 speakers with 100 ms padding and 50 ms crossfade
python multispeaker_remover.py --audio input.wav --csv diar.csv --out out_top2_pad.wav --keep-top-k 2 --pad 0.1 --crossfade 50
"""

import argparse
import math
import os
from typing import List, Tuple

import pandas as pd
from pydub import AudioSegment   # Audio I/O (needs ffmpeg)

# ---------- Column inference ----------

CANDIDATE_SPEAKER_COLS = ["speaker", "label", "speaker_label", "SPEAKER", "track"]
CANDIDATE_START_COLS   = ["start", "start_time", "start_seconds", "tstart", "begin"]
CANDIDATE_END_COLS     = ["end", "end_time", "end_seconds", "tend", "stop"]


def _pick_col(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a column for {kind}. "
        f"Looked for: {', '.join(candidates)}. "
        f"CSV columns present: {list(df.columns)}"
    )


# ---------- Time parsing ----------

def _parse_time_to_seconds(val) -> float:
    """
    Convert a value to seconds:
      - numeric (int/float string) -> float
      - 'HH:MM:SS(.sss)' -> float seconds
      - 'MM:SS(.sss)'    -> float seconds
    """
    if val is None:
        return math.nan
    s = str(val).strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        pass
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h, m, sec = parts
                return 3600.0 * float(h) + 60.0 * float(m) + float(sec)
            if len(parts) == 2:
                m, sec = parts
                return 60.0 * float(m) + float(sec)
        except ValueError:
            return math.nan
    return math.nan


def _series_to_seconds(series: pd.Series) -> pd.Series:
    return series.map(_parse_time_to_seconds).astype("float64")


# ---------- CSV -> segments ----------

def load_segments(csv_path: str) -> pd.DataFrame:
    """
    Load a diarization CSV and return a DataFrame with columns
      ['start', 'end', 'speaker'] (float64, float64, str)
    """
    df = pd.read_csv(csv_path)
    spk_col = _pick_col(df, CANDIDATE_SPEAKER_COLS, "speaker label")
    st_col  = _pick_col(df, CANDIDATE_START_COLS, "segment start")
    en_col  = _pick_col(df, CANDIDATE_END_COLS, "segment end")

    out = pd.DataFrame({
        "start": _series_to_seconds(df[st_col]),
        "end": _series_to_seconds(df[en_col]),
        "speaker": df[spk_col].astype(str),
    })
    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] > out["start"]]
    out = out.sort_values(["start", "end"]).reset_index(drop=True)
    return out


# ---------- Dominance computation ----------

def total_durations_by_speaker(df: pd.DataFrame) -> pd.Series:
    """Return a Series indexed by speaker with total speaking time (seconds), descending."""
    dur = (df["end"] - df["start"])
    totals = dur.groupby(df["speaker"]).sum()
    return totals.sort_values(ascending=False)


def choose_kept_speakers(
    totals: pd.Series, keep_top_k: int = None, min_share: float = None
) -> List[str]:
    if len(totals) == 0:
        return []
    if keep_top_k is not None:
        keep_top_k = max(1, int(keep_top_k))
        return list(totals.head(keep_top_k).index)
    if min_share is not None:
        total_all = float(totals.sum())
        if total_all <= 0:
            return []
        share = totals / total_all
        return list(share[share >= float(min_share)].index)
    return [totals.index[0]]


# ---------- Interval utilities ----------

def pad_interval(a: float, b: float, pad: float, lo: float, hi: float) -> Tuple[float, float]:
    a2 = max(lo, a - pad)
    b2 = min(hi, b + pad)
    return (a2, b2) if b2 >= a2 else (a, b)


def merge_intervals(intervals: List[Tuple[float, float]], tol: float = 0.02) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + tol:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def invert_intervals(intervals: List[Tuple[float, float]], lo: float, hi: float) -> List[Tuple[float, float]]:
    if not intervals:
        return [(lo, hi)]
    gaps = []
    cur = lo
    for s, e in intervals:
        if s > cur:
            gaps.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        gaps.append((cur, hi))
    return gaps


# ---------- Audio cutting ----------

def cut_audio_concatenate(
    audio_path: str,
    keep_intervals: List[Tuple[float, float]],
    out_path: str,
    crossfade_ms: int = 0
) -> None:
    audio = AudioSegment.from_file(audio_path)
    out = AudioSegment.silent(duration=0)
    for i, (s, e) in enumerate(keep_intervals):
        chunk = audio[int(s * 1000): int(e * 1000)]
        if i == 0 or crossfade_ms <= 0:
            out += chunk
        else:
            out = out.append(chunk, crossfade=crossfade_ms)
    out.export(out_path, format=os.path.splitext(out_path)[1][1:] or "wav")


def cut_audio_mute(
    audio_path: str,
    remove_intervals: List[Tuple[float, float]],
    out_path: str
) -> None:
    audio = AudioSegment.from_file(audio_path)
    mutable = audio
    # Process from end to start to keep indexes stable
    for s, e in sorted(remove_intervals, key=lambda x: x[0], reverse=True):
        start_ms, end_ms = int(s * 1000), int(e * 1000)
        mutable = mutable[:start_ms] + AudioSegment.silent(duration=end_ms - start_ms) + mutable[end_ms:]
    mutable.export(out_path, format=os.path.splitext(out_path)[1][1:] or "wav")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Cut parts from least-dominant speakers.")
    ap.add_argument("--audio", required=True, help="Path to input audio (wav/mp3/flac, etc.)")
    ap.add_argument("--csv",   required=True, help="Path to diarization CSV.")
    ap.add_argument("--out",   required=True, help="Path to output audio.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--concatenate", action="store_true", help="Concatenate kept segments (default).")
    mode.add_argument("--mute", action="store_true", help="Keep original length but mute least-dominant segments.")
    choice = ap.add_mutually_exclusive_group()
    choice.add_argument("--keep-top-k", type=int, default=None,
                        help="Keep only the top-K speakers by total speaking time.")
    choice.add_argument("--min-share", type=float, default=None,
                        help="Keep speakers whose duration share >= MIN_SHARE (e.g., 0.2 = 20%%).")
    ap.add_argument("--pad", type=float, default=0.0, help="Seconds of padding added before/after kept segments.")
    ap.add_argument("--merge-tol", type=float, default=0.02, help="Seconds tolerance for merging intervals.")
    ap.add_argument("--crossfade", type=int, default=0, help="ms of crossfade between concatenated chunks.")
    args = ap.parse_args()

    df = load_segments(args.csv)
    totals = total_durations_by_speaker(df)

    keep_speakers = choose_kept_speakers(totals, keep_top_k=args.keep_top_k, min_share=args.min_share)
    if not keep_speakers:
        raise SystemExit("No speakers selected to keep. Adjust --keep-top-k or --min-share.")

    total_spch = float(totals.sum())
    print("=== Dominance summary (seconds) ===")
    for spk, dur in totals.items():
        pct = (dur / total_spch * 100.0) if total_spch > 0 else 0.0
        print(f"  {str(spk):>16s}  {dur:8.2f}  ({pct:4.1f}%)")
    print("\nKeeping speakers:", ", ".join(map(str, keep_speakers)))

    kept = df[df["speaker"].isin(keep_speakers)]
    audio = AudioSegment.from_file(args.audio)
    audio_len_s = len(audio) / 1000.0

    keep_intervals_raw = [
        pad_interval(s, e, args.pad, lo=0.0, hi=audio_len_s)
        for s, e in zip(kept["start"].tolist(), kept["end"].tolist())
    ]
    keep_intervals = merge_intervals(keep_intervals_raw, tol=args.merge_tol)
    remove_intervals = invert_intervals(keep_intervals, lo=0.0, hi=audio_len_s)

    kept_dur = sum(b - a for a, b in keep_intervals)
    removed_dur = sum(b - a for a, b in remove_intervals)
    print(f"Kept audio duration:     {kept_dur:.2f} s")
    print(f"Removed/muted duration:  {removed_dur:.2f} s")

    if args.mute:
        print("\nMode: MUTE least-dominant intervals (preserve original length).")
        cut_audio_mute(args.audio, remove_intervals, args.out)
    else:
        print("\nMode: CONCATENATE kept intervals (default).")
        cut_audio_concatenate(args.audio, keep_intervals, args.out, crossfade_ms=args.crossfade)

    print(f"\nDone. Wrote: {args.out}")


if __name__ == "__main__":
    main()
