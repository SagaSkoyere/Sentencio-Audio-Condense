\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executable-friendly Silero VAD condenser (Option B: official utils).

- Uses the silero-vad package's official utilities:
    * load_silero_vad (TorchScript JIT model)
    * get_speech_timestamps
    * collect_chunks
- Applies a ±1 s pad to final merged speech segments.
- Extracts audio from any media via ffmpeg, resampled to mono/16k PCM.
"""

import os
import sys
import subprocess
import traceback
import contextlib
import wave
import tempfile
from typing import List, Tuple

import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks

# ----------------------------
# Prompts
# ----------------------------
def prompt_user() -> Tuple[str, str]:
    input_dir = input("Enter folder containing input audio/video files:\n").strip('"').strip()
    output_dir = input("Enter folder where condensed audio files will be saved:\n").strip('"').strip()
    return input_dir, output_dir


# ----------------------------
# ffmpeg extraction
# ----------------------------
def extract_audio(input_file: str, temp_audio_file: str):
    """
    Extract mono 16kHz, 16-bit PCM WAV suitable for VAD.
    Requires ffmpeg available on PATH.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        temp_audio_file,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {input_file}:\n{proc.stderr.decode('utf-8', errors='ignore')}")


# ----------------------------
# WAV helpers (16k mono 16-bit PCM only)
# ----------------------------
def read_wave_bytes(path: str):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)
    return data, sr, sw, nch


def write_wave_bytes(path: str, pcm_bytes: bytes, sample_rate=16000, sample_width=2, channels=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


# ----------------------------
# Core: condense using Silero's official post-processing
# ----------------------------
def condense_audio(
    temp_audio_file: str,
    output_file: str,
    pre_post_ms: int = 2000,
    threshold: float = 0.40,
    min_speech_ms: int = 200,
    min_silence_ms: int = 300,
    window_size_samples: int = 1024,
    VERBOSE: bool = True,
):
    """
    Produce a 'condensed' WAV that keeps detected speech (with ±2 s padding).
    Uses Silero's get_speech_timestamps + collect_chunks.
    """
    pcm_bytes, sr, sw, ch = read_wave_bytes(temp_audio_file)
    if not (sr == 16000 and ch == 1 and sw == 2):
        raise ValueError("Unexpected WAV format. Expected mono 16kHz 16-bit PCM.")

    # PCM -> float32 [-1, 1]
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio = torch.from_numpy(samples)

    # Load JIT model from silero_vad package
    model = load_silero_vad(onnx=False)

    # Run VAD to get raw timestamps (returned in samples by default)
    speech_ts = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        window_size_samples=window_size_samples,
        speech_pad_ms=0,            # disable internal pad; we add ±1s ourselves
        return_seconds=False,
    )

    # Apply ±1 s padding and merge overlaps
    total_samples = len(audio)
    pad = int(pre_post_ms * 16)  # 1 ms at 16 kHz = 16 samples
    segs = []
    for seg in speech_ts:
        s = max(0, seg["start"] - pad)
        e = min(total_samples, seg["end"] + pad)
        segs.append((s, e))

    if segs:
        segs.sort()
        merged = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = merged[-1]
            if s <= pe:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
    else:
        merged = []

    # Collect the audio using Silero helper on our merged segments
    # convert to dicts that collect_chunks expects
    merged_dicts = [{"start": s, "end": e} for (s, e) in merged]
    if VERBOSE and merged_dicts:
        def _fmt(seg):
            return f"[{seg['start']/16000:.2f}s, {seg['end']/16000:.2f}s]"
        print('First segments after padding:', ', '.join(_fmt(x) for x in merged_dicts[:3]))
        print('Total segments:', len(merged_dicts))
    final_audio = collect_chunks(merged_dicts, audio, seconds=False, sampling_rate=16000) if merged_dicts else torch.tensor([], dtype=torch.float32)

    out = (final_audio.numpy() * 32768.0).astype(np.int16).tobytes()
    write_wave_bytes(output_file, out, sample_rate=16000, sample_width=2, channels=1)


# ----------------------------
# File filtering + pipeline
# ----------------------------
def valid_media_file(name: str) -> bool:
    EXT = {
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"
    }
    return os.path.splitext(name.lower())[1] in EXT


def process_file(input_path: str, output_dir: str):
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_dir, f"{base}_condensed.wav")
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        temp_wav = os.path.join(td, "audio.wav")
        extract_audio(input_path, temp_wav)
        condense_audio(temp_wav, out_path)


def main():
    try:
        if len(sys.argv) >= 3:
            input_dir, output_dir = sys.argv[1], sys.argv[2]
        else:
            input_dir, output_dir = prompt_user()

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)

        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if valid_media_file(f)]
        if not files:
            print("No supported media files found in the input directory.")
            return

        # Pre-load model once
        _ = load_silero_vad(onnx=False)

        for i, path in enumerate(sorted(files), 1):
            print(f"[{i}/{len(files)}] Processing: {os.path.basename(path)}")
            process_file(path, output_dir)

        print("\nAll files processed successfully!")

    except Exception:
        print("An error occurred:")
        traceback.print_exc()
    finally:
        try:
            input("\nPress Enter to exit...")
        except Exception:
            pass


if __name__ == "__main__":
    main()
