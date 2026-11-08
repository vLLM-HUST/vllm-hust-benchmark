"""Lossless loader for the pinned MiniCPM-o reference voice, not task audio.

Preserves caller BytesIO position and every PCM16 sample/channel. No model or
neural execution. Other audio is refused rather than silently substituted.
"""

import hashlib
import io
from pathlib import Path
import wave

REFERENCE_SHA256 = "2c4109b2d685e1923ed66433eb08c92047a1f67510629a27edf49af4e5c606dd"  # pragma: allowlist secret - publisher reference WAV
MAX_REFERENCE_BYTES = 2 * 1024**2


def read_reference_pcm16(source):
    if isinstance(source, io.BytesIO):
        if source.getbuffer().nbytes > MAX_REFERENCE_BYTES:
            raise ValueError("reference WAV exceeds byte bound")
        raw = source.getvalue()
    else:
        path = Path(source)
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ValueError("reference WAV exceeds byte bound")
        raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("original pinned reference WAV bytes differ")
    with wave.open(io.BytesIO(raw), "rb") as stream:
        if stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
            raise ValueError("original uncompressed PCM16 reference required")
        channels, rate, frames = (
            stream.getnchannels(),
            stream.getframerate(),
            stream.getnframes(),
        )
        pcm = stream.readframes(frames)
    if len(pcm) != frames * channels * 2:
        raise ValueError("truncated reference PCM frames")
    return pcm, channels, rate, frames


def load_reference_wave(source):
    """Drop-in prepare_prompt loader; Torch tensor construction only, no model."""
    import numpy as np
    import torch

    pcm, channels, rate, _ = read_reference_pcm16(source)
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)
    decoded = samples.astype(np.float32) / 32768
    return torch.from_numpy(decoded.T.copy()), rate
