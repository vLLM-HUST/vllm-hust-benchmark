"""Invented PCM fixtures only; not reference-voice or neural execution evidence."""

import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import types
import wave

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "duplex_reference_audio",
    Path(__file__).resolve().parents[1] / "scripts/full_duplex_reference_audio.py",
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def fixture(monkeypatch, channels=1):
    samples = [-32768, -1, 0, 1, 32767, 100]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack("<6h", *samples))
    raw = buffer.getvalue()
    monkeypatch.setattr(ENTRY, "REFERENCE_SHA256", hashlib.sha256(raw).hexdigest())
    return raw, samples


def test_fixed_bytesio_position_and_path_are_equivalent(tmp_path, monkeypatch):
    raw, _ = fixture(monkeypatch)
    path = tmp_path / "reference.wav"
    path.write_bytes(raw)
    buffer = io.BytesIO(raw)
    buffer.seek(7)
    assert ENTRY.read_reference_pcm16(path) == ENTRY.read_reference_pcm16(buffer)
    assert buffer.tell() == 7


@pytest.mark.parametrize("channels", [1, 2])
def test_every_pcm16_sample_and_channel_is_preserved(monkeypatch, channels):
    raw, samples = fixture(monkeypatch, channels)
    # Tensor construction fixture, not a CPU model or Torch runtime substitute.
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(from_numpy=lambda array: array)
    )
    output, rate = ENTRY.load_reference_wave(io.BytesIO(raw))
    expected = np.array(samples, dtype=np.float32).reshape(-1, channels).T / 32768
    np.testing.assert_array_equal(output, expected)
    assert output.dtype == np.float32 and rate == 16000


def test_altered_reference_is_refused(monkeypatch):
    raw, _ = fixture(monkeypatch)
    with pytest.raises(ValueError, match="bytes differ"):
        ENTRY.read_reference_pcm16(io.BytesIO(raw[:-1] + b"x"))


def test_oversized_reference_is_refused_before_digest(monkeypatch):
    monkeypatch.setattr(ENTRY, "MAX_REFERENCE_BYTES", 2)
    with pytest.raises(ValueError, match="byte bound"):
        ENTRY.read_reference_pcm16(io.BytesIO(b"oversized"))
