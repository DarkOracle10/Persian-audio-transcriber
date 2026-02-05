"""Shared pytest fixtures for the Persian Transcriber test-suite."""

from __future__ import annotations

import sys
import types
import wave
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pytest


def _write_silence_wav(
    path: Path, duration_seconds: float = 1.0, sample_rate: int = 16_000
) -> None:
    """Create a silent mono WAV file for testing purposes."""
    frames = int(duration_seconds * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


@pytest.fixture
def mock_audio_file(tmp_path: Path) -> Path:
    """Return a path to a 1-second silent WAV file."""
    audio_path = tmp_path / "sample.wav"
    _write_silence_wav(audio_path)
    return audio_path


@pytest.fixture
def temp_folder(tmp_path: Path, mock_audio_file: Path) -> Path:
    """Create a temporary folder populated with sample audio files."""
    import shutil

    target_dir = tmp_path / "batch"
    target_dir.mkdir()

    for idx in range(2):
        destination = target_dir / f"sample_{idx}.wav"
        shutil.copyfile(mock_audio_file, destination)

    (target_dir / "notes.txt").write_text("not audio", encoding="utf-8")
    return target_dir


@pytest.fixture
def sample_persian_text() -> Dict[str, str]:
    """Provide reusable Persian text samples for normalizer tests."""
    return {
        "arabic_chars": "سلام كیف حالك؟",
        "whitespace": "  سلام    دنیا   ",
        "punctuation": "سلام، دنیا!؟",
        "numbers": "سال ١٤٠٢",
        "mixed": "سلام Tehran 2025",
        "special": "سلام   @#$%   دنیا",
        "unicode": "\u200cسلام\tدنیا\n",  # includes zero-width and control chars
    }


@pytest.fixture
def mock_openai_client(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Mock the OpenAI client so tests never hit the real API."""

    class _DummySegments(list):
        """Simple container that mimics the iterable API response segments."""

    class _DummyResponse:
        def __init__(self, text: str = "متن تست", language: str = "fa") -> None:
            self.text = text
            self.language = language
            self.duration = 1.0
            self.segments = [
                {"text": text, "start": 0.0, "end": 1.0},
            ]

    class _DummyAudioEndpoint:
        def __init__(self) -> None:
            self.transcriptions = self
            self.translations = self

        def create(self, *args, **kwargs):  # pylint: disable=unused-argument
            language = kwargs.get("language", "fa")
            return _DummyResponse(text="متن تست", language=language)

    class _DummyClient:
        def __init__(self, **kwargs):  # pylint: disable=unused-argument
            self.audio = _DummyAudioEndpoint()

    dummy_module = types.ModuleType("openai")
    dummy_module.OpenAI = _DummyClient
    monkeypatch.setitem(sys.modules, "openai", dummy_module)
    return dummy_module
