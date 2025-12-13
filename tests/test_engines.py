"""Unit tests for engine implementations using lightweight mocks."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable, Optional

import pytest

from persian_transcriber.engines.faster_whisper_engine import FasterWhisperEngine
from persian_transcriber.engines.openai_api_engine import OpenAIAPIEngine
from persian_transcriber.utils.exceptions import AuthenticationError, EngineError


def _patch_whisper_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    init_hook: Optional[Callable[..., None]] = None,
    transcribe_hook: Optional[Callable[[str], None]] = None,
    transcript_text: str = "سلام دنیا",
) -> None:
    """Patch the WhisperModel class with a lightweight fake implementation."""

    class _DummySegment(SimpleNamespace):
        pass

    class _DummyModel:
        def __init__(self, *args, **kwargs):
            if init_hook:
                init_hook(*args, **kwargs)

        def transcribe(self, audio_path: str, **kwargs):  # pylint: disable=unused-argument
            if transcribe_hook:
                transcribe_hook(audio_path)
            segment = _DummySegment(
                text=transcript_text,
                start=0.0,
                end=1.0,
                avg_logprob=-0.1,
                words=None,
            )
            info = SimpleNamespace(
                language=kwargs.get("language", "fa"),
                duration=1.0,
                language_probability=0.99,
            )
            return iter([segment]), info

    dummy_module = ModuleType("faster_whisper")
    dummy_module.WhisperModel = _DummyModel
    monkeypatch.setitem(sys.modules, "faster_whisper", dummy_module)


def _patch_device_helpers(monkeypatch: pytest.MonkeyPatch, *, best_device: str = "cpu") -> None:
    """Stub GPU helper utilities so tests do not touch real hardware."""

    monkeypatch.setattr(
        "persian_transcriber.engines.faster_whisper_engine.get_best_device",
        lambda: best_device,
    )
    monkeypatch.setattr(
        "persian_transcriber.engines.faster_whisper_engine.get_compute_type",
        lambda device: "float16" if device == "cuda" else "int8",
    )
    monkeypatch.setattr(
        "persian_transcriber.engines.faster_whisper_engine.setup_cuda_paths",
        lambda: None,
    )


def test_offline_engine_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_device_helpers(monkeypatch, best_device="cpu")
    _patch_whisper_model(monkeypatch)

    engine = FasterWhisperEngine(model_size="tiny", device="cpu", compute_type="int8")
    engine.load_model()

    assert engine.is_loaded is True
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"


def test_offline_engine_transcribe(monkeypatch: pytest.MonkeyPatch, mock_audio_file: Path) -> None:
    _patch_device_helpers(monkeypatch, best_device="cpu")
    _patch_whisper_model(monkeypatch, transcript_text="سلام دنیا")

    engine = FasterWhisperEngine(model_size="tiny", device="cpu", compute_type="int8")
    engine.load_model()
    result = engine.transcribe(str(mock_audio_file))

    assert result.text == "سلام دنیا"
    assert result.language == "fa"
    assert result.duration == pytest.approx(1.0)


def test_openai_engine_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure no API key from environment
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = OpenAIAPIEngine(api_key=None)
    with pytest.raises(AuthenticationError):
        engine.load_model()


def test_openai_engine_api_call(mock_openai_client, mock_audio_file: Path) -> None:  # pylint: disable=unused-argument
    engine = OpenAIAPIEngine(api_key="test-key")
    result = engine.transcribe(str(mock_audio_file), language="fa")
    assert result.text == "متن تست"
    assert result.language == "fa"


def test_engine_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _init_hook(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            raise RuntimeError("cuda device unavailable")

    _patch_device_helpers(monkeypatch, best_device="cuda")
    _patch_whisper_model(monkeypatch, init_hook=_init_hook)

    engine = FasterWhisperEngine(model_size="small", device=None)
    engine.load_model()

    assert engine.device == "cpu"
    assert engine.compute_type == "int8"


def test_engine_unsupported_format(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"fake")

    def _transcribe_hook(path: str) -> None:
        if path.endswith(".mp3"):
            raise ValueError("unsupported format")

    _patch_device_helpers(monkeypatch, best_device="cpu")
    _patch_whisper_model(monkeypatch, transcribe_hook=_transcribe_hook)

    engine = FasterWhisperEngine(model_size="tiny", device="cpu", compute_type="int8")
    engine.load_model()

    with pytest.raises(EngineError):
        engine.transcribe(str(audio_path))
