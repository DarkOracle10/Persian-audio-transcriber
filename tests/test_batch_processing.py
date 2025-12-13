"""Tests covering directory batch transcription workflows."""

from pathlib import Path
from typing import List, Tuple

import pytest

from persian_transcriber.transcriber import PersianAudioTranscriber


@pytest.fixture(autouse=True)
def _stub_cuda_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent CUDA setup side effects during tests."""
    monkeypatch.setattr("persian_transcriber.transcriber.setup_cuda_paths", lambda: None)


def _make_media_files(base_dir: Path, names: List[str]) -> List[Path]:
    created = []
    for name in names:
        path = base_dir / name
        path.write_bytes(b"fake")
        created.append(path)
    return created


def test_scan_transcribe_processes_supported_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_names = ["clip_a.wav", "clip_b.mp3", "ignore.txt"]
    _make_media_files(tmp_path, media_names)

    processed: List[str] = []

    def _fake_transcribe(self, file_path, **_):  # pylint: disable=unused-argument
        processed.append(Path(file_path).name)
        return {"file": str(file_path)}

    monkeypatch.setattr(PersianAudioTranscriber, "transcribe_file", _fake_transcribe)

    transcriber = PersianAudioTranscriber(normalize=False)
    results = transcriber.scan_and_transcribe(tmp_path)

    assert sorted(processed) == ["clip_a.wav", "clip_b.mp3"]
    assert len(results) == 2


def test_scan_transcribe_skips_existing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_file = tmp_path / "interview.wav"
    media_file.write_bytes(b"fake")
    existing_output = tmp_path / "interview.txt"
    existing_output.write_text("cached")

    def _fail_transcribe(*_):  # pragma: no cover
        raise AssertionError("transcribe_file should not be called when output exists")

    monkeypatch.setattr(PersianAudioTranscriber, "transcribe_file", _fail_transcribe)

    transcriber = PersianAudioTranscriber(normalize=False)
    results = transcriber.scan_and_transcribe(tmp_path, skip_existing=True)

    assert results == []


def test_scan_transcribe_progress_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_names = ["session1.wav", "session2.wav"]
    _make_media_files(tmp_path, media_names)

    monkeypatch.setattr(
        PersianAudioTranscriber,
        "transcribe_file",
        lambda self, file_path, **_: {"file": str(file_path)},
    )

    callbacks: List[Tuple[int, int, str]] = []

    def _progress(current: int, total: int, filename: str) -> None:
        callbacks.append((current, total, filename))

    transcriber = PersianAudioTranscriber(normalize=False)
    transcriber.scan_and_transcribe(tmp_path, progress_callback=_progress)

    assert callbacks == [
        (1, 2, "session1.wav"),
        (2, 2, "session2.wav"),
    ]


def test_scan_transcribe_collects_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = _make_media_files(tmp_path, ["good.wav", "bad.wav"])

    def _fake_transcribe(self, file_path, **_):
        if Path(file_path) == files[1]:
            raise ValueError("boom")
        return {"file": str(file_path), "success": True}

    monkeypatch.setattr(PersianAudioTranscriber, "transcribe_file", _fake_transcribe)

    transcriber = PersianAudioTranscriber(normalize=False)
    results = transcriber.scan_and_transcribe(tmp_path)

    assert any(r.get("error") == "boom" and r.get("success") is False for r in results)
    assert any(r.get("file") == str(files[0]) for r in results)
