"""
OpenAI Whisper transcription engine.

This module provides the original OpenAI Whisper implementation
for audio transcription.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from ..utils.exceptions import EngineError, ModelLoadError
from .base import BaseEngine, EngineType, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class WhisperEngine(BaseEngine):
    """
    Transcription engine using OpenAI's Whisper model.

    This engine uses the original OpenAI Whisper implementation.
    For faster inference, consider using FasterWhisperEngine instead.

    Attributes:
        model_size: Size of the Whisper model to use.
        device: Compute device ("cuda", "cpu", or "auto").

    Example:
        >>> engine = WhisperEngine(model_size="medium")
        >>> engine.load_model()
        >>> result = engine.transcribe("audio.mp3", language="fa")
        >>> print(result.text)
    """

    # Available model sizes
    AVAILABLE_MODELS: List[str] = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
        "turbo",
    ]

    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        download_root: Optional[str] = None,
    ) -> None:
        """
        Initialize the Whisper engine.

        Args:
            model_size: Size of the model to use. Options:
                       "tiny", "base", "small", "medium", "large", "large-v3", "turbo"
                       For Persian, "medium" or larger is recommended.
            device: Device to run on ("cuda", "cpu", or None for auto-detect).
            download_root: Directory to download/cache models. Uses default if None.
        """
        super().__init__()
        self.model_size = model_size
        self.device = device
        self.download_root = download_root
        self._actual_device: str = "cpu"

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "Whisper"

    @property
    def engine_type(self) -> EngineType:
        """Get the engine type."""
        return EngineType.WHISPER

    def load_model(self) -> None:
        """
        Load the Whisper model.

        Raises:
            ModelLoadError: If the model fails to load.
        """
        if self.is_loaded:
            logger.debug("Whisper model already loaded")
            return

        try:
            import whisper
        except ImportError as e:
            raise ModelLoadError(
                self.model_size,
                engine_name=self.name,
                reason="openai-whisper not installed. Run: pip install openai-whisper",
            ) from e

        logger.info(f"Loading Whisper {self.model_size} model...")

        try:
            # Determine device
            if self.device is None:
                try:
                    import torch

                    self._actual_device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    self._actual_device = "cpu"
            else:
                self._actual_device = self.device

            # Load model
            self._model = whisper.load_model(
                self.model_size,
                device=self._actual_device,
                download_root=self.download_root,
            )
            self._is_loaded = True

            logger.info(f"Whisper {self.model_size} model loaded on {self._actual_device}")

        except Exception as e:
            raise ModelLoadError(
                self.model_size,
                engine_name=self.name,
                reason=str(e),
            ) from e

    def transcribe(
        self,
        audio_path: str,
        language: str = "fa",
        task: str = "transcribe",
        verbose: bool = False,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using Whisper.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., "fa" for Persian).
            task: Task type - "transcribe" or "translate" (to English).
            verbose: Whether to print progress during transcription.
            temperature: Sampling temperature. 0 for deterministic output.
            **kwargs: Additional arguments passed to whisper.transcribe().

        Returns:
            TranscriptionResult: Transcription result with text and segments.

        Raises:
            EngineError: If transcription fails.
        """
        if not self.is_loaded:
            self.load_model()

        _audio_path = Path(audio_path)
        if not _audio_path.exists():
            raise EngineError(
                f"Audio file not found: {_audio_path}",
                engine_name=self.name,
            )

        logger.info(f"Transcribing with Whisper: {_audio_path.name}")

        try:
            result = self._model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                verbose=verbose,
                temperature=temperature,
                **kwargs,
            )

            # Convert segments
            segments: List[TranscriptionSegment] = []
            for seg in result.get("segments", []):
                segments.append(
                    TranscriptionSegment(
                        text=seg.get("text", "").strip(),
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        confidence=seg.get("avg_logprob"),
                    )
                )

            # Calculate total duration
            duration = 0.0
            if segments:
                duration = segments[-1].end

            return TranscriptionResult(
                text=result.get("text", "").strip(),
                text_raw=result.get("text", "").strip(),
                segments=segments,
                language=result.get("language", language),
                duration=duration,
                engine=self.name,
                model=self.model_size,
                metadata={
                    "device": self._actual_device,
                    "task": task,
                },
            )

        except Exception as e:
            raise EngineError(
                f"Transcription failed: {e}",
                engine_name=self.name,
            ) from e

    def detect_language(self, audio_path: str) -> tuple:
        """
        Detect the language of an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            tuple: (language_code, probability)
        """
        if not self.is_loaded:
            self.load_model()

        try:
            import whisper

            # Load audio and pad/trim to 30 seconds
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # Make log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self._model.device)

            # Detect language
            _, probs = self._model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)

            return detected_lang, probs[detected_lang]

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "fa", 0.0
