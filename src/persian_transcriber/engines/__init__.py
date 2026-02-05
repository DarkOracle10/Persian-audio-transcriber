"""
Transcription engines for Persian Transcriber.

This package provides multiple transcription engine implementations:

- WhisperEngine: Original OpenAI Whisper implementation
- FasterWhisperEngine: Optimized CTranslate2 implementation (recommended)
- OpenAIAPIEngine: Cloud-based OpenAI Whisper API
- GoogleEngine: Google Speech Recognition

Example:
    >>> from persian_transcriber.engines import get_engine, EngineType
    >>> engine = get_engine(EngineType.FASTER_WHISPER, model_size="large-v3")
    >>> engine.load_model()
    >>> result = engine.transcribe("audio.mp3", language="fa")
    >>> print(result.text)
"""

from typing import Any, Optional, Union

from .base import (
    BaseEngine,
    EngineType,
    TranscriptionResult,
    TranscriptionSegment,
)
from .faster_whisper_engine import FasterWhisperEngine
from .google_engine import GoogleEngine
from .openai_api_engine import OpenAIAPIEngine
from .whisper_engine import WhisperEngine

__all__ = [
    # Base classes and types
    "BaseEngine",
    "EngineType",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Engine implementations
    "WhisperEngine",
    "FasterWhisperEngine",
    "OpenAIAPIEngine",
    "GoogleEngine",
    # Factory function
    "get_engine",
]


def get_engine(
    engine_type: Union[str, EngineType] = EngineType.FASTER_WHISPER,
    model_size: str = "medium",
    api_key: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> BaseEngine:
    """
    Factory function to create a transcription engine.

    Args:
        engine_type: Type of engine to create. Options:
            - "whisper" or EngineType.WHISPER: Original OpenAI Whisper
            - "faster_whisper" or EngineType.FASTER_WHISPER: Faster-Whisper (recommended)
            - "openai_api" or EngineType.OPENAI_API: OpenAI cloud API
            - "google" or EngineType.GOOGLE: Google Speech Recognition
        model_size: Model size for Whisper engines (e.g., "tiny", "base", "small",
                   "medium", "large", "large-v3"). Ignored for API-based engines.
        api_key: API key for OpenAI API or Google Cloud. Can also be set via
                environment variables (OPENAI_API_KEY).
        device: Compute device ("cuda", "cpu", "auto"). Ignored for API-based engines.
        **kwargs: Additional engine-specific arguments.

    Returns:
        BaseEngine: A configured transcription engine instance.

    Raises:
        ValueError: If the engine type is not recognized.

    Examples:
        >>> # Create Faster-Whisper engine (recommended for local inference)
        >>> engine = get_engine("faster_whisper", model_size="large-v3")

        >>> # Create OpenAI API engine for cloud transcription
        >>> engine = get_engine("openai_api", api_key="sk-...")

        >>> # Create engine with explicit device selection
        >>> engine = get_engine("whisper", model_size="medium", device="cuda")
    """
    # Normalize engine type
    if isinstance(engine_type, str):
        engine_type = engine_type.lower().replace("-", "_")
        try:
            engine_type = EngineType(engine_type)
        except ValueError:
            pass

    # Create appropriate engine
    if engine_type in (EngineType.WHISPER, "whisper"):
        return WhisperEngine(
            model_size=model_size,
            device=device,
            **kwargs,
        )

    if engine_type in (EngineType.FASTER_WHISPER, "faster_whisper"):
        return FasterWhisperEngine(
            model_size=model_size,
            device=device,
            **kwargs,
        )

    if engine_type in (EngineType.OPENAI_API, "openai_api"):
        return OpenAIAPIEngine(
            api_key=api_key,
            **kwargs,
        )

    if engine_type in (EngineType.GOOGLE, "google"):
        return GoogleEngine(
            api_key=api_key,
            **kwargs,
        )

    raise ValueError(
        f"Unknown engine type: {engine_type}. "
        f"Available types: {', '.join(t.value for t in EngineType)}"
    )
