"""
Persian Audio Transcriber - A powerful Persian speech recognition tool.

This package provides tools for transcribing Persian (Farsi) audio and video
files using various speech recognition engines including Whisper, Faster-Whisper,
OpenAI API, and Google Speech Recognition.

Basic Usage:
    >>> from persian_transcriber import PersianAudioTranscriber
    >>> transcriber = PersianAudioTranscriber(model_size="medium")
    >>> result = transcriber.transcribe_file("audio.mp3")
    >>> print(result["text"])

Quick Transcription:
    >>> from persian_transcriber import transcribe_file
    >>> result = transcribe_file("audio.mp3")
    >>> print(result["text"])

Configuration:
    >>> from persian_transcriber import TranscriberConfig, PersianAudioTranscriber
    >>> config = TranscriberConfig(language="fa")
    >>> transcriber = PersianAudioTranscriber(config=config)

Available Engines:
    - faster_whisper: CTranslate2-based Whisper (recommended, fastest)
    - whisper: Original OpenAI Whisper
    - openai_api: OpenAI API (requires API key)
    - google: Google Speech Recognition

Output Formats:
    - txt: Plain text
    - json: JSON with metadata
    - srt: SRT subtitles
    - vtt: WebVTT subtitles

For more information, see the documentation or visit the GitHub repository.
"""

__version__ = "2.0.0"
__author__ = "Persian Transcriber Contributors"
__email__ = ""
__license__ = "MIT"

# Public API exports
from .transcriber import PersianAudioTranscriber, transcribe_file
from .config import (
    TranscriberConfig,
    EngineConfig,
    NormalizerConfig,
    OutputConfig,
    DeviceType,
)
from .engines.base import EngineType, TranscriptionResult, TranscriptionSegment
from .normalizers import NormalizerType
from .output import OutputFormat

# Exception classes
from .utils.exceptions import (
    TranscriberError,
    EngineError,
    AudioProcessingError,
    ConfigurationError,
    CUDAError,
    APIError,
    NormalizationError,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Main classes
    "PersianAudioTranscriber",
    "transcribe_file",
    # Configuration
    "TranscriberConfig",
    "EngineConfig",
    "NormalizerConfig",
    "OutputConfig",
    "DeviceType",
    # Enums
    "EngineType",
    "NormalizerType",
    "OutputFormat",
    # Data classes
    "TranscriptionResult",
    "TranscriptionSegment",
    # Exceptions
    "TranscriberError",
    "EngineError",
    "AudioProcessingError",
    "ConfigurationError",
    "CUDAError",
    "APIError",
    "NormalizationError",
]
