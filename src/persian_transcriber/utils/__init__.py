"""
Utility modules for Persian Transcriber.

This package provides utility functions and classes used throughout
the persian-transcriber package, including:

- cuda_setup: Cross-platform GPU detection and configuration
- audio: Audio processing and format conversion
- logging: Logging configuration and utilities
- exceptions: Custom exception classes
"""

from .audio import (
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    cleanup_temp_file,
    convert_audio,
    extract_audio_from_video,
    get_audio_duration,
    is_supported_format,
    is_video_file,
    prepare_audio_for_transcription,
)
from .cuda_setup import (
    GPUInfo,
    ensure_cuda_initialized,
    get_best_device,
    get_compute_type,
    get_device_info,
    get_platform,
    is_cuda_available,
    is_mps_available,
    setup_cuda_paths,
)
from .exceptions import (
    APIError,
    AudioProcessingError,
    AuthenticationError,
    ConfigurationError,
    CUDAError,
    EngineError,
    EngineNotFoundError,
    ModelLoadError,
    RateLimitError,
    TranscriberError,
    UnsupportedFormatError,
)
from .logging import (
    disable_logging,
    enable_logging,
    get_logger,
    set_log_level,
    setup_logging,
)

__all__ = [
    # Audio utilities
    "SUPPORTED_AUDIO_FORMATS",
    "SUPPORTED_VIDEO_FORMATS",
    "SUPPORTED_FORMATS",
    "is_supported_format",
    "is_video_file",
    "get_audio_duration",
    "extract_audio_from_video",
    "convert_audio",
    "prepare_audio_for_transcription",
    "cleanup_temp_file",
    # CUDA utilities
    "GPUInfo",
    "get_platform",
    "setup_cuda_paths",
    "is_cuda_available",
    "is_mps_available",
    "get_best_device",
    "get_compute_type",
    "get_device_info",
    "ensure_cuda_initialized",
    # Exceptions
    "TranscriberError",
    "EngineError",
    "EngineNotFoundError",
    "ModelLoadError",
    "AudioProcessingError",
    "UnsupportedFormatError",
    "CUDAError",
    "ConfigurationError",
    "APIError",
    "RateLimitError",
    "AuthenticationError",
    # Logging
    "setup_logging",
    "get_logger",
    "set_log_level",
    "disable_logging",
    "enable_logging",
]
