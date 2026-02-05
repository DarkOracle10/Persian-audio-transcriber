"""
Custom exceptions for the Persian Transcriber package.

This module defines a hierarchy of exceptions used throughout the package
to provide clear and specific error information.
"""


class TranscriberError(Exception):
    """
    Base exception for all persian-transcriber errors.

    All custom exceptions in this package inherit from this class,
    allowing users to catch all package-specific errors with a single
    except clause if desired.

    Example:
        >>> try:
        ...     transcriber.transcribe_file("nonexistent.mp3")
        ... except TranscriberError as e:
        ...     print(f"Transcription failed: {e}")
    """

    def __init__(self, message: str, *args, **kwargs) -> None:
        self.message = message
        super().__init__(message, *args, **kwargs)

    def __str__(self) -> str:
        return self.message


class EngineError(TranscriberError):
    """
    Exception raised when a transcription engine encounters an error.

    This includes errors during:
    - Engine initialization
    - Model loading
    - Transcription processing

    Attributes:
        engine_name: Name of the engine that raised the error.
        message: Description of what went wrong.
    """

    def __init__(self, message: str, engine_name: str = "unknown") -> None:
        self.engine_name = engine_name
        super().__init__(f"[{engine_name}] {message}")


class EngineNotFoundError(EngineError):
    """Exception raised when a requested engine is not available."""

    def __init__(self, engine_name: str) -> None:
        super().__init__(
            f"Engine '{engine_name}' not found. Check that required dependencies are installed.",
            engine_name=engine_name,
        )


class ModelLoadError(EngineError):
    """Exception raised when a model fails to load."""

    def __init__(self, model_name: str, engine_name: str = "unknown", reason: str = "") -> None:
        self.model_name = model_name
        message = f"Failed to load model '{model_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, engine_name=engine_name)


class AudioProcessingError(TranscriberError):
    """
    Exception raised when audio processing fails.

    This includes errors during:
    - Audio file loading
    - Format conversion
    - Audio extraction from video

    Attributes:
        file_path: Path to the audio file that caused the error.
    """

    def __init__(self, message: str, file_path: str = "") -> None:
        self.file_path = file_path
        if file_path:
            message = f"{message} (file: {file_path})"
        super().__init__(message)


class UnsupportedFormatError(AudioProcessingError):
    """Exception raised when an audio/video format is not supported."""

    def __init__(self, file_path: str, format_ext: str) -> None:
        self.format_ext = format_ext
        super().__init__(
            f"Unsupported audio/video format: '{format_ext}'",
            file_path=file_path,
        )


class FileNotFoundError(AudioProcessingError):
    """Exception raised when an audio/video file is not found."""

    def __init__(self, file_path: str) -> None:
        super().__init__(f"File not found: '{file_path}'", file_path=file_path)


class CUDAError(TranscriberError):
    """
    Exception raised when CUDA/GPU operations fail.

    This includes errors during:
    - CUDA initialization
    - GPU detection
    - CUDA library loading
    """

    def __init__(self, message: str, cuda_version: str = "") -> None:
        self.cuda_version = cuda_version
        if cuda_version:
            message = f"{message} (CUDA version: {cuda_version})"
        super().__init__(message)


class ConfigurationError(TranscriberError):
    """
    Exception raised when configuration is invalid.

    This includes errors during:
    - Config file parsing
    - Invalid parameter values
    - Missing required settings
    """

    def __init__(self, message: str, config_key: str = "") -> None:
        self.config_key = config_key
        if config_key:
            message = f"Configuration error for '{config_key}': {message}"
        super().__init__(message)


class APIError(TranscriberError):
    """
    Exception raised when an external API call fails.

    This includes errors from:
    - OpenAI API
    - Google Speech API
    - Other cloud services

    Attributes:
        status_code: HTTP status code if available.
        api_name: Name of the API that failed.
    """

    def __init__(
        self,
        message: str,
        api_name: str = "unknown",
        status_code: int = 0,
    ) -> None:
        self.api_name = api_name
        self.status_code = status_code
        prefix = f"[{api_name}]"
        if status_code:
            prefix += f" (HTTP {status_code})"
        super().__init__(f"{prefix} {message}")


class RateLimitError(APIError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, api_name: str, retry_after: int = 0) -> None:
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(message, api_name=api_name, status_code=429)


class AuthenticationError(APIError):
    """Exception raised when API authentication fails."""

    def __init__(self, api_name: str) -> None:
        super().__init__(
            "Authentication failed. Check your API key.",
            api_name=api_name,
            status_code=401,
        )


class NormalizationError(TranscriberError):
    """Exception raised when text normalization fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
