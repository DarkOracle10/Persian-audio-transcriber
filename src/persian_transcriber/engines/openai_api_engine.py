"""
OpenAI API transcription engine.

This module provides transcription using the OpenAI Whisper API,
which runs inference in the cloud rather than locally.
"""

import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from ..utils.exceptions import APIError, AuthenticationError, EngineError, RateLimitError
from .base import BaseEngine, EngineType, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class OpenAIAPIEngine(BaseEngine):
    """
    Transcription engine using OpenAI's Whisper API.

    This engine uses the OpenAI cloud API for transcription, which
    requires an API key but doesn't need local GPU resources.

    Attributes:
        api_key: OpenAI API key.
        model: Model to use (currently only "whisper-1").

    Example:
        >>> engine = OpenAIAPIEngine(api_key="sk-...")
        >>> result = engine.transcribe("audio.mp3", language="fa")
        >>> print(result.text)
    """

    # Available models
    AVAILABLE_MODELS: List[str] = ["whisper-1"]

    # Maximum file size (25 MB)
    MAX_FILE_SIZE_MB: int = 25

    # Supported formats by the API
    SUPPORTED_FORMATS: List[str] = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
    ) -> None:
        """
        Initialize the OpenAI API engine.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model to use (default: "whisper-1").
            organization: Optional OpenAI organization ID.
            base_url: Optional custom API base URL.
            timeout: Request timeout in seconds (default: 300).
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout
        self._client: Any = None

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "OpenAI API"

    @property
    def engine_type(self) -> EngineType:
        """Get the engine type."""
        return EngineType.OPENAI_API

    def load_model(self) -> None:
        """
        Initialize the OpenAI client.

        Raises:
            AuthenticationError: If no API key is provided.
            ModelLoadError: If the OpenAI library is not installed.
        """
        if self.is_loaded:
            logger.debug("OpenAI client already initialized")
            return

        if not self.api_key:
            raise AuthenticationError("OpenAI API")

        try:
            from openai import OpenAI
        except ImportError as e:
            raise EngineError(
                "openai library not installed. Run: pip install openai",
                engine_name=self.name,
            ) from e

        logger.info("Initializing OpenAI API client...")

        try:
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            self._model = self.model  # For is_loaded check
            self._is_loaded = True

            logger.info("OpenAI API client initialized successfully")

        except Exception as e:
            raise EngineError(
                f"Failed to initialize OpenAI client: {e}",
                engine_name=self.name,
            ) from e

    def transcribe(
        self,
        audio_path: str,
        language: str = "fa",
        prompt: Optional[str] = None,
        response_format: str = "verbose_json",
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using OpenAI's Whisper API.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., "fa" for Persian).
            prompt: Optional prompt to guide the transcription.
            response_format: Response format - "json", "text", "srt", "verbose_json", "vtt".
            temperature: Sampling temperature (0-1).
            **kwargs: Additional arguments passed to the API.

        Returns:
            TranscriptionResult: Transcription result with text and segments.

        Raises:
            EngineError: If transcription fails.
            APIError: If the API returns an error.
            RateLimitError: If rate limit is exceeded.
        """
        if not self.is_loaded:
            self.load_model()

        _audio_path = Path(audio_path)
        if not _audio_path.exists():
            raise EngineError(
                f"Audio file not found: {_audio_path}",
                engine_name=self.name,
            )

        # Check file size
        file_size_mb = _audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.MAX_FILE_SIZE_MB:
            raise EngineError(
                f"File too large ({file_size_mb:.1f} MB). "
                f"Maximum size is {self.MAX_FILE_SIZE_MB} MB.",
                engine_name=self.name,
            )

        logger.info(f"Transcribing with OpenAI API: {_audio_path.name}")

        try:
            with open(_audio_path, "rb") as audio_file:
                response = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=language,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature,
                )

            # Parse response based on format
            if response_format == "verbose_json":
                text = response.text
                segments = self._parse_segments(response)
                duration = response.duration if hasattr(response, "duration") else 0.0
                detected_language = response.language if hasattr(response, "language") else language
            else:
                text = response if isinstance(response, str) else str(response)
                segments = []
                duration = 0.0
                detected_language = language

            return TranscriptionResult(
                text=text,
                text_raw=text,
                segments=segments,
                language=detected_language,
                duration=duration,
                engine=self.name,
                model=self.model,
                metadata={
                    "response_format": response_format,
                    "api_model": self.model,
                },
            )

        except Exception as e:
            self._handle_api_error(e)
            raise EngineError(
                f"Transcription failed: {e}",
                engine_name=self.name,
            ) from e

    def _parse_segments(self, response: Any) -> List[TranscriptionSegment]:
        """
        Parse segments from verbose_json response.

        Args:
            response: API response object.

        Returns:
            List[TranscriptionSegment]: Parsed segments.
        """
        segments: List[TranscriptionSegment] = []

        if hasattr(response, "segments") and response.segments:
            for seg in response.segments:
                segments.append(
                    TranscriptionSegment(
                        text=(
                            seg.get("text", "").strip()
                            if isinstance(seg, dict)
                            else seg.text.strip()
                        ),
                        start=seg.get("start", 0.0) if isinstance(seg, dict) else seg.start,
                        end=seg.get("end", 0.0) if isinstance(seg, dict) else seg.end,
                        confidence=(
                            seg.get("avg_logprob")
                            if isinstance(seg, dict)
                            else getattr(seg, "avg_logprob", None)
                        ),
                    )
                )

        return segments

    def _handle_api_error(self, error: Exception) -> None:
        """
        Handle OpenAI API errors and raise appropriate exceptions.

        Args:
            error: The exception that occurred.

        Raises:
            AuthenticationError: For authentication failures.
            RateLimitError: For rate limit errors.
            APIError: For other API errors.
        """
        error_str = str(error).lower()

        if "authentication" in error_str or "api key" in error_str or "401" in error_str:
            raise AuthenticationError("OpenAI API")

        if "rate limit" in error_str or "429" in error_str:
            raise RateLimitError("OpenAI API")

        if "api" in error_str:
            raise APIError(str(error), api_name="OpenAI API")

    def translate(
        self,
        audio_path: str,
        prompt: Optional[str] = None,
        response_format: str = "verbose_json",
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """
        Translate audio to English using OpenAI's API.

        Args:
            audio_path: Path to the audio file.
            prompt: Optional prompt to guide the translation.
            response_format: Response format.
            temperature: Sampling temperature.

        Returns:
            TranscriptionResult: Translation result.
        """
        if not self.is_loaded:
            self.load_model()

        _audio_path = Path(audio_path)
        if not _audio_path.exists():
            raise EngineError(
                f"Audio file not found: {_audio_path}",
                engine_name=self.name,
            )

        logger.info(f"Translating with OpenAI API: {_audio_path.name}")

        try:
            with open(_audio_path, "rb") as audio_file:
                response = self._client.audio.translations.create(
                    model=self.model,
                    file=audio_file,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature,
                )

            if response_format == "verbose_json":
                text = response.text
                segments = self._parse_segments(response)
                duration = response.duration if hasattr(response, "duration") else 0.0
            else:
                text = response if isinstance(response, str) else str(response)
                segments = []
                duration = 0.0

            return TranscriptionResult(
                text=text,
                text_raw=text,
                segments=segments,
                language="en",  # Translation is always to English
                duration=duration,
                engine=self.name,
                model=self.model,
                metadata={
                    "response_format": response_format,
                    "task": "translate",
                },
            )

        except Exception as e:
            self._handle_api_error(e)
            raise EngineError(
                f"Translation failed: {e}",
                engine_name=self.name,
            ) from e
