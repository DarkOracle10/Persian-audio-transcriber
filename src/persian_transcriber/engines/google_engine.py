"""
Google Speech Recognition transcription engine.

This module provides transcription using Google's Speech Recognition
service via the SpeechRecognition library.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from ..utils.exceptions import APIError, EngineError
from .base import BaseEngine, EngineType, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class GoogleEngine(BaseEngine):
    """
    Transcription engine using Google Speech Recognition.

    This engine uses the free Google Speech Recognition API via the
    SpeechRecognition library. Note that it has limited Persian support
    compared to Whisper-based engines.

    Limitations:
    - Maximum audio length is typically limited
    - Persian support may be less accurate than Whisper
    - Requires internet connection
    - No word-level timestamps

    Example:
        >>> engine = GoogleEngine()
        >>> result = engine.transcribe("audio.wav", language="fa-IR")
        >>> print(result.text)
    """

    # Supported languages (subset - Google supports many more)
    SUPPORTED_LANGUAGES: List[str] = [
        "fa-IR",  # Persian (Iran)
        "ar-SA",  # Arabic (Saudi Arabia)
        "en-US",  # English (US)
        "en-GB",  # English (UK)
        "de-DE",  # German
        "es-ES",  # Spanish
        "fr-FR",  # French
        "it-IT",  # Italian
        "ja-JP",  # Japanese
        "ko-KR",  # Korean
        "pt-BR",  # Portuguese (Brazil)
        "ru-RU",  # Russian
        "zh-CN",  # Chinese (Simplified)
        "tr-TR",  # Turkish
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "fa-IR",
        show_all: bool = False,
    ) -> None:
        """
        Initialize the Google Speech Recognition engine.

        Args:
            api_key: Optional Google Cloud API key. If None, uses the free
                    Google Speech Recognition (which has usage limits).
            language: Default language code (e.g., "fa-IR" for Persian).
            show_all: If True, return all possible transcriptions.
        """
        super().__init__()
        self.api_key = api_key
        self.default_language = language
        self.show_all = show_all
        self._recognizer: Any = None

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "Google Speech Recognition"

    @property
    def engine_type(self) -> EngineType:
        """Get the engine type."""
        return EngineType.GOOGLE

    def load_model(self) -> None:
        """
        Initialize the speech recognizer.

        Raises:
            EngineError: If SpeechRecognition is not installed.
        """
        if self.is_loaded:
            logger.debug("Google recognizer already initialized")
            return

        try:
            import speech_recognition as sr
        except ImportError as e:
            raise EngineError(
                "SpeechRecognition not installed. Run: pip install SpeechRecognition",
                engine_name=self.name,
            ) from e

        logger.info("Initializing Google Speech Recognition...")

        self._recognizer = sr.Recognizer()
        self._model = self._recognizer  # For is_loaded check
        self._is_loaded = True

        logger.info("Google Speech Recognition initialized")
        logger.warning(
            "Note: Google Speech Recognition has limited Persian support. "
            "For better Persian accuracy, use Whisper or Faster-Whisper engines."
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using Google Speech Recognition.

        Args:
            audio_path: Path to the audio file. Should be WAV format for
                       best compatibility.
            language: Language code (e.g., "fa-IR"). If None, uses default.
            **kwargs: Additional arguments (not used).

        Returns:
            TranscriptionResult: Transcription result.

        Raises:
            EngineError: If transcription fails.
            APIError: If the Google API returns an error.
        """
        if not self.is_loaded:
            self.load_model()

        import speech_recognition as sr

        _audio_path = Path(audio_path)
        if not _audio_path.exists():
            raise EngineError(
                f"Audio file not found: {_audio_path}",
                engine_name=self.name,
            )

        # Map short language codes to full codes
        lang = language or self.default_language
        lang = self._normalize_language_code(lang)

        logger.info(f"Transcribing with Google Speech Recognition: {_audio_path.name}")

        # Convert to WAV if needed
        processing_path = self._prepare_audio(str(audio_path))

        try:
            with sr.AudioFile(processing_path) as source:
                audio_data = self._recognizer.record(source)

            # Perform recognition
            if self.api_key:
                text = self._recognizer.recognize_google_cloud(
                    audio_data,
                    credentials_json=self.api_key,
                    language=lang,
                    show_all=self.show_all,
                )
            else:
                text = self._recognizer.recognize_google(
                    audio_data,
                    language=lang,
                    show_all=self.show_all,
                )

            # Handle show_all response
            if self.show_all and isinstance(text, dict):
                alternatives = text.get("alternative", [])
                if alternatives:
                    text = alternatives[0].get("transcript", "")
                else:
                    text = ""

            # Clean up temporary file if created
            if processing_path != str(audio_path):
                try:
                    Path(processing_path).unlink()
                except Exception:
                    pass

            return TranscriptionResult(
                text=text,
                text_raw=text,
                segments=(
                    [
                        TranscriptionSegment(
                            text=text,
                            start=0.0,
                            end=0.0,  # Google doesn't provide timestamps
                        )
                    ]
                    if text
                    else []
                ),
                language=lang,
                engine=self.name,
                model="google-speech-recognition",
                metadata={
                    "api_type": "cloud" if self.api_key else "free",
                },
            )

        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return TranscriptionResult(
                text="",
                text_raw="",
                segments=[],
                language=lang,
                engine=self.name,
                model="google-speech-recognition",
                metadata={"error": "Could not understand audio"},
            )
        except sr.RequestError as e:
            raise APIError(
                f"Google Speech Recognition request failed: {e}",
                api_name="Google Speech Recognition",
            ) from e
        except Exception as e:
            raise EngineError(
                f"Transcription failed: {e}",
                engine_name=self.name,
            ) from e

    def _normalize_language_code(self, lang: str) -> str:
        """
        Normalize language code to Google's format.

        Args:
            lang: Language code (e.g., "fa" or "fa-IR").

        Returns:
            str: Normalized language code (e.g., "fa-IR").
        """
        # Map short codes to full codes
        language_map = {
            "fa": "fa-IR",
            "ar": "ar-SA",
            "en": "en-US",
            "de": "de-DE",
            "es": "es-ES",
            "fr": "fr-FR",
            "it": "it-IT",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "pt": "pt-BR",
            "ru": "ru-RU",
            "zh": "zh-CN",
            "tr": "tr-TR",
        }

        return language_map.get(lang.lower(), lang)

    def _prepare_audio(self, audio_path: str) -> str:
        """
        Prepare audio file for Google Speech Recognition.

        Google works best with WAV files, so convert if needed.

        Args:
            audio_path: Path to the audio file.

        Returns:
            str: Path to the prepared audio file.
        """
        path = Path(audio_path)

        # If already WAV, use directly
        if path.suffix.lower() == ".wav":
            return audio_path

        # Convert to WAV
        try:
            from pydub import AudioSegment

            logger.debug(f"Converting {path.suffix} to WAV for Google Speech Recognition")

            audio = AudioSegment.from_file(audio_path)

            # Convert to mono, 16kHz for best recognition
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            )
            audio.export(temp_file.name, format="wav")

            return temp_file.name

        except ImportError:
            logger.warning(
                "pydub not installed. Audio conversion skipped. " "For best results, use WAV files."
            )
            return audio_path
        except Exception as e:
            logger.warning(f"Audio conversion failed: {e}. Using original file.")
            return audio_path

    def supports_language(self, language: str) -> bool:
        """
        Check if a language is supported.

        Args:
            language: Language code to check.

        Returns:
            bool: True if supported (note: Google supports many more than listed).
        """
        normalized = self._normalize_language_code(language)
        # Google supports many languages, so we're permissive here
        return True

    def get_supported_languages(self) -> List[str]:
        """
        Get list of known supported language codes.

        Returns:
            List[str]: List of supported language codes.
        """
        return self.SUPPORTED_LANGUAGES.copy()
