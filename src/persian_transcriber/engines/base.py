"""
Base engine interface for transcription engines.

This module defines the abstract base class and data structures
that all transcription engines must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EngineType(str, Enum):
    """Enumeration of available transcription engine types."""
    
    WHISPER = "whisper"
    FASTER_WHISPER = "faster_whisper"
    OPENAI_API = "openai_api"
    GOOGLE = "google"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class TranscriptionSegment:
    """
    A single segment of transcribed text with timing information.
    
    Attributes:
        text: The transcribed text for this segment.
        start: Start time in seconds.
        end: End time in seconds.
        confidence: Confidence score (0.0 to 1.0), if available.
        words: Optional list of word-level timings.
    """
    
    text: str
    start: float
    end: float
    confidence: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    
    @property
    def duration(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary representation."""
        result = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.words is not None:
            result["words"] = self.words
        return result


@dataclass
class TranscriptionResult:
    """
    Complete result from a transcription operation.
    
    Attributes:
        text: The full transcribed text (normalized if applicable).
        text_raw: The original transcribed text before normalization.
        segments: List of transcription segments with timing.
        language: Detected or specified language code.
        language_probability: Confidence of language detection (0.0 to 1.0).
        duration: Total audio duration in seconds.
        engine: Name of the engine that produced this result.
        model: Model identifier used for transcription.
        metadata: Additional metadata from the transcription engine.
    """
    
    text: str
    text_raw: str = ""
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = "fa"
    language_probability: Optional[float] = None
    duration: float = 0.0
    engine: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Set text_raw to text if not provided."""
        if not self.text_raw:
            self.text_raw = self.text
    
    @property
    def word_count(self) -> int:
        """Get the approximate word count."""
        return len(self.text.split())
    
    @property
    def segment_count(self) -> int:
        """Get the number of segments."""
        return len(self.segments)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "text": self.text,
            "text_raw": self.text_raw,
            "segments": [seg.to_dict() for seg in self.segments],
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "engine": self.engine,
            "model": self.model,
            "metadata": self.metadata,
        }


class BaseEngine(ABC):
    """
    Abstract base class for all transcription engines.
    
    All engine implementations must inherit from this class and
    implement the required abstract methods.
    
    Example:
        >>> class MyEngine(BaseEngine):
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyEngine"
        ...     
        ...     def load_model(self) -> None:
        ...         # Load model implementation
        ...         pass
        ...     
        ...     def transcribe(self, audio_path: str, language: str = "fa") -> TranscriptionResult:
        ...         # Transcription implementation
        ...         pass
    """
    
    def __init__(self) -> None:
        """Initialize the base engine."""
        self._model: Any = None
        self._is_loaded: bool = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the human-readable name of this engine.
        
        Returns:
            str: Engine name (e.g., "Whisper", "Faster-Whisper").
        """
        pass
    
    @property
    def engine_type(self) -> EngineType:
        """
        Get the engine type enum.
        
        Returns:
            EngineType: The type of this engine.
        """
        # Default implementation - subclasses should override
        return EngineType.WHISPER
    
    @property
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.
        
        Returns:
            bool: True if model is loaded and ready for transcription.
        """
        return self._is_loaded and self._model is not None
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the transcription model into memory.
        
        This method should be called before transcribe() if the model
        is not automatically loaded during initialization.
        
        Raises:
            ModelLoadError: If the model fails to load.
        """
        pass
    
    def unload_model(self) -> None:
        """
        Unload the model to free memory.
        
        This is useful when switching between engines or when
        memory is constrained.
        """
        self._model = None
        self._is_loaded = False
    
    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str = "fa",
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file to transcribe.
            language: Language code for transcription (default: "fa" for Persian).
            **kwargs: Additional engine-specific options.
            
        Returns:
            TranscriptionResult: The transcription result with text and metadata.
            
        Raises:
            EngineError: If transcription fails.
            AudioProcessingError: If the audio file cannot be processed.
        """
        pass
    
    def supports_language(self, language: str) -> bool:
        """
        Check if this engine supports a specific language.
        
        Args:
            language: Language code to check (e.g., "fa", "en").
            
        Returns:
            bool: True if the language is supported.
        """
        # Default: assume all languages supported
        # Subclasses should override with actual support info
        return True
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List[str]: List of supported language codes.
        """
        # Default implementation - subclasses should override
        return ["fa", "en", "ar", "de", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"]
    
    def __repr__(self) -> str:
        loaded_str = "loaded" if self.is_loaded else "not loaded"
        return f"{self.__class__.__name__}({loaded_str})"
    
    def __enter__(self) -> "BaseEngine":
        """Context manager entry - load model."""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - unload model."""
        self.unload_model()
