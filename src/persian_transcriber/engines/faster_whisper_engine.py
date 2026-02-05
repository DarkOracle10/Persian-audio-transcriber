"""
Faster-Whisper transcription engine.

This module provides the Faster-Whisper implementation using CTranslate2
for optimized inference with GPU acceleration.
"""

import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from ..utils.cuda_setup import get_best_device, get_compute_type, setup_cuda_paths
from ..utils.exceptions import EngineError, ModelLoadError
from .base import BaseEngine, EngineType, TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


# Initial prompts for different languages to guide the model
LANGUAGE_INITIAL_PROMPTS = {
    "fa": "این یک متن فارسی است.",
    "ar": "هذا نص عربي.",
    "zh": "这是中文文本。",
    "ja": "これは日本語のテキストです。",
    "ko": "이것은 한국어 텍스트입니다.",
}


def _remove_repetitions(text: str, threshold: int = 3) -> str:
    """
    Remove repeated phrases from transcription output.

    Args:
        text: The transcription text to clean
        threshold: Minimum number of repetitions to trigger removal

    Returns:
        Cleaned text with repetitions removed
    """
    if not text:
        return text

    # Pattern to find repeated phrases (3+ word sequences repeated)
    # This handles cases like "word word word" or "phrase phrase phrase"
    pattern = r"(\b.{3,50}?\b)(?:\s*\1){" + str(threshold - 1) + r",}"
    cleaned = re.sub(pattern, r"\1", text, flags=re.UNICODE)

    # Also handle simple word repetitions like "و و و و"
    word_pattern = r"(\b\S+\b)(?:\s+\1){" + str(threshold - 1) + r",}"
    cleaned = re.sub(word_pattern, r"\1", cleaned, flags=re.UNICODE)

    return cleaned


def _add_punctuation_breaks(text: str) -> str:
    """
    Add line breaks after Persian/Arabic punctuation for better readability.

    Args:
        text: The text to format

    Returns:
        Text with line breaks after sentence-ending punctuation
    """
    # Add line break after Persian/Arabic full stop, question mark, exclamation
    text = re.sub(r"([.؟!،])\s*", r"\1\n", text)
    # Clean up multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class FasterWhisperEngine(BaseEngine):
    """
    Transcription engine using Faster-Whisper (CTranslate2).

    Faster-Whisper is a reimplementation of Whisper using CTranslate2,
    which provides significantly faster inference (4-8x) with lower
    memory usage and GPU acceleration via CUDA.

    Attributes:
        model_size: Size of the Whisper model to use.
        device: Compute device ("cuda", "cpu", "auto").
        compute_type: Precision type ("float16", "int8", "float32").

    Example:
        >>> engine = FasterWhisperEngine(model_size="large-v3")
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
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-large-v3",
        "distil-medium.en",
        "distil-small.en",
    ]

    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        cpu_threads: int = 4,
        num_workers: int = 1,
        download_root: Optional[str] = None,
    ) -> None:
        """
        Initialize the Faster-Whisper engine.

        Args:
            model_size: Size of the model to use. Options:
                       "tiny", "base", "small", "medium", "large-v3", etc.
                       For Persian, "medium" or larger is recommended.
            device: Device to run on ("cuda", "cpu", "auto", or None for auto-detect).
            compute_type: Computation precision type:
                         - "float16": Fast GPU inference (recommended for CUDA)
                         - "int8": CPU-optimized inference
                         - "float32": Full precision
                         If None, automatically selected based on device.
            cpu_threads: Number of CPU threads for inference (when using CPU).
            num_workers: Number of workers for parallel processing.
            download_root: Directory to download/cache models.
        """
        super().__init__()
        self.model_size = model_size
        self._requested_device = device
        self._requested_compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self.download_root = download_root

        # Actual values set during model loading
        self._actual_device: str = "cpu"
        self._actual_compute_type: str = "int8"

    @property
    def name(self) -> str:
        """Get the engine name."""
        return "Faster-Whisper"

    @property
    def engine_type(self) -> EngineType:
        """Get the engine type."""
        return EngineType.FASTER_WHISPER

    @property
    def device(self) -> str:
        """Get the actual device being used."""
        return self._actual_device

    @property
    def compute_type(self) -> str:
        """Get the actual compute type being used."""
        return self._actual_compute_type

    def load_model(self) -> None:
        """
        Load the Faster-Whisper model.

        This method sets up CUDA paths (if needed) and loads the model
        with appropriate device and compute type settings.

        Raises:
            ModelLoadError: If the model fails to load.
        """
        if self.is_loaded:
            logger.debug("Faster-Whisper model already loaded")
            return

        # Setup CUDA paths before importing faster_whisper
        setup_cuda_paths()

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ModelLoadError(
                self.model_size,
                engine_name=self.name,
                reason="faster-whisper not installed. Run: pip install faster-whisper",
            ) from e

        logger.info(f"Loading Faster-Whisper {self.model_size} model...")

        # Determine device
        if self._requested_device is None or self._requested_device == "auto":
            self._actual_device = get_best_device()
        else:
            self._actual_device = self._requested_device

        # Determine compute type
        if self._requested_compute_type is None:
            self._actual_compute_type = get_compute_type(self._actual_device)
        else:
            self._actual_compute_type = self._requested_compute_type

        # Try to load model with GPU first, fallback to CPU if needed
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self._actual_device,
                compute_type=self._actual_compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
                download_root=self.download_root,
            )
            self._is_loaded = True

            logger.info(
                f"Faster-Whisper {self.model_size} loaded on {self._actual_device} "
                f"with {self._actual_compute_type} precision"
            )

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a CUDA/GPU error
            if self._actual_device == "cuda" and any(
                x in error_msg for x in ["cuda", "cudnn", "gpu", "dll", "cublas"]
            ):
                logger.warning(f"GPU loading failed: {e}")
                logger.info("Falling back to CPU...")

                # Retry with CPU
                self._actual_device = "cpu"
                self._actual_compute_type = "int8"

                try:
                    self._model = WhisperModel(
                        self.model_size,
                        device=self._actual_device,
                        compute_type=self._actual_compute_type,
                        cpu_threads=self.cpu_threads,
                        num_workers=self.num_workers,
                        download_root=self.download_root,
                    )
                    self._is_loaded = True

                    logger.info(
                        f"Faster-Whisper {self.model_size} loaded on CPU "
                        f"with {self._actual_compute_type} precision"
                    )

                except Exception as cpu_e:
                    raise ModelLoadError(
                        self.model_size,
                        engine_name=self.name,
                        reason=f"CPU fallback also failed: {cpu_e}",
                    ) from cpu_e
            else:
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
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        temperature: Union[float, List[float]] = 0.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,  # Enable by default for hallucination detection
        vad_filter: bool = True,  # Enable by default for better segmentation
        vad_parameters: Optional[dict] = None,
        repetition_penalty: float = 1.1,  # Anti-repetition
        no_repeat_ngram_size: int = 3,  # Prevent n-gram repetition
        hallucination_silence_threshold: Optional[float] = 2.0,  # Skip hallucinations in silence
        **kwargs: Any,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using Faster-Whisper.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., "fa" for Persian).
            task: Task type - "transcribe" or "translate" (to English).
            beam_size: Beam size for decoding.
            best_of: Number of candidates when sampling.
            patience: Beam search patience factor.
            length_penalty: Exponential length penalty.
            temperature: Sampling temperature (0 for greedy).
            compression_ratio_threshold: Threshold for gzip compression ratio.
            log_prob_threshold: Threshold for average log probability.
            no_speech_threshold: Threshold for no_speech probability.
            condition_on_previous_text: Use previous output as prompt.
            initial_prompt: Optional initial prompt for the model.
            word_timestamps: Extract word-level timestamps (enabled by default).
            vad_filter: Enable voice activity detection filter (enabled by default).
            vad_parameters: Parameters for VAD filter.
            repetition_penalty: Penalty for repeated tokens (1.1 recommended).
            no_repeat_ngram_size: Size of n-grams to prevent repetition.
            hallucination_silence_threshold: Skip text during silence periods.
            **kwargs: Additional arguments.

        Returns:
            TranscriptionResult: Transcription result with text and segments.

        Raises:
            EngineError: If transcription fails.
        """
        if not self.is_loaded:
            self.load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise EngineError(
                f"Audio file not found: {audio_path}",
                engine_name=self.name,
            )

        logger.info(f"Transcribing with Faster-Whisper: {audio_path.name}")

        try:
            # Set default VAD parameters if not provided
            if vad_parameters is None:
                vad_parameters = {
                    "min_silence_duration_ms": 500,
                }

            # Auto-set initial prompt for supported languages if not provided
            if initial_prompt is None and language in LANGUAGE_INITIAL_PROMPTS:
                initial_prompt = LANGUAGE_INITIAL_PROMPTS[language]
                logger.debug(f"Using default initial prompt for {language}: {initial_prompt}")

            # Run transcription with anti-repetition settings
            segments_generator, info = self._model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                length_penalty=length_penalty,
                temperature=temperature,
                compression_ratio_threshold=compression_ratio_threshold,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                hallucination_silence_threshold=hallucination_silence_threshold,
            )

            # Collect segments
            segments: List[TranscriptionSegment] = []
            full_text_parts: List[str] = []

            for seg in segments_generator:
                segment_text = seg.text.strip()
                # Apply post-processing to remove repetitions
                segment_text = _remove_repetitions(segment_text)
                full_text_parts.append(segment_text)

                # Build word list if available
                words = None
                if word_timestamps and hasattr(seg, "words") and seg.words:
                    words = [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in seg.words
                    ]

                segments.append(
                    TranscriptionSegment(
                        text=segment_text,
                        start=seg.start,
                        end=seg.end,
                        confidence=seg.avg_logprob if hasattr(seg, "avg_logprob") else None,
                        words=words,
                    )
                )

            # Join and apply final post-processing
            full_text = " ".join(full_text_parts)
            full_text = _remove_repetitions(full_text)

            # Apply formatting for RTL languages (Persian, Arabic)
            if language in ("fa", "ar"):
                full_text = _add_punctuation_breaks(full_text)

            return TranscriptionResult(
                text=full_text,
                text_raw=" ".join(full_text_parts),  # Keep raw without formatting
                segments=segments,
                language=info.language if info.language else language,
                language_probability=(
                    info.language_probability if hasattr(info, "language_probability") else None
                ),
                duration=(
                    info.duration
                    if hasattr(info, "duration")
                    else (segments[-1].end if segments else 0.0)
                ),
                engine=self.name,
                model=self.model_size,
                metadata={
                    "device": self._actual_device,
                    "compute_type": self._actual_compute_type,
                    "task": task,
                    "vad_filter": vad_filter,
                    "initial_prompt": initial_prompt,
                },
            )

        except Exception as e:
            raise EngineError(
                f"Transcription failed: {e}",
                engine_name=self.name,
            ) from e

    def detect_language(
        self,
        audio_path: str,
    ) -> tuple:
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
            # Transcribe with language detection
            segments, info = self._model.transcribe(
                audio_path,
                language=None,  # Auto-detect
            )
            # Consume the generator to get info
            for _ in segments:
                pass

            return info.language, info.language_probability

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "fa", 0.0
