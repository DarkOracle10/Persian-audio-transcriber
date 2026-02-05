"""
Main transcriber module.

This module provides the PersianAudioTranscriber class, which orchestrates
the transcription pipeline including engine selection, normalization, and
output formatting.
"""

import glob
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

from .config import TranscriberConfig, DeviceType, EngineConfig, NormalizerConfig, OutputConfig
from .engines import get_engine, EngineType
from .engines.base import BaseEngine, TranscriptionResult, TranscriptionSegment
from .normalizers import get_normalizer, NormalizerType
from .normalizers.base import BaseNormalizer
from .output import get_formatter, OutputFormat
from .output.base import BaseFormatter
from .utils.audio import prepare_audio_for_transcription
from .utils.cuda_setup import setup_cuda_paths
from .utils.exceptions import (
    AudioProcessingError,
    ConfigurationError,
    EngineError,
    TranscriberError,
)
from .utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


# Supported file extensions
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class PersianAudioTranscriber:
    """
    Persian audio and video transcription system.

    This class provides a unified interface for transcribing Persian (Farsi)
    audio and video files using various transcription engines.

    Features:
        - Multiple engine support (Whisper, Faster-Whisper, OpenAI API, Google)
        - Automatic Persian text normalization
        - GPU acceleration with automatic fallback
        - Multiple output formats (TXT, JSON, SRT, VTT)
        - Batch processing of directories

    Example:
        >>> transcriber = PersianAudioTranscriber(model_size="large-v3")
        >>> result = transcriber.transcribe_file("audio.mp3")
        >>> print(result["text"])

        >>> # With configuration object
        >>> config = TranscriberConfig(
        ...     engine=EngineConfig(type=EngineType.WHISPER, model_size="medium"),
        ...     output=OutputConfig(format=OutputFormat.SRT),
        ... )
        >>> transcriber = PersianAudioTranscriber(config=config)
    """

    def __init__(
        self,
        engine: Optional[Union[str, EngineType]] = None,
        model_size: Optional[str] = None,
        device: Optional[Union[str, DeviceType]] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        normalize: Optional[bool] = None,
        output_format: Optional[Union[str, OutputFormat]] = None,
        openai_api_key: Optional[str] = None,
        verbose: bool = False,
        config: Optional[TranscriberConfig] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            engine: Transcription engine type. Defaults to 'faster_whisper'.
            model_size: Model size for Whisper engines. Defaults to 'medium'.
            device: Computation device ('auto', 'cuda', 'cpu', 'mps').
            compute_type: Precision type for Faster Whisper.
            language: Language code. Defaults to 'fa'.
            normalize: Enable text normalization. Defaults to True.
            output_format: Default output format.
            openai_api_key: API key for OpenAI engine.
            verbose: Enable verbose logging.
            config: Configuration object (overrides other parameters).
        """
        # Use config object or create from parameters
        if config is not None:
            self._config = config
        else:
            self._config = TranscriberConfig(
                language=language or "fa",
                engine=EngineConfig(
                    type=engine or EngineType.FASTER_WHISPER,
                    model_size=model_size or "medium",
                    device=device or DeviceType.AUTO,
                    compute_type=compute_type,
                ),
                normalizer=NormalizerConfig(
                    enabled=normalize if normalize is not None else True,
                ),
                output=OutputConfig(
                    format=output_format or OutputFormat.TXT,
                ),
                openai_api_key=openai_api_key,
                verbose=verbose,
            )

        # Setup logging
        setup_logging(verbose=self._config.verbose)

        # Setup CUDA paths for GPU acceleration
        setup_cuda_paths()

        # Initialize components (lazy loaded)
        self._engine: Optional[BaseEngine] = None
        self._normalizer: Optional[BaseNormalizer] = None
        self._formatter: Optional[BaseFormatter] = None

        logger.info(f"Initialized PersianAudioTranscriber")
        logger.debug(f"Config: {self._config.to_dict()}")

    @property
    def config(self) -> TranscriberConfig:
        """Get current configuration."""
        return self._config

    @property
    def engine(self) -> BaseEngine:
        """Get or create the transcription engine (lazy initialization)."""
        if self._engine is None:
            engine_type = self._config.engine.type
            if isinstance(engine_type, str):
                engine_type = EngineType(engine_type)

            logger.info(f"Initializing {engine_type.value} engine...")

            self._engine = get_engine(
                engine_type=engine_type,
                model_size=self._config.engine.model_size,
                device=str(self._config.engine.device),
                compute_type=self._config.engine.compute_type,
                api_key=self._config.openai_api_key,
            )

        return self._engine

    @property
    def normalizer(self) -> Optional[BaseNormalizer]:
        """Get or create the text normalizer (lazy initialization)."""
        if self._normalizer is None and self._config.normalizer.enabled:
            normalizer_type = self._config.normalizer.type
            if isinstance(normalizer_type, str):
                normalizer_type = NormalizerType(normalizer_type)

            self._normalizer = get_normalizer(normalizer_type)
            logger.debug(f"Using normalizer: {self._normalizer.__class__.__name__}")

        return self._normalizer

    def _normalize_text(self, text: str) -> str:
        """Normalize text if normalization is enabled."""
        if self.normalizer is not None:
            return self.normalizer.normalize(text)
        return text

    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        output_format: Optional[Union[str, OutputFormat]] = None,
        output_path: Optional[Union[str, Path]] = None,
        save_output: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Transcribe a single audio or video file.

        Args:
            file_path: Path to the audio/video file.
            language: Language code override.
            output_format: Output format override.
            output_path: Custom output file path.
            save_output: Whether to save the output file.
            **kwargs: Additional arguments passed to the engine.

        Returns:
            Dictionary containing:
                - text: Full transcription text.
                - segments: List of transcription segments.
                - language: Detected/specified language.
                - duration: Audio duration in seconds.
                - output_path: Path to saved output (if save_output=True).
                - processing_time: Time taken to transcribe.

        Raises:
            FileNotFoundError: If the input file doesn't exist.
            AudioProcessingError: If audio extraction/conversion fails.
            EngineError: If transcription fails.
        """
        file_path = Path(file_path)
        start_time = time.time()

        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise AudioProcessingError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        logger.info(f"Transcribing: {file_path.name}")

        # Prepare audio for transcription
        try:
            audio_path_str, is_temp_file = prepare_audio_for_transcription(str(file_path))
            audio_path = Path(audio_path_str)
        except Exception as e:
            raise AudioProcessingError(f"Failed to prepare audio: {e}") from e

        try:
            # Perform transcription
            lang = language or self._config.language
            result = self.engine.transcribe(str(audio_path), language=lang, **kwargs)

            # Normalize text
            normalized_text = self._normalize_text(result.text)
            normalized_segments: List[Dict[str, Any]] = []

            for segment in result.segments:
                normalized_segment: Dict[str, Any] = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": self._normalize_text(segment.text),
                }
                normalized_segments.append(normalized_segment)

            # Prepare output
            output: Dict[str, Any] = {
                "text": normalized_text,
                "segments": normalized_segments,
                "language": result.language,
                "duration": result.duration,
                "engine": self.engine.name,
                "model": self._config.engine.model_size,
                "processing_time": time.time() - start_time,
            }

            # Save output if requested
            if save_output:
                fmt = output_format or self._config.output.format
                if isinstance(fmt, str):
                    fmt = OutputFormat(fmt)

                formatter = get_formatter(fmt)

                # Determine output path
                if output_path is None:
                    out_dir = self._config.output.directory or file_path.parent
                    output_path = out_dir / f"{file_path.stem}.{fmt.value}"
                else:
                    output_path = Path(output_path)

                # Create a result object with normalized text for saving
                save_result = TranscriptionResult(
                    text=normalized_text,
                    text_raw=result.text,
                    segments=[
                        TranscriptionSegment(
                            text=str(seg["text"]),
                            start=cast(float, seg["start"]),
                            end=cast(float, seg["end"]),
                        )
                        for seg in normalized_segments
                    ],
                    language=result.language,
                    duration=result.duration,
                    engine=self.engine.name,
                    model=self._config.engine.model_size,
                )
                formatter.save(save_result, output_path)
                output["output_path"] = str(output_path)
                logger.info(f"Saved output to: {output_path}")

            logger.info(f"Transcription completed in {output['processing_time']:.2f}s")

            return output

        finally:
            # Clean up temporary file
            if is_temp_file and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass

    def transcribe(
        self,
        source: Union[str, Path],
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Transcribe a file or directory.

        This is a convenience method that automatically detects whether
        the source is a file or directory.

        Args:
            source: Path to file or directory.
            language: Language code override.
            **kwargs: Additional arguments passed to transcribe_file.

        Returns:
            For single file: Transcription result dictionary.
            For directory: List of transcription result dictionaries.
        """
        source = Path(source)

        if source.is_file():
            return self.transcribe_file(source, language=language, **kwargs)
        elif source.is_dir():
            return self.scan_and_transcribe(source, language=language, **kwargs)
        else:
            raise FileNotFoundError(f"Path not found: {source}")

    def scan_and_transcribe(
        self,
        directory: Union[str, Path],
        language: Optional[str] = None,
        recursive: bool = False,
        output_format: Optional[Union[str, OutputFormat]] = None,
        output_directory: Optional[Union[str, Path]] = None,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Scan a directory and transcribe all supported media files.

        Args:
            directory: Directory to scan for media files.
            language: Language code override.
            recursive: Search subdirectories recursively.
            output_format: Output format for all files.
            output_directory: Directory to save all output files.
            skip_existing: Skip files that already have transcription output.
            progress_callback: Callback function(current, total, filename).
            **kwargs: Additional arguments passed to transcribe_file.

        Returns:
            List of transcription result dictionaries.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Collect all media files
        files: List[Path] = []

        for ext in SUPPORTED_EXTENSIONS:
            if recursive:
                pattern = f"**/*{ext}"
            else:
                pattern = f"*{ext}"

            files.extend(directory.glob(pattern))

        # Sort files for consistent ordering
        files = sorted(set(files))

        if not files:
            logger.warning(f"No media files found in {directory}")
            return []

        logger.info(f"Found {len(files)} media file(s) to transcribe")

        # Determine output format
        fmt = output_format or self._config.output.format
        if isinstance(fmt, str):
            fmt = OutputFormat(fmt)

        results: List[Dict[str, Any]] = []

        for idx, file_path in enumerate(files, 1):
            # Progress callback
            if progress_callback:
                progress_callback(idx, len(files), file_path.name)

            # Determine output path
            if output_directory:
                out_dir = Path(output_directory)
            else:
                out_dir = file_path.parent

            output_path = out_dir / f"{file_path.stem}.{fmt.value}"

            # Skip if output exists
            if skip_existing and output_path.exists():
                logger.info(f"Skipping (output exists): {file_path.name}")
                continue

            # Transcribe
            try:
                logger.info(f"[{idx}/{len(files)}] Processing: {file_path.name}")

                result = self.transcribe_file(
                    file_path,
                    language=language,
                    output_format=fmt,
                    output_path=output_path,
                    **kwargs,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to transcribe {file_path.name}: {e}")
                results.append(
                    {
                        "file": str(file_path),
                        "error": str(e),
                        "success": False,
                    }
                )

        # Summary
        successful = sum(1 for r in results if r.get("success", True) and "error" not in r)
        logger.info(f"Batch transcription complete: {successful}/{len(results)} successful")

        return results

    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get lists of supported file formats."""
        return {
            "audio": sorted(AUDIO_EXTENSIONS),
            "video": sorted(VIDEO_EXTENSIONS),
            "all": sorted(SUPPORTED_EXTENSIONS),
        }

    def get_available_engines(self) -> List[str]:
        """Get list of available engine types."""
        return [e.value for e in EngineType]

    def get_available_output_formats(self) -> List[str]:
        """Get list of available output formats."""
        return [f.value for f in OutputFormat]

    def __repr__(self) -> str:
        return (
            f"PersianAudioTranscriber("
            f"engine={self._config.engine.type}, "
            f"model={self._config.engine.model_size}, "
            f"language={self._config.language})"
        )


def transcribe_file(
    file_path: Union[str, Path],
    engine: str = "faster_whisper",
    model_size: str = "medium",
    language: str = "fa",
    output_format: str = "txt",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience function for quick transcription.

    This function creates a temporary transcriber instance for
    one-off transcriptions.

    Args:
        file_path: Path to audio/video file.
        engine: Engine type.
        model_size: Model size.
        language: Language code.
        output_format: Output format.
        **kwargs: Additional arguments.

    Returns:
        Transcription result dictionary.

    Example:
        >>> result = transcribe_file("audio.mp3", model_size="large-v3")
        >>> print(result["text"])
    """
    transcriber = PersianAudioTranscriber(
        engine=engine,
        model_size=model_size,
        language=language,
        output_format=output_format,
    )

    return transcriber.transcribe_file(file_path, **kwargs)
