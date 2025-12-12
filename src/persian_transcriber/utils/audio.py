"""
Audio processing utilities.

This module provides utilities for audio file handling, including:
- Audio extraction from video files
- Format conversion
- Audio duration detection
- Supported format validation
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Set, Tuple

from .exceptions import AudioProcessingError, UnsupportedFormatError

logger = logging.getLogger(__name__)


# Supported audio formats
SUPPORTED_AUDIO_FORMATS: Set[str] = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".opus",
    ".webm",
}

# Supported video formats (audio will be extracted)
SUPPORTED_VIDEO_FORMATS: Set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
}

# All supported formats
SUPPORTED_FORMATS: Set[str] = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS


def is_supported_format(file_path: str) -> bool:
    """
    Check if a file format is supported.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        bool: True if the format is supported, False otherwise.
    
    Example:
        >>> is_supported_format("audio.mp3")
        True
        >>> is_supported_format("document.pdf")
        False
    """
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS


def is_video_file(file_path: str) -> bool:
    """
    Check if a file is a video file.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        bool: True if the file is a video, False otherwise.
    """
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_VIDEO_FORMATS


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file.
        
    Returns:
        float: Duration in seconds.
        
    Raises:
        AudioProcessingError: If the duration cannot be determined.
    
    Example:
        >>> duration = get_audio_duration("audio.mp3")
        >>> print(f"Duration: {duration:.2f} seconds")
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except ImportError:
        logger.warning("pydub not installed, cannot get audio duration")
        raise AudioProcessingError(
            "pydub library not installed. Run: pip install pydub",
            file_path=file_path,
        )
    except Exception as e:
        raise AudioProcessingError(
            f"Failed to get audio duration: {e}",
            file_path=file_path,
        )


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file.
        output_path: Optional output path for the audio file.
                    If not provided, a temporary file will be created.
        output_format: Output audio format (default: "wav").
        sample_rate: Output sample rate in Hz (default: 16000).
        channels: Number of audio channels (default: 1 for mono).
        
    Returns:
        str: Path to the extracted audio file.
        
    Raises:
        AudioProcessingError: If audio extraction fails.
    
    Example:
        >>> audio_path = extract_audio_from_video("video.mp4")
        >>> print(f"Audio extracted to: {audio_path}")
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise AudioProcessingError(
            f"Video file not found: {video_path}",
            file_path=str(video_path),
        )
    
    if not is_video_file(str(video_path)):
        raise UnsupportedFormatError(
            str(video_path),
            video_path.suffix,
        )
    
    try:
        from pydub import AudioSegment
        
        logger.info(f"Extracting audio from video: {video_path.name}")
        
        # Load video file (pydub/ffmpeg handles the extraction)
        audio = AudioSegment.from_file(str(video_path))
        
        # Convert to specified format
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(channels)
        
        # Determine output path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{output_format}",
                delete=False,
            )
            output_path = temp_file.name
            temp_file.close()
        
        # Export audio
        audio.export(output_path, format=output_format)
        
        logger.info(f"Audio extracted successfully: {output_path}")
        return output_path
        
    except ImportError:
        raise AudioProcessingError(
            "pydub library not installed. Run: pip install pydub",
            file_path=str(video_path),
        )
    except Exception as e:
        raise AudioProcessingError(
            f"Failed to extract audio from video: {e}",
            file_path=str(video_path),
        )


def convert_audio(
    input_path: str,
    output_path: Optional[str] = None,
    output_format: str = "wav",
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Convert audio file to a different format.
    
    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, a temporary
                    file will be created.
        output_format: Output audio format (default: "wav").
        sample_rate: Output sample rate in Hz (default: 16000).
        channels: Number of audio channels (default: 1 for mono).
        
    Returns:
        str: Path to the converted audio file.
        
    Raises:
        AudioProcessingError: If conversion fails.
    
    Example:
        >>> wav_path = convert_audio("audio.mp3", output_format="wav")
        >>> print(f"Converted to: {wav_path}")
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise AudioProcessingError(
            f"Audio file not found: {input_path}",
            file_path=str(input_path),
        )
    
    try:
        from pydub import AudioSegment
        
        logger.debug(f"Converting audio: {input_path.name} -> {output_format}")
        
        # Load audio file
        audio = AudioSegment.from_file(str(input_path))
        
        # Convert parameters
        audio = audio.set_frame_rate(sample_rate)
        audio = audio.set_channels(channels)
        
        # Determine output path
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=f".{output_format}",
                delete=False,
            )
            output_path = temp_file.name
            temp_file.close()
        
        # Export audio
        audio.export(output_path, format=output_format)
        
        logger.debug(f"Audio converted successfully: {output_path}")
        return output_path
        
    except ImportError:
        raise AudioProcessingError(
            "pydub library not installed. Run: pip install pydub",
            file_path=str(input_path),
        )
    except Exception as e:
        raise AudioProcessingError(
            f"Failed to convert audio: {e}",
            file_path=str(input_path),
        )


def prepare_audio_for_transcription(
    file_path: str,
    target_sample_rate: int = 16000,
) -> Tuple[str, bool]:
    """
    Prepare an audio or video file for transcription.
    
    This function handles:
    - Video to audio extraction
    - Audio format conversion if needed
    - Sample rate adjustment
    
    Args:
        file_path: Path to the audio or video file.
        target_sample_rate: Target sample rate for transcription (default: 16000).
        
    Returns:
        Tuple[str, bool]: Tuple of (prepared_file_path, is_temporary).
                         is_temporary indicates if the file should be deleted after use.
        
    Raises:
        AudioProcessingError: If preparation fails.
        UnsupportedFormatError: If the file format is not supported.
    
    Example:
        >>> audio_path, is_temp = prepare_audio_for_transcription("video.mp4")
        >>> # Use audio_path for transcription
        >>> if is_temp:
        ...     os.remove(audio_path)  # Clean up temporary file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise AudioProcessingError(
            f"File not found: {file_path}",
            file_path=str(file_path),
        )
    
    ext = file_path.suffix.lower()
    
    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(str(file_path), ext)
    
    # If it's a video file, extract audio
    if ext in SUPPORTED_VIDEO_FORMATS:
        extracted_path = extract_audio_from_video(
            str(file_path),
            sample_rate=target_sample_rate,
        )
        return extracted_path, True
    
    # For audio files, check if conversion is needed
    # WAV files at correct sample rate can be used directly
    if ext == ".wav":
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(file_path))
            if audio.frame_rate == target_sample_rate:
                return str(file_path), False
        except Exception:
            pass
    
    # Convert to WAV for optimal compatibility
    converted_path = convert_audio(
        str(file_path),
        output_format="wav",
        sample_rate=target_sample_rate,
    )
    return converted_path, True


def cleanup_temp_file(file_path: str) -> None:
    """
    Safely remove a temporary file.
    
    Args:
        file_path: Path to the file to remove.
    
    Example:
        >>> cleanup_temp_file("/tmp/audio_12345.wav")
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary file {file_path}: {e}")
