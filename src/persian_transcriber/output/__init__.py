"""
Output formatters for Persian Transcriber.

This package provides formatters for converting transcription results
to various output formats:

- TxtFormatter: Plain text output
- JsonFormatter: JSON with full metadata
- SrtFormatter: SRT subtitle format
- VttFormatter: WebVTT subtitle format

Example:
    >>> from persian_transcriber.output import get_formatter, OutputFormat
    >>> formatter = get_formatter(OutputFormat.SRT)
    >>> srt_content = formatter.format(result)
    >>> formatter.save(result, "output.srt")
"""

from enum import Enum
from typing import Union

from .base import BaseFormatter
from .json_formatter import JsonFormatter
from .srt_formatter import SrtFormatter, VttFormatter
from .txt_formatter import TxtFormatter

__all__ = [
    # Base class
    "BaseFormatter",
    # Formatter implementations
    "TxtFormatter",
    "JsonFormatter",
    "SrtFormatter",
    "VttFormatter",
    # Enum and factory
    "OutputFormat",
    "get_formatter",
]


class OutputFormat(str, Enum):
    """Enumeration of available output formats."""

    TXT = "txt"
    JSON = "json"
    SRT = "srt"
    VTT = "vtt"

    def __str__(self) -> str:
        return self.value


def get_formatter(
    output_format: Union[str, OutputFormat] = OutputFormat.TXT,
    **kwargs,
) -> BaseFormatter:
    """
    Factory function to create an output formatter.

    Args:
        output_format: Format type to create. Options:
            - "txt" or OutputFormat.TXT: Plain text
            - "json" or OutputFormat.JSON: JSON with metadata
            - "srt" or OutputFormat.SRT: SRT subtitles
            - "vtt" or OutputFormat.VTT: WebVTT subtitles
        **kwargs: Additional arguments passed to formatter constructor.

    Returns:
        BaseFormatter: A configured formatter instance.

    Raises:
        ValueError: If the format type is not recognized.

    Examples:
        >>> formatter = get_formatter("txt")
        >>> formatter = get_formatter(OutputFormat.JSON, indent=4)
        >>> formatter = get_formatter("srt", max_line_length=50)
    """
    # Normalize format type
    if isinstance(output_format, str):
        output_format = output_format.lower()

    if output_format in (OutputFormat.TXT, "txt", "text"):
        return TxtFormatter(**kwargs)

    if output_format in (OutputFormat.JSON, "json"):
        return JsonFormatter(**kwargs)

    if output_format in (OutputFormat.SRT, "srt"):
        return SrtFormatter(**kwargs)

    if output_format in (OutputFormat.VTT, "vtt", "webvtt"):
        return VttFormatter(**kwargs)

    raise ValueError(
        f"Unknown output format: {output_format}. "
        f"Available formats: {', '.join(f.value for f in OutputFormat)}"
    )
