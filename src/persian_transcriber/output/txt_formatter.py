"""
Plain text output formatter.

This module provides a simple text formatter that outputs
just the transcription text.
"""

from ..engines.base import TranscriptionResult
from .base import BaseFormatter


class TxtFormatter(BaseFormatter):
    """
    Plain text output formatter.

    Outputs the transcription as plain text. Can optionally include
    timestamps for each segment.

    Example:
        >>> formatter = TxtFormatter()
        >>> text = formatter.format(result)
        >>> print(text)
        سلام این یک متن آزمایشی است.

        >>> formatter = TxtFormatter(include_timestamps=True)
        >>> text = formatter.format(result)
        >>> print(text)
        [00:00:00 - 00:00:05] سلام
        [00:00:05 - 00:00:10] این یک متن آزمایشی است.
    """

    def __init__(
        self,
        include_timestamps: bool = False,
        include_metadata: bool = False,
        segment_separator: str = "\n",
    ) -> None:
        """
        Initialize the text formatter.

        Args:
            include_timestamps: If True, include timestamps for each segment.
            include_metadata: If True, include metadata header.
            segment_separator: Separator between segments (default: newline).
        """
        self.include_timestamps = include_timestamps
        self.include_metadata = include_metadata
        self.segment_separator = segment_separator

    @property
    def name(self) -> str:
        """Get the formatter name."""
        return "Plain Text"

    @property
    def extension(self) -> str:
        """Get the file extension."""
        return "txt"

    def format(self, result: TranscriptionResult) -> str:
        """
        Format the transcription result as plain text.

        Args:
            result: The transcription result to format.

        Returns:
            str: The formatted text.
        """
        lines: list = []

        # Add metadata header if requested
        if self.include_metadata:
            lines.append("=" * 60)
            lines.append("TRANSCRIPTION RESULT")
            lines.append("=" * 60)
            lines.append(f"Language: {result.language}")
            lines.append(f"Duration: {self._format_duration(result.duration)}")
            lines.append(f"Engine: {result.engine}")
            if result.model:
                lines.append(f"Model: {result.model}")
            lines.append("=" * 60)
            lines.append("")

        # Add transcription content
        if self.include_timestamps and result.segments:
            # Format with timestamps
            for segment in result.segments:
                start_time = self._format_timestamp(segment.start)
                end_time = self._format_timestamp(segment.end)
                lines.append(f"[{start_time} - {end_time}] {segment.text}")
        else:
            # Just the text
            lines.append(result.text)

        return self.segment_separator.join(lines)

    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as HH:MM:SS.

        Args:
            seconds: Time in seconds.

        Returns:
            str: Formatted timestamp.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration with more detail.

        Args:
            seconds: Duration in seconds.

        Returns:
            str: Formatted duration string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {secs:.1f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.1f}s"
        else:
            return f"{secs:.1f}s"
