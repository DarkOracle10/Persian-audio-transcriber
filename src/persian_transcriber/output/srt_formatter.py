"""
SRT subtitle output formatter.

This module provides an SRT (SubRip) subtitle formatter for
creating subtitle files from transcriptions.
"""

from ..engines.base import TranscriptionResult
from .base import BaseFormatter


class SrtFormatter(BaseFormatter):
    """
    SRT subtitle output formatter.
    
    Outputs the transcription in SRT (SubRip) subtitle format,
    which is widely supported by video players and editors.
    
    SRT Format:
        1
        00:00:00,000 --> 00:00:05,000
        First subtitle text
        
        2
        00:00:05,000 --> 00:00:10,000
        Second subtitle text
    
    Example:
        >>> formatter = SrtFormatter()
        >>> srt_content = formatter.format(result)
        >>> print(srt_content)
        1
        00:00:00,000 --> 00:00:05,230
        سلام
        
        2
        00:00:05,230 --> 00:00:10,450
        این یک متن آزمایشی است.
    """
    
    def __init__(
        self,
        max_line_length: int = 42,
        max_lines_per_subtitle: int = 2,
    ) -> None:
        """
        Initialize the SRT formatter.
        
        Args:
            max_line_length: Maximum characters per line. Set to 0 to disable
                            line breaking.
            max_lines_per_subtitle: Maximum lines per subtitle block.
        """
        self.max_line_length = max_line_length
        self.max_lines_per_subtitle = max_lines_per_subtitle
    
    @property
    def name(self) -> str:
        """Get the formatter name."""
        return "SRT Subtitle"
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        return "srt"
    
    def format(self, result: TranscriptionResult) -> str:
        """
        Format the transcription result as SRT subtitles.
        
        Args:
            result: The transcription result to format.
            
        Returns:
            str: The SRT-formatted string.
        """
        if not result.segments:
            # If no segments, create a single subtitle from the full text
            if result.text:
                return self._create_subtitle_block(
                    index=1,
                    start=0.0,
                    end=result.duration or 5.0,
                    text=result.text,
                )
            return ""
        
        blocks: list = []
        
        for index, segment in enumerate(result.segments, start=1):
            block = self._create_subtitle_block(
                index=index,
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
            blocks.append(block)
        
        return "\n\n".join(blocks)
    
    def _create_subtitle_block(
        self,
        index: int,
        start: float,
        end: float,
        text: str,
    ) -> str:
        """
        Create a single SRT subtitle block.
        
        Args:
            index: Subtitle index (1-based).
            start: Start time in seconds.
            end: End time in seconds.
            text: Subtitle text.
            
        Returns:
            str: Formatted subtitle block.
        """
        start_time = self._format_timestamp(start)
        end_time = self._format_timestamp(end)
        
        # Optionally wrap text
        if self.max_line_length > 0:
            text = self._wrap_text(text)
        
        return f"{index}\n{start_time} --> {end_time}\n{text}"
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as SRT timestamp (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            str: SRT-formatted timestamp.
        """
        if seconds < 0:
            seconds = 0
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _wrap_text(self, text: str) -> str:
        """
        Wrap text to fit within line length limits.
        
        Args:
            text: Text to wrap.
            
        Returns:
            str: Wrapped text.
        """
        if len(text) <= self.max_line_length:
            return text
        
        words = text.split()
        lines: list = []
        current_line: list = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            
            # Check if adding this word would exceed the limit
            if current_length + word_length + (1 if current_line else 0) > self.max_line_length:
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    # Word itself is too long, add it anyway
                    lines.append(word)
                    current_length = 0
            else:
                current_line.append(word)
                current_length += word_length + (1 if len(current_line) > 1 else 0)
        
        # Add remaining words
        if current_line:
            lines.append(" ".join(current_line))
        
        # Limit number of lines
        if len(lines) > self.max_lines_per_subtitle:
            lines = lines[:self.max_lines_per_subtitle]
        
        return "\n".join(lines)


class VttFormatter(BaseFormatter):
    """
    WebVTT subtitle output formatter.
    
    Outputs the transcription in WebVTT format, commonly used for
    web-based video players and HTML5 video.
    
    VTT Format:
        WEBVTT
        
        00:00:00.000 --> 00:00:05.000
        First subtitle text
        
        00:00:05.000 --> 00:00:10.000
        Second subtitle text
    """
    
    def __init__(
        self,
        max_line_length: int = 42,
        max_lines_per_subtitle: int = 2,
    ) -> None:
        """
        Initialize the VTT formatter.
        
        Args:
            max_line_length: Maximum characters per line.
            max_lines_per_subtitle: Maximum lines per subtitle block.
        """
        self.max_line_length = max_line_length
        self.max_lines_per_subtitle = max_lines_per_subtitle
        self._srt_formatter = SrtFormatter(max_line_length, max_lines_per_subtitle)
    
    @property
    def name(self) -> str:
        """Get the formatter name."""
        return "WebVTT Subtitle"
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        return "vtt"
    
    def format(self, result: TranscriptionResult) -> str:
        """
        Format the transcription result as WebVTT subtitles.
        
        Args:
            result: The transcription result to format.
            
        Returns:
            str: The VTT-formatted string.
        """
        blocks: list = ["WEBVTT", ""]
        
        if not result.segments:
            if result.text:
                start_time = self._format_timestamp(0.0)
                end_time = self._format_timestamp(result.duration or 5.0)
                blocks.append(f"{start_time} --> {end_time}")
                blocks.append(result.text)
            return "\n".join(blocks)
        
        for segment in result.segments:
            start_time = self._format_timestamp(segment.start)
            end_time = self._format_timestamp(segment.end)
            
            text = segment.text
            if self.max_line_length > 0:
                text = self._srt_formatter._wrap_text(text)
            
            blocks.append(f"{start_time} --> {end_time}")
            blocks.append(text)
            blocks.append("")
        
        return "\n".join(blocks)
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds as VTT timestamp (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            str: VTT-formatted timestamp.
        """
        if seconds < 0:
            seconds = 0
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
