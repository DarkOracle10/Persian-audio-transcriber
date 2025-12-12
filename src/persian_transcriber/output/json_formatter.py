"""
JSON output formatter.

This module provides a JSON formatter that outputs the full
transcription result with metadata and segments.
"""

import json
from typing import Any, Dict

from ..engines.base import TranscriptionResult
from .base import BaseFormatter


class JsonFormatter(BaseFormatter):
    """
    JSON output formatter.
    
    Outputs the transcription as a JSON object containing:
    - The full transcription text
    - Individual segments with timestamps
    - Metadata (language, duration, engine, etc.)
    
    Example:
        >>> formatter = JsonFormatter()
        >>> json_str = formatter.format(result)
        >>> print(json_str)
        {
          "text": "سلام این یک متن آزمایشی است.",
          "language": "fa",
          "duration": 10.5,
          ...
        }
    """
    
    def __init__(
        self,
        indent: int = 2,
        ensure_ascii: bool = False,
        include_raw_text: bool = True,
        include_words: bool = True,
    ) -> None:
        """
        Initialize the JSON formatter.
        
        Args:
            indent: Indentation level for pretty printing.
                   Set to None for compact output.
            ensure_ascii: If True, escape non-ASCII characters.
                         Set to False for Persian text readability.
            include_raw_text: Include the raw (unnormalized) text.
            include_words: Include word-level timestamps if available.
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii
        self.include_raw_text = include_raw_text
        self.include_words = include_words
    
    @property
    def name(self) -> str:
        """Get the formatter name."""
        return "JSON"
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        return "json"
    
    def format(self, result: TranscriptionResult) -> str:
        """
        Format the transcription result as JSON.
        
        Args:
            result: The transcription result to format.
            
        Returns:
            str: The JSON-formatted string.
        """
        output: Dict[str, Any] = {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
        }
        
        # Include raw text if requested and different from normalized
        if self.include_raw_text and result.text_raw != result.text:
            output["text_raw"] = result.text_raw
        
        # Add language probability if available
        if result.language_probability is not None:
            output["language_probability"] = result.language_probability
        
        # Add segments
        output["segments"] = []
        for segment in result.segments:
            seg_dict: Dict[str, Any] = {
                "text": segment.text,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
            }
            
            if segment.confidence is not None:
                seg_dict["confidence"] = round(segment.confidence, 4)
            
            if self.include_words and segment.words:
                seg_dict["words"] = segment.words
            
            output["segments"].append(seg_dict)
        
        # Add metadata
        output["metadata"] = {
            "engine": result.engine,
            "model": result.model,
            "segment_count": len(result.segments),
            "word_count": result.word_count,
        }
        
        # Add any additional metadata from the result
        if result.metadata:
            output["metadata"].update(result.metadata)
        
        return json.dumps(
            output,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
        )
    
    def format_minimal(self, result: TranscriptionResult) -> str:
        """
        Format with minimal output (just text and segments).
        
        Args:
            result: The transcription result to format.
            
        Returns:
            str: Minimal JSON string.
        """
        output = {
            "text": result.text,
            "segments": [
                {
                    "text": seg.text,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                }
                for seg in result.segments
            ],
        }
        
        return json.dumps(
            output,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
        )
