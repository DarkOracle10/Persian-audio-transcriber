"""
Base output formatter interface.

This module defines the abstract base class for all output formatters.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from ..engines.base import TranscriptionResult


class BaseFormatter(ABC):
    """
    Abstract base class for output formatters.
    
    All formatter implementations must inherit from this class
    and implement the required abstract methods.
    
    Example:
        >>> class MyFormatter(BaseFormatter):
        ...     @property
        ...     def name(self) -> str:
        ...         return "MyFormat"
        ...     
        ...     @property
        ...     def extension(self) -> str:
        ...         return "myf"
        ...     
        ...     def format(self, result: TranscriptionResult) -> str:
        ...         return result.text
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the human-readable name of this formatter.
        
        Returns:
            str: Formatter name (e.g., "Plain Text", "JSON").
        """
        pass
    
    @property
    @abstractmethod
    def extension(self) -> str:
        """
        Get the file extension for this format.
        
        Returns:
            str: File extension without dot (e.g., "txt", "json").
        """
        pass
    
    @abstractmethod
    def format(self, result: TranscriptionResult) -> str:
        """
        Format a transcription result.
        
        Args:
            result: The transcription result to format.
            
        Returns:
            str: The formatted output string.
        """
        pass
    
    def save(
        self,
        result: TranscriptionResult,
        output_path: Union[str, Path],
        encoding: str = "utf-8",
    ) -> Path:
        """
        Format and save a transcription result to a file.
        
        Args:
            result: The transcription result to format and save.
            output_path: Path to save the output file. If no extension is
                        provided, the formatter's default extension is used.
            encoding: File encoding (default: "utf-8").
            
        Returns:
            Path: The path to the saved file.
        """
        output_path = Path(output_path)
        
        # Add extension if not present
        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{self.extension}")
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format and write
        content = self.format(result)
        output_path.write_text(content, encoding=encoding)
        
        return output_path
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
