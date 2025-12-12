"""
Base normalizer interface.

This module defines the abstract base class for all text normalizers.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseNormalizer(ABC):
    """
    Abstract base class for text normalizers.
    
    All normalizer implementations must inherit from this class
    and implement the normalize() method.
    
    Example:
        >>> class MyNormalizer(BaseNormalizer):
        ...     def normalize(self, text: str) -> str:
        ...         return text.strip().lower()
        ...
        >>> normalizer = MyNormalizer()
        >>> normalizer.normalize("  HELLO  ")
        'hello'
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of this normalizer.
        
        Returns:
            str: Human-readable name of the normalizer.
        """
        pass
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize the given text.
        
        Args:
            text: The text to normalize.
            
        Returns:
            str: The normalized text.
        """
        pass
    
    def __call__(self, text: str) -> str:
        """
        Allow using normalizer as a callable.
        
        Args:
            text: The text to normalize.
            
        Returns:
            str: The normalized text.
        """
        return self.normalize(text)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
