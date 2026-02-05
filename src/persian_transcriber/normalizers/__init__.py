"""
Text normalizers for Persian/Farsi language.

This module provides text normalization utilities for Persian text,
including character conversion, whitespace handling, and NLP-based
normalization using the Hazm library.

Available normalizers:
- PersianNormalizer: Full-featured normalizer using Hazm (with fallback)
- BasicNormalizer: Simple character mapping and whitespace normalization
- BaseNormalizer: Abstract base class for custom normalizers

Example:
    >>> from persian_transcriber.normalizers import get_normalizer
    >>> normalizer = get_normalizer("persian")
    >>> normalizer.normalize("سلام   دنيا")
    'سلام دنیا'

    >>> from persian_transcriber.normalizers import PersianNormalizer
    >>> normalizer = PersianNormalizer()
    >>> normalizer("متن فارسی")  # Can also call directly
    'متن فارسی'
"""

from enum import Enum
from typing import Union

from .base import BaseNormalizer
from .basic import BasicNormalizer
from .persian import PersianNormalizer, HAZM_AVAILABLE

__all__ = [
    "BaseNormalizer",
    "BasicNormalizer",
    "PersianNormalizer",
    "NormalizerType",
    "get_normalizer",
    "HAZM_AVAILABLE",
]


class NormalizerType(str, Enum):
    """Enumeration of available normalizer types."""

    PERSIAN = "persian"
    BASIC = "basic"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


def get_normalizer(
    normalizer_type: Union[str, NormalizerType] = NormalizerType.PERSIAN,
    **kwargs,
) -> BaseNormalizer:
    """
    Factory function to create a normalizer instance.

    Args:
        normalizer_type: Type of normalizer to create.
            - "persian": PersianNormalizer (Hazm with fallback)
            - "basic": BasicNormalizer (simple character mapping)
            - "none": NullNormalizer (returns text unchanged)
        **kwargs: Additional arguments passed to normalizer constructor.

    Returns:
        BaseNormalizer: A normalizer instance.

    Raises:
        ValueError: If normalizer_type is not recognized.

    Example:
        >>> normalizer = get_normalizer("persian")
        >>> normalizer.normalize("تست")
        'تست'
    """
    if isinstance(normalizer_type, str):
        normalizer_type = normalizer_type.lower()

    if normalizer_type in (NormalizerType.PERSIAN, "persian"):
        return PersianNormalizer(**kwargs)

    if normalizer_type in (NormalizerType.BASIC, "basic"):
        return BasicNormalizer(**kwargs)

    if normalizer_type in (NormalizerType.NONE, "none"):
        return _NullNormalizer()

    raise ValueError(
        f"Unknown normalizer type: {normalizer_type}. "
        f"Available types: {', '.join(t.value for t in NormalizerType)}"
    )


class _NullNormalizer(BaseNormalizer):
    """Normalizer that returns text unchanged."""

    @property
    def name(self) -> str:
        return "NullNormalizer"

    def normalize(self, text: str) -> str:
        return text if text else ""
