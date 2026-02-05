"""
Persian/Farsi text normalizer using Hazm library.

This module provides advanced Persian text normalization using the
Hazm NLP library. If Hazm is not available, use BasicNormalizer instead.
"""

import logging
from typing import Any, Optional

from .base import BaseNormalizer
from .basic import BasicNormalizer

logger = logging.getLogger(__name__)

# Try to import Hazm
HAZM_AVAILABLE = False
_HazmNormalizerClass: Optional[type] = None

try:
    from hazm import Normalizer as _HazmNormalizerImport

    _HazmNormalizerClass = _HazmNormalizerImport
    HAZM_AVAILABLE = True
except ImportError:
    logger.debug("Hazm library not available, will use BasicNormalizer as fallback")
except Exception as e:
    # Catch other errors (e.g., fasttext dependency issues on Python 3.13)
    logger.debug(f"Hazm import failed with error: {e}")


class PersianNormalizer(BaseNormalizer):
    """
    Advanced Persian text normalizer using Hazm library.

    This normalizer uses the Hazm NLP library for comprehensive
    Persian text normalization, including:

    - Arabic to Persian character conversion
    - Spacing corrections (half-space handling)
    - Punctuation normalization
    - Number normalization
    - Affix spacing

    If Hazm is not available, falls back to BasicNormalizer.

    Example:
        >>> normalizer = PersianNormalizer()
        >>> normalizer.normalize("سلام   دنيا!")
        'سلام دنیا!'
        >>> normalizer.is_hazm_available
        True  # or False if Hazm not installed

    Attributes:
        is_hazm_available: Whether Hazm library is being used.
    """

    def __init__(
        self,
        remove_extra_spaces: bool = True,
        persian_style: bool = True,
        persian_numbers: bool = True,
        unicodes_replacement: bool = True,
        seperate_mi: bool = True,
    ) -> None:
        """
        Initialize the Persian normalizer.

        Args:
            remove_extra_spaces: Remove extra whitespace. Defaults to True.
            persian_style: Apply Persian-style formatting. Defaults to True.
            persian_numbers: Convert to Persian numerals. Defaults to True.
            unicodes_replacement: Replace Unicode variants. Defaults to True.
            seperate_mi: Handle Persian "می" prefix spacing. Defaults to True.
        """
        self._hazm_normalizer: Optional[Any] = None
        self._fallback_normalizer: Optional[BasicNormalizer] = None
        self.is_hazm_available: bool = False

        if HAZM_AVAILABLE and _HazmNormalizerClass is not None:
            try:
                self._hazm_normalizer = _HazmNormalizerClass(
                    remove_extra_spaces=remove_extra_spaces,
                    persian_style=persian_style,
                    persian_numbers=persian_numbers,
                    unicodes_replacement=unicodes_replacement,
                    seperate_mi=seperate_mi,
                )
                self.is_hazm_available = True
                logger.info("Hazm normalizer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Hazm normalizer: {e}")
                self._fallback_normalizer = BasicNormalizer(convert_numerals=persian_numbers)
        else:
            logger.info("Hazm not available, using BasicNormalizer")
            self._fallback_normalizer = BasicNormalizer(convert_numerals=persian_numbers)

    @property
    def name(self) -> str:
        """Return the normalizer name."""
        if self.is_hazm_available:
            return "PersianNormalizer (Hazm)"
        return "PersianNormalizer (Basic fallback)"

    def normalize(self, text: str) -> str:
        """
        Normalize Persian text.

        Uses Hazm library if available, otherwise falls back to BasicNormalizer.

        Args:
            text: The text to normalize.

        Returns:
            str: The normalized text.
        """
        if not text:
            return ""

        if self._hazm_normalizer is not None:
            try:
                result: str = self._hazm_normalizer.normalize(text)
                return result
            except Exception as e:
                logger.warning(f"Hazm normalization failed: {e}, using fallback")
                if self._fallback_normalizer is None:
                    self._fallback_normalizer = BasicNormalizer()
                return self._fallback_normalizer.normalize(text)

        if self._fallback_normalizer is not None:
            return self._fallback_normalizer.normalize(text)

        # Last resort: return text as-is
        return text

    def affix_spacing(self, text: str) -> str:
        """
        Correct spacing around Persian affixes.

        Only available when Hazm is installed.

        Args:
            text: The text to process.

        Returns:
            str: Text with corrected affix spacing.
        """
        if not text:
            return ""

        if self._hazm_normalizer is not None and hasattr(self._hazm_normalizer, "affix_spacing"):
            try:
                result: str = self._hazm_normalizer.affix_spacing(text)
                return result
            except Exception:
                pass

        return text
