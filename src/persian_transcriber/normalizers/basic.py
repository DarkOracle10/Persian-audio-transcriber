"""
Basic Persian text normalizer.

This module provides a fallback normalizer when the Hazm library
is not available. It handles basic Arabic-to-Persian character
conversion and whitespace normalization.
"""

import re
from typing import Dict

from .base import BaseNormalizer


class BasicNormalizer(BaseNormalizer):
    """
    Basic Persian text normalizer (fallback when Hazm is not available).

    Handles:
    - Arabic to Persian character conversion (ك → ک, ي → ی, etc.)
    - Whitespace normalization (multiple spaces → single space)
    - Leading/trailing whitespace removal

    Example:
        >>> normalizer = BasicNormalizer()
        >>> normalizer.normalize("سلام   دنيا")
        'سلام دنیا'
    """

    # Arabic to Persian character mappings
    ARABIC_TO_PERSIAN: Dict[str, str] = {
        "ك": "ک",  # Arabic kaf to Persian kaf
        "ي": "ی",  # Arabic yeh to Persian yeh
        "ى": "ی",  # Arabic alef maksura to Persian yeh
        "ؤ": "و",  # Arabic waw with hamza
        "أ": "ا",  # Arabic alef with hamza above
        "ئ": "ی",  # Arabic yeh with hamza
        "ة": "ه",  # Arabic teh marbuta
        "إ": "ا",  # Arabic alef with hamza below
        "ء": "",  # Arabic hamza (often removed)
        "آ": "آ",  # Persian alef with maddah (keep as is)
    }

    # Arabic numerals to Persian/Farsi numerals
    ARABIC_TO_PERSIAN_NUMERALS: Dict[str, str] = {
        "٠": "۰",  # 0
        "١": "۱",  # 1
        "٢": "۲",  # 2
        "٣": "۳",  # 3
        "٤": "۴",  # 4
        "٥": "۵",  # 5
        "٦": "۶",  # 6
        "٧": "۷",  # 7
        "٨": "۸",  # 8
        "٩": "۹",  # 9
    }

    def __init__(self, convert_numerals: bool = True) -> None:
        """
        Initialize the basic normalizer.

        Args:
            convert_numerals: Whether to convert Arabic numerals to Persian.
                             Defaults to True.
        """
        self._convert_numerals = convert_numerals

        # Build combined translation table
        self._char_map: Dict[str, str] = dict(self.ARABIC_TO_PERSIAN)
        if convert_numerals:
            self._char_map.update(self.ARABIC_TO_PERSIAN_NUMERALS)

    @property
    def name(self) -> str:
        """Return the normalizer name."""
        return "BasicNormalizer"

    def normalize(self, text: str) -> str:
        """
        Normalize Persian text with basic rules.

        Steps:
        1. Convert Arabic characters to Persian equivalents
        2. Optionally convert Arabic numerals to Persian
        3. Normalize whitespace (multiple spaces/tabs/newlines to single space)
        4. Trim leading/trailing whitespace

        Args:
            text: The text to normalize.

        Returns:
            str: The normalized text, or empty string if input is None/empty.
        """
        if not text:
            return ""

        # Convert Arabic characters to Persian
        normalized = text
        for arabic_char, persian_char in self._char_map.items():
            normalized = normalized.replace(arabic_char, persian_char)

        # Normalize whitespace (multiple spaces/tabs/newlines to single space)
        normalized = re.sub(r"\s+", " ", normalized)

        # Trim leading/trailing whitespace
        normalized = normalized.strip()

        return normalized

    def normalize_characters_only(self, text: str) -> str:
        """
        Normalize only character mappings without whitespace changes.

        Args:
            text: The text to normalize.

        Returns:
            str: Text with Arabic characters converted to Persian.
        """
        if not text:
            return ""

        normalized = text
        for arabic_char, persian_char in self._char_map.items():
            normalized = normalized.replace(arabic_char, persian_char)

        return normalized
