"""Unit tests for the basic Persian normalizer."""

from persian_transcriber.normalizers.basic import BasicNormalizer


def test_normalize_empty_string() -> None:
    normalizer = BasicNormalizer()
    assert normalizer.normalize("") == ""


def test_normalize_persian_characters(sample_persian_text) -> None:
    normalizer = BasicNormalizer()
    normalized = normalizer.normalize(sample_persian_text["arabic_chars"])
    assert normalized == "سلام کیف حالک؟"


def test_normalize_whitespace(sample_persian_text) -> None:
    normalizer = BasicNormalizer()
    normalized = normalizer.normalize(sample_persian_text["whitespace"])
    assert normalized == "سلام دنیا"


def test_normalize_punctuation(sample_persian_text) -> None:
    normalizer = BasicNormalizer()
    normalized = normalizer.normalize(sample_persian_text["punctuation"])
    assert normalized == "سلام، دنیا!؟"


def test_normalize_numbers(sample_persian_text) -> None:
    normalizer = BasicNormalizer(convert_numerals=True)
    normalized = normalizer.normalize(sample_persian_text["numbers"])
    assert normalized == "سال ۱۴۰۲"


def test_normalize_mixed_text(sample_persian_text) -> None:
    normalizer = BasicNormalizer()
    normalized = normalizer.normalize(sample_persian_text["mixed"])
    assert normalized == "سلام Tehran 2025"


def test_normalize_special_chars(sample_persian_text) -> None:
    normalizer = BasicNormalizer()
    normalized = normalizer.normalize(sample_persian_text["special"])
    assert normalized == "سلام @#$% دنیا"


def test_normalize_unicode(sample_persian_text) -> None:
    normalizer = BasicNormalizer()
    normalized = normalizer.normalize(sample_persian_text["unicode"])
    assert normalized == "\u200cسلام دنیا"
