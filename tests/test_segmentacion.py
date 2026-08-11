"""Tests for sentence segmentation by language."""
import pytest
from segmentacion import get_sentence_segmenter, clean_text
from fixtures import LANGUAGE_FIXTURES, EMPTY_STRING, WHITESPACE_ONLY


class TestSegmentacion:
    """Test sentence segmentation across languages."""

    @pytest.mark.parametrize("idioma,fixtures", LANGUAGE_FIXTURES.items())
    def test_segmenter_single_sentence(self, idioma, fixtures):
        """Single sentence should return a list with one element."""
        segmenter = get_sentence_segmenter(idioma)
        text = fixtures["single"]
        sentences = segmenter(text)

        assert isinstance(sentences, list)
        assert len(sentences) >= 1
        # Single sentence should return exactly 1 sentence (or close to it)
        assert len(sentences) == 1, f"{idioma}: expected 1 sentence, got {len(sentences)}"

    @pytest.mark.parametrize("idioma,fixtures", LANGUAGE_FIXTURES.items())
    def test_segmenter_multiple_sentences(self, idioma, fixtures):
        """Multiple sentences should be segmented correctly."""
        if "multiple" not in fixtures:
            pytest.skip(f"No 'multiple' fixture for {idioma}")

        segmenter = get_sentence_segmenter(idioma)
        text = fixtures["multiple"]
        sentences = segmenter(text)

        assert isinstance(sentences, list)
        assert len(sentences) >= 2, f"{idioma}: expected ≥2 sentences, got {len(sentences)}"
        # All sentences should be non-empty strings
        for sent in sentences:
            assert isinstance(sent, str) and sent.strip(), f"{idioma}: empty sentence in result"

    @pytest.mark.parametrize("idioma,fixtures", LANGUAGE_FIXTURES.items())
    def test_segmenter_paragraph(self, idioma, fixtures):
        """Paragraph should be segmented into multiple sentences."""
        if "paragraph" not in fixtures:
            pytest.skip(f"No 'paragraph' fixture for {idioma}")

        segmenter = get_sentence_segmenter(idioma)
        text = fixtures["paragraph"]
        sentences = segmenter(text)

        assert isinstance(sentences, list)
        assert len(sentences) >= 3, f"{idioma}: expected ≥3 sentences in paragraph, got {len(sentences)}"

    def test_segmenter_empty_string(self):
        """Empty string should return empty list."""
        segmenter = get_sentence_segmenter("en")
        result = segmenter(EMPTY_STRING)
        assert result == []

    def test_segmenter_whitespace_only(self):
        """Whitespace-only string should return empty list."""
        segmenter = get_sentence_segmenter("en")
        result = segmenter(WHITESPACE_ONLY)
        assert result == []

    def test_segmenter_no_crash_on_unknown_language(self):
        """Unknown language should not crash, should use fallback."""
        segmenter = get_sentence_segmenter("unknown_lang_xyz")
        # Segmenter should return a callable, and it should not crash
        text = "This is a test. This is another sentence."
        result = segmenter(text)
        assert isinstance(result, list)
        # Fallback may return 1 or more sentences, depending on implementation
        assert len(result) >= 1


class TestCleanText:
    """Test text cleaning function."""

    def test_clean_normalizes_line_breaks(self):
        """Should normalize CRLF and CR to LF."""
        text = "Line1\r\nLine2\rLine3\nLine4"
        result = clean_text(text)
        assert "\r\n" not in result
        assert "\r" not in result
        assert "Line1\nLine2\nLine3\nLine4" == result

    def test_clean_collapses_whitespace(self):
        """Should collapse multiple spaces and tabs."""
        text = "Word1  \t  Word2   Word3"
        result = clean_text(text)
        assert result == "Word1 Word2 Word3"

    def test_clean_reduces_multiple_newlines(self):
        """Should reduce 3+ newlines to max 2."""
        text = "Para1\n\n\n\nPara2\n\n\n\n\nPara3"
        result = clean_text(text)
        assert "\n\n\n" not in result
        # Should have exactly double newlines as separators
        assert result == "Para1\n\nPara2\n\nPara3"

    def test_clean_strips_leading_trailing(self):
        """Should strip leading and trailing whitespace."""
        text = "  \n  Content  \n  "
        result = clean_text(text)
        assert result == "Content"

    def test_clean_empty_string(self):
        """Should handle empty string."""
        result = clean_text("")
        assert result == ""

    def test_clean_non_string_input(self):
        """Should return empty string for non-string input."""
        result = clean_text(None)
        assert result == ""

        result = clean_text(123)
        assert result == ""
