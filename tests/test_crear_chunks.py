"""Tests for sentence-aware chunk creation."""
import pytest
from chunking_core import crear_chunks, count_tokens
from fixtures import (
    LANGUAGE_FIXTURES,
    EMPTY_STRING,
    WHITESPACE_ONLY,
    EN_SINGLE_SENTENCE,
    EN_MULTIPLE_SENTENCES,
    VERY_LONG_SENTENCE,
)


class TestCrearChunks:
    """Test sentence-aware chunk creation."""

    def test_single_sentence_short(self):
        """Short text <= CHUNK_SIZE should return a single chunk."""
        chunks = crear_chunks(EN_SINGLE_SENTENCE, "en")

        assert len(chunks) == 1
        assert chunks[0]["texto"] == EN_SINGLE_SENTENCE
        assert chunks[0]["posicion"] == 0
        assert chunks[0]["num_tokens"] > 0

    def test_multiple_sentences_fits_one_chunk(self):
        """Multiple sentences that fit in one chunk should return one chunk."""
        chunks = crear_chunks(EN_MULTIPLE_SENTENCES, "en")

        assert len(chunks) == 1
        assert "Machine learning" in chunks[0]["texto"]
        assert chunks[0]["posicion"] == 0

    def test_posicion_field_increments(self):
        """Posicion should increment from 0 for each chunk."""
        # Use a long paragraph to force multiple chunks
        long_text = EN_MULTIPLE_SENTENCES * 5  # Repeat to exceed CHUNK_SIZE
        chunks = crear_chunks(long_text, "en")

        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                assert chunk["posicion"] == i

    def test_num_tokens_present(self):
        """Every chunk should have num_tokens > 0."""
        chunks = crear_chunks(EN_SINGLE_SENTENCE, "en")

        for chunk in chunks:
            assert "num_tokens" in chunk
            assert isinstance(chunk["num_tokens"], int)
            assert chunk["num_tokens"] > 0

    def test_empty_string_returns_empty_list(self):
        """Empty string should return empty list."""
        chunks = crear_chunks(EMPTY_STRING, "en")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only string should return empty list."""
        chunks = crear_chunks(WHITESPACE_ONLY, "en")
        assert chunks == []

    def test_text_content_preserved(self):
        """Original text content should be preserved in chunks."""
        text = EN_MULTIPLE_SENTENCES
        chunks = crear_chunks(text, "en")

        # Reconstruct by joining all chunk texts
        reconstructed = " ".join(c["texto"] for c in chunks)
        # Should contain the key phrases from original
        assert "Machine learning" in reconstructed
        assert "neural network" in reconstructed.lower()

    @pytest.mark.parametrize("idioma,fixtures", LANGUAGE_FIXTURES.items())
    def test_multiple_languages(self, idioma, fixtures):
        """Chunking should work across multiple languages."""
        text = fixtures["single"]
        chunks = crear_chunks(text, idioma)

        # Even a single sentence should return at least one chunk
        assert len(chunks) >= 1
        assert chunks[0]["num_tokens"] > 0

    def test_very_long_sentence_preserved(self):
        """Very long single sentence should be preserved as one chunk (no mid-sentence break)."""
        chunks = crear_chunks(VERY_LONG_SENTENCE, "en")

        # Should return exactly 1 chunk with the full sentence
        assert len(chunks) == 1
        # The chunk should contain the start and end of the sentence
        assert "extremely long" in chunks[0]["texto"]
        assert "run together" in chunks[0]["texto"]

    def test_chunks_have_required_fields(self):
        """Each chunk should have texto, posicion, and num_tokens."""
        chunks = crear_chunks(EN_SINGLE_SENTENCE, "en")

        for chunk in chunks:
            assert "texto" in chunk
            assert "posicion" in chunk
            assert "num_tokens" in chunk
            assert isinstance(chunk["texto"], str)
            assert isinstance(chunk["posicion"], int)
            assert isinstance(chunk["num_tokens"], int)


class TestCountTokens:
    """Test token counting."""

    def test_count_tokens_positive(self):
        """Token count should be positive for non-empty text."""
        count = count_tokens("Hello world")
        assert count > 0

    def test_count_tokens_empty(self):
        """Token count for empty string should be 0."""
        count = count_tokens("")
        assert count == 0

    def test_count_tokens_increases_with_length(self):
        """Longer text should have more tokens."""
        short = "Hello"
        long = "Hello " * 100
        assert count_tokens(long) > count_tokens(short)

    def test_count_tokens_consistent(self):
        """Same text should always return same token count."""
        text = "This is a test sentence for token counting."
        count1 = count_tokens(text)
        count2 = count_tokens(text)
        assert count1 == count2
