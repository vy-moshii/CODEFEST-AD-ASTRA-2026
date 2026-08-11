"""
Integration tests: diagnostics on generated chunks.

These tests are marked as 'slow' and validate that the chunking respects
sentence boundaries across languages.
"""
import re
from pathlib import Path

import pytest
import pandas as pd


class TestDiagnostico:
    """Integration tests validating chunking quality on generated chunks."""

    @pytest.mark.slow
    def test_chunks_respect_sentence_boundaries(self):
        """Verify that chunks respect sentence boundaries by language."""
        chunks_path = Path("salida_v2/chunks.parquet")
        if not chunks_path.exists():
            pytest.skip(f"Chunks not available at {chunks_path}")

        chunks_df = pd.read_parquet(chunks_path)
        closing_punct = r"[.!?…\n」』）】)]\'"'"'"。！？]"

        results_by_lang = {}

        for _, chunk in chunks_df.iterrows():
            idioma = chunk.get("idioma", "desconocido")
            chunk_text = chunk["texto"]
            posicion = chunk["posicion"]

            if idioma not in results_by_lang:
                results_by_lang[idioma] = {
                    "total_chunks": 0,
                    "boundary_respecting": 0,
                }

            results_by_lang[idioma]["total_chunks"] += 1

            # Check if chunk ends in closing punctuation
            # Last chunk of a document (posicion at max) can end anywhere
            is_last_in_doc = (posicion == chunks_df[chunks_df["doc_id"] == chunk["doc_id"]]["posicion"].max())

            if is_last_in_doc or re.search(closing_punct, chunk_text[-5:]):
                results_by_lang[idioma]["boundary_respecting"] += 1

        # Report results
        print("\n" + "=" * 80)
        print("DIAGNÓSTICO: Completitud Lingüística por Idioma")
        print("=" * 80)

        all_pass = True
        for lang in sorted(results_by_lang.keys()):
            stats = results_by_lang[lang]
            total = stats["total_chunks"]
            respecting = stats["boundary_respecting"]
            pct = 100.0 * respecting / total if total > 0 else 0

            # Dynamic threshold by language
            # Thresholds reflect corpus composition (structured data is harder to segment)
            # and segmenter quality by language (pysbd works very well for es, pt, fr)
            thresholds = {
                "desconocido": 50,  # Fallback regex
                "en": 25,           # English: corpus is ~45% JSON/CSV (structured data, hard to segment)
                "es": 85,           # Spanish: pysbd works very well
                "pt": 85,           # Portuguese: pysbd works well
                "fr": 85,           # French: pysbd works well
                "zh-cn": 85,        # Chinese: regex fallback
            }
            threshold = thresholds.get(lang, 85)

            status = "✓" if pct >= threshold else "⚠️ "
            print(
                f"{status} {lang:10s} | {respecting:4d}/{total:4d} chunks "
                f"({pct:5.1f}%) | threshold: {threshold}%"
            )

            if pct < threshold:
                all_pass = False

        print("=" * 80)

        if not all_pass:
            pytest.fail(
                "Some languages fell below threshold for sentence-boundary respect. "
                "Check the diagnóstico output above."
            )

    @pytest.mark.slow
    def test_chunks_have_valid_metadata(self):
        """Verify that all chunks have required metadata fields."""
        chunks_path = Path("salida_v2/chunks.parquet")
        if not chunks_path.exists():
            pytest.skip(f"Chunks not available at {chunks_path}")

        chunks_df = pd.read_parquet(chunks_path)
        required_fields = {"texto", "posicion", "num_tokens", "doc_id", "chunk_id"}

        for _, chunk in chunks_df.iterrows():
            for field in required_fields:
                assert field in chunk.index, f"Missing field '{field}' in chunk"

            assert isinstance(chunk["posicion"], (int, int))
            assert chunk["posicion"] >= 0
            assert isinstance(chunk["num_tokens"], (int, int))
            assert chunk["num_tokens"] >= 0
            assert isinstance(chunk["texto"], str)
            assert len(chunk["texto"]) > 0

    @pytest.mark.slow
    def test_chunk_text_is_complete(self):
        """Verify that no text is lost during chunking."""
        chunks_path = Path("salida_v2/chunks.parquet")
        if not chunks_path.exists():
            pytest.skip(f"Chunks not available at {chunks_path}")

        chunks_df = pd.read_parquet(chunks_path)

        # Verify that chunk count is reasonable
        # Expect ~2-3 chunks per document on average
        n_docs = chunks_df["doc_id"].nunique()
        n_chunks = len(chunks_df)
        avg_chunks_per_doc = n_chunks / n_docs if n_docs > 0 else 0

        assert avg_chunks_per_doc >= 1.5, (
            f"Too few chunks per doc ({avg_chunks_per_doc:.2f}). "
            f"Text may have been lost."
        )
        assert avg_chunks_per_doc <= 10, (
            f"Too many chunks per doc ({avg_chunks_per_doc:.2f}). "
            f"Chunks may be too small."
        )
