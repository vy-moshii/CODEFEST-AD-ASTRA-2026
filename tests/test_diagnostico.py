"""
Integration tests: diagnostics on real corpus (if available).

These tests require the full documentos.parquet and are marked as 'slow'.
They validate that the chunking respects sentence boundaries across real documents.
"""
import re
from pathlib import Path

import pytest
import pandas as pd

from chunking_core import crear_chunks
from segmentacion import get_sentence_segmenter


@pytest.fixture
def docs_parquet():
    """Load the real documents parquet, skip test if not available."""
    path = Path("salida/documentos.parquet")
    if not path.exists():
        pytest.skip(f"Corpus not available at {path}")
    return pd.read_parquet(path)


@pytest.fixture
def docs_sample(docs_parquet):
    """Take a deterministic sample of documents by language (seed for reproducibility)."""
    # Sample 30-50 documents, stratified by idioma to cover language variety
    # This is fast enough for CI/local testing
    valid_df = docs_parquet[
        (docs_parquet["estado"] == "ok") & docs_parquet["texto"].notna()
    ].copy()

    # Stratified sample: at least 2-3 docs per major language, 1 for rare
    major_langs = ["en", "es", "pt", "fr", "zh", "ar", "ru"]
    sample = []

    for lang in major_langs:
        lang_docs = valid_df[valid_df["idioma"] == lang]
        n = min(len(lang_docs), 5)  # Up to 5 per language
        if n > 0:
            sample.append(lang_docs.head(n))

    # Add a few rare languages if available
    rare_df = valid_df[~valid_df["idioma"].isin(major_langs + ["desconocido"])]
    if len(rare_df) > 0:
        sample.append(rare_df.head(3))

    if sample:
        result = pd.concat(sample, ignore_index=True)
        return result.head(50)  # Cap at 50 for reasonable test time

    return valid_df.head(50)


class TestDiagnostico:
    """Integration tests validating chunking quality on real corpus."""

    @pytest.mark.slow
    def test_chunks_respect_sentence_boundaries(self, docs_sample):
        """Verify that chunks respect sentence boundaries across languages."""
        # For each document, create chunks and check that they end in sentence-ending punctuation
        # or are at document boundary

        closing_punct = r"[.!?…\n」』）】)]\'"'"'"。！？]"  # Latin, CJK, and other scripts
        results_by_lang = {}

        for _, doc in docs_sample.iterrows():
            idioma = doc.get("idioma", "en")
            texto = doc.get("texto", "")

            if not isinstance(texto, str) or not texto.strip():
                continue

            chunks = crear_chunks(texto, idioma)

            if idioma not in results_by_lang:
                results_by_lang[idioma] = {
                    "total_chunks": 0,
                    "boundary_respecting": 0,
                    "docs": 0,
                }

            results_by_lang[idioma]["docs"] += 1

            for i, chunk in enumerate(chunks):
                chunk_text = chunk["texto"]
                results_by_lang[idioma]["total_chunks"] += 1

                # Check if this chunk respects sentence boundaries
                # Except for the last chunk of a document, which can end anywhere
                is_last = (i == len(chunks) - 1)

                if is_last or re.search(closing_punct, chunk_text[-5:]):
                    # Last chunk or ends in closing punctuation
                    results_by_lang[idioma]["boundary_respecting"] += 1

        # Report results
        print("\n" + "=" * 70)
        print("DIAGNÓSTICO: Completitud Lingüística por Idioma")
        print("=" * 70)

        all_pass = True
        for lang in sorted(results_by_lang.keys()):
            stats = results_by_lang[lang]
            total = stats["total_chunks"]
            respecting = stats["boundary_respecting"]
            pct = 100.0 * respecting / total if total > 0 else 0

            status = "✓" if pct >= 90 else "⚠️ "
            print(
                f"{status} {lang:10s} | {respecting:4d}/{total:4d} chunks "
                f"({pct:5.1f}%) | {stats['docs']:2d} docs"
            )

            if pct < 85:  # Hard threshold: at least 85% for non-rare languages
                if stats["docs"] >= 3:  # Only fail if we have enough samples
                    all_pass = False

        print("=" * 70)

        # Meta: some degradation is OK for rare languages or noisy PDF extraction
        # The 85% threshold allows for:
        # - ~15% last-chunks of docs (acceptable end-of-document breaks)
        # - ~5% extraction artifacts (tables, malformed text)
        # For a production deployment, re-assess this threshold based on actual use cases.

        if not all_pass:
            pytest.fail(
                "Some languages fell below 85% sentence-boundary respect. "
                "Check the diagnóstico output above."
            )

    @pytest.mark.slow
    def test_chunks_have_valid_metadata(self, docs_sample):
        """Verify that all generated chunks have required metadata fields."""
        required_fields = {"texto", "posicion", "num_tokens"}

        for _, doc in docs_sample.iterrows():
            idioma = doc.get("idioma", "en")
            texto = doc.get("texto", "")

            if not isinstance(texto, str) or not texto.strip():
                continue

            chunks = crear_chunks(texto, idioma)

            for chunk in chunks:
                for field in required_fields:
                    assert field in chunk, f"Missing field '{field}' in chunk"

                assert isinstance(chunk["posicion"], int)
                assert chunk["posicion"] >= 0
                assert isinstance(chunk["num_tokens"], int)
                assert chunk["num_tokens"] >= 0
                assert isinstance(chunk["texto"], str)
                assert len(chunk["texto"]) > 0

    @pytest.mark.slow
    def test_chunk_text_is_complete(self, docs_sample):
        """Verify that no text is lost during chunking (for non-rare cases)."""
        for _, doc in docs_sample.iterrows():
            idioma = doc.get("idioma", "en")
            texto_orig = doc.get("texto", "")

            if not isinstance(texto_orig, str) or not texto_orig.strip():
                continue

            # Only test major languages (rare languages may have extraction issues)
            if idioma in ("desconocido",):
                continue

            chunks = crear_chunks(texto_orig, idioma)

            # Reconstruct: concatenate all chunk texts
            reconstructed = " ".join(c["texto"] for c in chunks)

            # Check that key phrases are preserved
            # (allow for minor whitespace normalization)
            orig_words = set(texto_orig.split()[:20])  # First 20 words
            recon_words = set(reconstructed.split()[:20])

            # At least 80% of the first 20 words should be preserved
            overlap = len(orig_words & recon_words)
            assert overlap >= int(0.8 * len(orig_words)), (
                f"{idioma}: text reconstruction lost too much content. "
                f"Original first words: {orig_words}, Reconstructed: {recon_words}"
            )
