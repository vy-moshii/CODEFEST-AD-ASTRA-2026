"""
Fast unit tests for indice_core.py — no model download, no real corpus.
Uses synthetic embeddings and a stub encode_fn.
"""
import numpy as np
import pytest

from indice_core import (
    chunk_row_to_metadata_record,
    is_l2_normalized,
    build_faiss_index,
    search,
)


def make_unit_vector(dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


class TestIsL2Normalized:
    def test_normalized_vectors_pass(self):
        vectors = np.array([make_unit_vector(16, i) for i in range(5)])
        assert is_l2_normalized(vectors)

    def test_unnormalized_vectors_fail(self):
        vectors = np.array([[3.0, 4.0], [1.0, 0.0]])  # norm 5 and 1
        assert not is_l2_normalized(vectors)

    def test_empty_array_passes(self):
        assert is_l2_normalized(np.empty((0, 8)))


class TestBuildFaissIndex:
    def test_rejects_unnormalized_vectors(self):
        vectors = np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            build_faiss_index(vectors)

    def test_index_size_matches_input(self):
        dim = 32
        vectors = np.array([make_unit_vector(dim, i) for i in range(10)], dtype=np.float32)
        index = build_faiss_index(vectors)
        assert index.ntotal == 10
        assert index.d == dim

    def test_identical_vector_is_top1(self):
        """A query identical to a stored vector must retrieve itself as the top match."""
        dim = 32
        vectors = np.array([make_unit_vector(dim, i) for i in range(20)], dtype=np.float32)
        index = build_faiss_index(vectors)

        query = vectors[7:8]  # exact copy of vector at id 7
        scores, ids = index.search(query, 1)

        assert ids[0][0] == 7
        assert scores[0][0] == pytest.approx(1.0, abs=1e-4)


class TestSearch:
    def test_search_returns_metadata_and_score_in_faiss_order(self):
        dim = 16
        vectors = np.array([make_unit_vector(dim, i) for i in range(5)], dtype=np.float32)
        index = build_faiss_index(vectors)

        metadata_records = [
            {"doc_id": f"DOC-{i}", "chunk_id": f"DOC-{i}_chunk_0", "texto": f"texto {i}"}
            for i in range(5)
        ]

        def stub_encode_fn(texts):
            # "query" always resolves to the vector for doc 3, to make the
            # top-1 result deterministic and independent of query text.
            return vectors[3:4]

        results = search("cualquier query", index, metadata_records, stub_encode_fn, k=3)

        assert len(results) == 3
        assert results[0]["doc_id"] == "DOC-3"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)
        assert "score" in results[0]

    def test_search_respects_k(self):
        dim = 8
        vectors = np.array([make_unit_vector(dim, i) for i in range(10)], dtype=np.float32)
        index = build_faiss_index(vectors)
        metadata_records = [{"doc_id": f"DOC-{i}", "texto": "x"} for i in range(10)]

        def stub_encode_fn(texts):
            return vectors[0:1]

        results = search("q", index, metadata_records, stub_encode_fn, k=4)
        assert len(results) == 4


class TestChunkRowToMetadataRecord:
    def _base_row(self):
        return {
            "doc_id": "DOC-001",
            "chunk_id": "DOC-001_chunk_00000",
            "fuente": "documento_original.pdf",
            "formato": "pdf",
            "fenomeno": 1,
            "posicion": 0,
            "num_tokens": 120,
            "texto": "Texto de prueba.",
            "idioma": "es",
            "titulo": "Título",
            "n_caracteres_chunk": 17,  # not in REQUIRED_FIELDS/EXTRA_FIELDS, must be dropped
        }

    def test_required_fields_present(self):
        record = chunk_row_to_metadata_record(self._base_row())
        for field in ["doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion", "num_tokens", "texto"]:
            assert field in record

    def test_types_are_coerced_to_int(self):
        row = self._base_row()
        row["fenomeno"] = "1"
        row["posicion"] = "0"
        row["num_tokens"] = "120"
        record = chunk_row_to_metadata_record(row)
        assert isinstance(record["fenomeno"], int)
        assert isinstance(record["posicion"], int)
        assert isinstance(record["num_tokens"], int)

    def test_extra_fields_included_when_present(self):
        record = chunk_row_to_metadata_record(self._base_row())
        assert record["idioma"] == "es"
        assert record["titulo"] == "Título"

    def test_unlisted_fields_are_dropped(self):
        record = chunk_row_to_metadata_record(self._base_row())
        assert "n_caracteres_chunk" not in record

    def test_missing_extra_field_is_omitted_not_errored(self):
        row = self._base_row()
        del row["idioma"]
        record = chunk_row_to_metadata_record(row)
        assert "idioma" not in record
