"""
Core (testable) logic for the FAISS vector index: metadata shaping,
index construction, normalization checks, and search.

Kept model-agnostic and encoder-agnostic on purpose: every function that
needs embeddings takes them (or an `encode_fn`) as a parameter, so tests
can inject a stub encoder instead of loading BAAI/bge-m3.
"""
from typing import Any, Callable, Dict, List

import numpy as np
import faiss

REQUIRED_FIELDS = [
    "doc_id", "chunk_id", "fuente", "formato",
    "fenomeno", "posicion", "num_tokens", "texto",
]
EXTRA_FIELDS = ["observatorio", "idioma", "titulo", "url", "fecha"]


def chunk_row_to_metadata_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build one metadata.jsonl record from a chunks.parquet row (as dict)."""
    record = {}
    for field in REQUIRED_FIELDS:
        record[field] = row[field]
    record["fenomeno"] = int(record["fenomeno"])
    record["posicion"] = int(record["posicion"])
    record["num_tokens"] = int(record["num_tokens"])
    for field in EXTRA_FIELDS:
        if field in row:
            record[field] = row[field]
    return record


def is_l2_normalized(vectors: np.ndarray, tol: float = 1e-3) -> bool:
    """Check that every row has unit L2 norm (within tolerance)."""
    if len(vectors) == 0:
        return True
    norms = np.linalg.norm(vectors, axis=1)
    return bool(np.all(np.abs(norms - 1.0) < tol))


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Build an IndexFlatIP over already-normalized vectors (inner product
    over unit vectors == cosine similarity). Vectors are added in the
    given row order, which becomes the FAISS internal id order.
    """
    if not is_l2_normalized(vectors):
        raise ValueError(
            "Vectors must be L2-normalized before building an IndexFlatIP "
            "(inner product == cosine similarity only for unit vectors)."
        )
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def search(
    query: str,
    index: faiss.Index,
    metadata_records: List[Dict[str, Any]],
    encode_fn: Callable[[List[str]], np.ndarray],
    k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Encode `query` with `encode_fn` (must return L2-normalized vectors,
    matching how the index was built) and return the top-k metadata
    records with their similarity score, best first.
    """
    query_vector = encode_fn([query])
    query_vector = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32))

    scores, ids = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        record = dict(metadata_records[idx])
        record["score"] = float(score)
        results.append(record)
    return results
