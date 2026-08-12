"""
Reusable module to load the FAISS index + metadata and run searches.

Meant to be imported both by sanity_check.py (manual eyeball checks) and,
later, by Senior 2's generador.py — so index-loading/search logic isn't
duplicated across scripts.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from indice_core import search as _search

MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 8192


def cargar_indice(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load index.faiss + metadata.jsonl from an encoder_<nombre>/ directory."""
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))

    metadata_records = []
    with open(index_dir / "metadata.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata_records.append(json.loads(line))

    if index.ntotal != len(metadata_records):
        raise ValueError(
            f"Index/metadata mismatch: index has {index.ntotal} vectors, "
            f"metadata.jsonl has {len(metadata_records)} lines"
        )
    return index, metadata_records


def cargar_modelo(device: str = None) -> SentenceTransformer:
    if device is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def buscar(
    query: str,
    index: faiss.Index,
    metadata_records: List[Dict[str, Any]],
    modelo: SentenceTransformer,
    k: int = 10,
) -> List[Dict[str, Any]]:
    """Search the index for `query`, returning top-k metadata records + score."""
    encode_fn = lambda texts: modelo.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )
    return _search(query, index, metadata_records, encode_fn, k=k)
