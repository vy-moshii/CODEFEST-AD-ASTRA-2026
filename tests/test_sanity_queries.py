"""
Integration test: run sanity-check queries against the real FAISS index
and confirm the top results are topically relevant.

Uses keyword presence rather than the `fenomeno` metadata field to judge
relevance: `fenomeno` in this corpus is assigned per source institution/folder
by ADL (see ruta_relativa in the original extraction), not per-document
semantic topic — some institutions (e.g. CEEEP) publish AI/defense articles
that are filed under fenomeno=3 (Dinámicas Territoriales) rather than
fenomeno=1 (IA y Capacidades Estratégicas). Matching on fenomeno therefore
produces false negatives even when retrieval is working correctly.

Slow: loads the real bge-m3 model and the real index. Skipped automatically
if the index hasn't been generated yet (entrega/base_vectorial/encoder_bgem3/).
"""
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "INDICE"))

INDEX_DIR = Path("entrega/base_vectorial/encoder_bgem3")

# (query, keywords expected somewhere in the top-3 combined text, case-insensitive)
QUERIES = [
    (
        "¿Cómo se está utilizando la inteligencia artificial en capacidades de defensa?",
        ["inteligencia artificial", "artificial intelligence"],
    ),
    (
        "How is artificial intelligence being applied to military and defense capabilities?",
        ["artificial intelligence", "military", "inteligencia artificial"],
    ),
    (
        "¿Qué riesgos existen para la seguridad del entorno espacial y los satélites?",
        ["space", "satellite", "espacial", "satélite"],
    ),
    (
        "¿Cómo operan los grupos armados organizados en el territorio?",
        ["armados", "armed group", "territorio", "narcotráfico"],
    ),
]


@pytest.mark.slow
def test_index_and_metadata_are_aligned():
    if not (INDEX_DIR / "index.faiss").exists():
        pytest.skip(f"Index not available at {INDEX_DIR}")

    from buscar import cargar_indice

    index, metadata_records = cargar_indice(INDEX_DIR)
    assert index.ntotal == len(metadata_records)
    assert index.ntotal > 0


@pytest.mark.slow
def test_sanity_queries_are_topically_relevant():
    if not (INDEX_DIR / "index.faiss").exists():
        pytest.skip(f"Index not available at {INDEX_DIR}")

    from buscar import cargar_indice, cargar_modelo, buscar

    index, metadata_records = cargar_indice(INDEX_DIR)
    modelo = cargar_modelo()

    print("\n" + "=" * 80)
    print("SANITY CHECK: relevancia temática del top-3 por query")
    print("=" * 80)

    n_correct = 0
    for query, keywords in QUERIES:
        resultados = buscar(query, index, metadata_records, modelo, k=3)
        assert len(resultados) > 0

        combined_text = " ".join(r["texto"] for r in resultados).lower()
        is_relevant = any(kw.lower() in combined_text for kw in keywords)
        n_correct += int(is_relevant)

        status = "✓" if is_relevant else "⚠️ "
        top_score = resultados[0]["score"]
        print(
            f"{status} query={query[:60]!r} | top1_score={top_score:.4f} "
            f"| keywords_found={is_relevant}"
        )

    print("=" * 80)
    # A full miss across all queries signals something structurally wrong
    # with the index (e.g. wrong texts embedded, id misalignment), not noise.
    assert n_correct >= len(QUERIES) // 2 + 1, (
        f"Only {n_correct}/{len(QUERIES)} sanity queries were topically relevant"
    )
