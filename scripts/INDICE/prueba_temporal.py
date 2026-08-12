"""
PRUEBA TEMPORAL — no es el generador.py final de la entrega.

Corre las 50 preguntas oficiales contra el índice actual (parcial: 955/1826
docs, ~52% del corpus, ver análisis de cobertura) para validar el pipeline
de recuperación end-to-end antes de decidir si vale la pena arreglar la
extracción primero. Los resultados NO son representativos del desempeño
final — les falta ~48% del corpus (sobre todo PDFs no extraídos).

Sigue el esquema de resultados.jsonl del spec (Sección 9.3):
- 3 documentos por query (dedup por doc_id, score = máximo entre sus chunks)
- 10 fragmentos por query, cada uno <= 250 palabras (los chunks reales
  tienen mediana de ~549 palabras, así que casi todos se dividen en
  sub-fragmentos respetando límites de oración vía el segmentador de
  Junior 2 — mismo chunk_id se repite entre sub-fragmentos, como permite
  el spec para trazabilidad).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from buscar import cargar_indice, cargar_modelo, buscar
from segmentacion import get_sentence_segmenter
from preguntas_50 import PREGUNTAS

MAX_PALABRAS_FRAGMENTO = 250
POOL_SIZE = 30  # cuántos chunks candidatos considerar por query


def dividir_en_fragmentos(texto: str, idioma: str, max_palabras: int = MAX_PALABRAS_FRAGMENTO):
    """Split texto into <=max_palabras sub-fragments, respecting sentence boundaries."""
    try:
        segmenter = get_sentence_segmenter(idioma)
        oraciones = segmenter(texto)
    except Exception:
        oraciones = [texto]

    fragmentos = []
    actual = []
    palabras_actual = 0

    for oracion in oraciones:
        n_palabras = len(oracion.split())
        if actual and palabras_actual + n_palabras > max_palabras:
            fragmentos.append(" ".join(actual))
            actual = []
            palabras_actual = 0
        actual.append(oracion)
        palabras_actual += n_palabras

    if actual:
        fragmentos.append(" ".join(actual))

    # Caso borde: una sola oración ya excede el límite -> se preserva completa
    # (igual que crear_chunks: preferible a cortar a mitad de frase).
    return fragmentos if fragmentos else [texto]


def responder_query(query_id: str, query_text: str, index, metadata_records, modelo):
    resultados = buscar(query_text, index, metadata_records, modelo, k=POOL_SIZE)

    # --- documentos: top 3 por doc_id, score = máximo entre sus chunks ---
    mejor_score_por_doc = {}
    for r in resultados:
        doc_id = r["doc_id"]
        if doc_id not in mejor_score_por_doc or r["score"] > mejor_score_por_doc[doc_id]:
            mejor_score_por_doc[doc_id] = r["score"]

    top_docs = sorted(mejor_score_por_doc.items(), key=lambda x: -x[1])[:3]
    documents = [{"rank": i + 1, "doc_id": doc_id} for i, (doc_id, _) in enumerate(top_docs)]

    # --- fragmentos: top 10 sub-fragmentos <= 250 palabras, en orden de score ---
    fragments = []
    for r in resultados:
        if len(fragments) >= 10:
            break
        sub_fragmentos = dividir_en_fragmentos(r["texto"], r.get("idioma", "en"))
        for sub in sub_fragmentos:
            if len(fragments) >= 10:
                break
            fragments.append({
                "rank": len(fragments) + 1,
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "text": sub,
            })

    return {
        "query_id": query_id,
        "documents": documents,
        "fragments": fragments,
    }


def main():
    index_dir = Path("entrega/base_vectorial/encoder_bgem3")
    output_path = Path("salida_v2/resultados_prueba_temporal.jsonl")

    print(f"Cargando índice desde {index_dir} ...")
    index, metadata_records = cargar_indice(index_dir)
    print(f"Índice: {index.ntotal:,} vectores (corpus parcial, ~52%)")

    print("Cargando modelo bge-m3 ...")
    modelo = cargar_modelo()

    resultados_todos = []
    for query_id in sorted(PREGUNTAS.keys()):
        query_text = PREGUNTAS[query_id]
        resultado = responder_query(query_id, query_text, index, metadata_records, modelo)
        resultados_todos.append(resultado)
        n_frag_ok = sum(1 for f in resultado["fragments"] if len(f["text"].split()) <= MAX_PALABRAS_FRAGMENTO)
        print(
            f"{query_id}: {len(resultado['documents'])} docs, "
            f"{len(resultado['fragments'])} fragmentos "
            f"({n_frag_ok}/{len(resultado['fragments'])} <= 250 palabras)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in resultados_todos:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✓ {output_path} ({len(resultados_todos)} líneas)")
    print("RECORDATORIO: esto es una prueba temporal sobre corpus parcial (52%).")
    print("No representa el desempeño final — falta arreglar extracción de PDFs.")


if __name__ == "__main__":
    main()
