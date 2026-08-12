"""
Manual sanity check: run a handful of test queries against the real
index and print the top-5 results for visual review, before handing the
index off to Senior 2.

Covers a mix of es/en and the 3 fenómenos of the corpus (IA en defensa,
seguridad espacial, dinámicas territoriales / GAO-GAOR-GDO).

Note: `fenomeno` is assigned per source institution/folder by ADL, not per
document topic — an institution like CEEEP can publish AI/defense content
filed under fenomeno=3. The match marker below is a rough visual aid, not
ground truth: judge relevance by reading the actual retrieved text.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from buscar import cargar_indice, cargar_modelo, buscar

QUERIES = [
    ("es", 1, "¿Cómo se está utilizando la inteligencia artificial en capacidades de defensa?"),
    ("en", 1, "How is artificial intelligence being applied to military and defense capabilities?"),
    ("es", 2, "¿Qué riesgos existen para la seguridad del entorno espacial y los satélites?"),
    ("en", 2, "What are the main threats to space security and satellite infrastructure?"),
    ("es", 3, "¿Cómo operan los grupos armados organizados en el territorio?"),
]


def main():
    parser = argparse.ArgumentParser(description="Manual sanity check for the FAISS index")
    parser.add_argument(
        "--index-dir", type=Path,
        default=Path("entrega/base_vectorial/encoder_bgem3"),
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    print(f"Cargando índice desde {args.index_dir} ...")
    index, metadata_records = cargar_indice(args.index_dir)
    print(f"Índice cargado: {index.ntotal:,} vectores\n")

    print("Cargando modelo bge-m3 ...")
    modelo = cargar_modelo()
    print("Modelo cargado.\n")

    for idioma, fenomeno_esperado, query in QUERIES:
        print("=" * 90)
        print(f"[{idioma}] fenómeno esperado: {fenomeno_esperado} | QUERY: {query}")
        print("=" * 90)

        resultados = buscar(query, index, metadata_records, modelo, k=args.k)
        for i, r in enumerate(resultados, 1):
            texto_preview = r["texto"][:200].replace("\n", " ")
            match = "✓" if r["fenomeno"] == fenomeno_esperado else " "
            print(
                f"{match} [{i}] score={r['score']:.4f} doc_id={r['doc_id']} "
                f"fenomeno={r['fenomeno']} idioma={r.get('idioma', '?')}"
            )
            print(f"      {texto_preview}...")
        print()


if __name__ == "__main__":
    main()
