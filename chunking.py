import re
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT = Path(r".\salida\documentos.parquet")
OUTPUT = Path(r".\salida\chunks.parquet")

# Tamaño aproximado de cada chunk en caracteres.
# ~1000 tokens dependiendo del idioma.
CHUNK_SIZE = 4000

# Solapamiento entre chunks consecutivos.
CHUNK_OVERLAP = 500

# No crear chunks de textos demasiado pequeños.
MIN_CHUNK_CHARS = 200


# ============================================================
# LIMPIEZA
# ============================================================

def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""

    # Normalizar saltos de línea
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")

    # Eliminar espacios repetidos
    texto = re.sub(r"[ \t]+", " ", texto)

    # Reducir demasiados saltos de línea
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


# ============================================================
# CHUNKING
# ============================================================

def crear_chunks(texto):
    """
    Divide el texto intentando respetar párrafos.
    Usa ventanas de CHUNK_SIZE caracteres con solapamiento.
    """

    if len(texto) <= CHUNK_SIZE:
        if len(texto) >= MIN_CHUNK_CHARS:
            return [texto]
        return []

    chunks = []

    inicio = 0
    longitud = len(texto)

    while inicio < longitud:
        limite = min(inicio + CHUNK_SIZE, longitud)

        # Intentar terminar en un salto de párrafo
        corte = texto.rfind("\n\n", inicio, limite)

        # Si no hay párrafo, buscar salto de línea
        if corte <= inicio:
            corte = texto.rfind("\n", inicio, limite)

        # Si tampoco hay salto, buscar espacio
        if corte <= inicio:
            corte = texto.rfind(" ", inicio, limite)

        # Si no encontramos un corte razonable,
        # cortamos directamente en el límite.
        if corte <= inicio:
            corte = limite

        chunk = texto[inicio:corte].strip()

        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        # Avanzar manteniendo overlap
        siguiente = corte - CHUNK_OVERLAP

        if siguiente <= inicio:
            siguiente = corte

        inicio = siguiente

    return chunks


# ============================================================
# PROCESAMIENTO
# ============================================================

def main():

    print("=" * 70)
    print("CHUNKING DEL CORPUS")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {INPUT}"
        )

    print(f"\nLeyendo: {INPUT}")

    df = pd.read_parquet(INPUT)

    print(f"Documentos cargados: {len(df):,}")

    # Solo documentos válidos con texto
    df = df[
        df["estado"].eq("ok")
        & df["texto"].notna()
    ].copy()

    print(f"Documentos válidos: {len(df):,}")

    registros = []

    total_chunks = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        texto = limpiar_texto(row["texto"])

        if not texto:
            continue

        chunks = crear_chunks(texto)

        for numero, chunk in enumerate(chunks):

            registros.append({
                "chunk_id": f"{row['doc_id']}_chunk_{numero:05d}",
                "doc_id": row["doc_id"],
                "chunk_index": numero,
                "texto": chunk,

                # Metadata original
                "observatorio": row["observatorio"],
                "fenomeno": row["fenomeno"],
                "fuente": row["fuente"],
                "formato": row["formato"],
                "idioma": row["idioma"],
                "titulo": row["titulo"],
                "url": row["url"],
                "fecha": row["fecha"],

                # Información útil para control
                "n_caracteres_chunk": len(chunk),
                "n_palabras_chunk": len(chunk.split()),
            })

        total_chunks += len(chunks)

        if i % 100 == 0:
            print(
                f"Procesados: {i:,}/{len(df):,} | "
                f"chunks: {total_chunks:,}"
            )

    chunks_df = pd.DataFrame(registros)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    chunks_df.to_parquet(
        OUTPUT,
        index=False,
        engine="pyarrow"
    )

    print("\n" + "=" * 70)
    print("CHUNKING TERMINADO")
    print("=" * 70)

    print(f"Documentos procesados: {len(df):,}")
    print(f"Chunks generados:      {len(chunks_df):,}")

    if len(chunks_df) > 0:
        print(
            f"Promedio chunks/doc:   "
            f"{len(chunks_df) / len(df):.2f}"
        )

        print(
            f"Caracteres promedio:   "
            f"{chunks_df['n_caracteres_chunk'].mean():.0f}"
        )

        print(
            f"Palabras promedio:     "
            f"{chunks_df['n_palabras_chunk'].mean():.0f}"
        )

    print(f"\nSalida: {OUTPUT}")


if __name__ == "__main__":
    main()