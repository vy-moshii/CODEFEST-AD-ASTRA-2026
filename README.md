# CodeFest AD ASTRA 2026

## Descripción

Proyecto desarrollado para el procesamiento y análisis del corpus documental
del CodeFest AD ASTRA 2026.

El proyecto incluye un pipeline de extracción de información capaz de procesar
diferentes formatos documentales y consolidar los resultados en archivos
estructurados para su posterior análisis.

## Objetivo

Construir una representación estructurada y consultable del corpus,
extrayendo texto y metadatos de los documentos disponibles.

## Formatos procesados

El pipeline permite trabajar con:

- PDF
- JSON
- CSV
- XLSX
- TXT
- JPG
- AVIF
- PBF / vector tiles

## Extractor

La versión principal del extractor es:

`src/extraer_documentos_v4_por_windows.py`

Las primeras versiones del extractor se conservan únicamente como referencia
del desarrollo.

v2: primera evolución del extractor.
v3: versión adaptada para Windows.
v4: versión actual y recomendada.

El extractor:

- inventaría los documentos del corpus;
- clasifica los archivos por formato;
- extrae texto y metadatos;
- procesa archivos grandes de forma separada;
- procesa archivos PBF;
- evita duplicados en los vector tiles;
- permite reanudar procesos;
- genera resultados estructurados.

## Ejecución

Desde la raíz del proyecto:

```powershell
python src/extraer_documentos_v4_por_windows.py --limite 10 --procesos 1 --procesos-grandes 1  

Si se quiere procesar todo el corpus:
```powershell
python src/extraer_documentos_v4_por_windows.py --procesos 1 --procesos-grandes 1

Si se quiere iniciar desde cero se usa el mismo comando con --reiniciar al final

""Salidas del programa
El procesamiento genera localmente:

salida/documentos.jsonl
salida/documentos.parquet
salida/documentos_metadata.csv
salida/errores.csv
salida/resumen.json

Estos archivos no forman parte del repositorio debido a su tamaño y son
generados nuevamente mediante el pipeline. De igual manera el corpus completo no se encuentra en el repo por su tamaño.

## Chunking (Etapa 1 — Etapa 2026)

El pipeline de chunking (`scripts/CHUNKS/chunking.py`) divide documentos en fragmentos respetando límites de oración (completitud lingüística obligatoria para CODEFEST AD ASTRA).

### Features (v2)

- **Segmentación multilingüe**: usa `pysbd` (~22 idiomas) con fallbacks por regex para idiomas no soportados (pt, ko, ms, qu).
- **Sentence-aware**: empaqueta oraciones completas, nunca corta a mitad de frase.
- **Paralelización**: `ProcessPoolExecutor` con checkpoint JSONL resumible (no reprocesa si se interrumpe).
- **Metadata completa**: genera campos obligatorios (`posicion`, `num_tokens`, `fenomeno` como int).

### Ejecución

```bash
# Instalar dependencias
pip install -r requirements.txt

# Correr chunking sobre 1745 documentos OK
python scripts/CHUNKS/chunking.py --procesos 6

# Con checkpoint resumible (si se interrumpe, retoma desde último doc procesado)
python scripts/CHUNKS/chunking.py --procesos 6 --check-battery

# Desde cero (borra checkpoint anterior)
python scripts/CHUNKS/chunking.py --procesos 6 --reiniciar
```

### Salidas

```
salida/chunks.parquet       # Parquet v2 (77,853+ chunks, multilingüe, sentence-aware)
salida/errores_chunking.csv # Documentos que fallaron (si aplica)
salida/.chunking_checkpoint.jsonl  # Checkpoint resumible (se borra al completar)
```

### Testing

```bash
# Tests unitarios (fixtures sintéticas, sin corpus real)
pytest tests/ -m "not slow" -v

# Tests de integración (requiere documentos.parquet)
pytest tests/ -m slow -v
```

Campos de `chunks.parquet`:
- `chunk_id`, `doc_id`, `fuente`, `formato`, `fenomeno` (int), `posicion`, `num_tokens`, `texto`
- Plus: `observatorio`, `idioma`, `titulo`, `url`, `fecha`, `n_caracteres_chunk`, `n_palabras_chunk`

## Índice vectorial (Etapa 1 — Senior 1)

Genera embeddings con **BAAI/bge-m3** (encoder multilingüe, familia BERT, licencia MIT,
contexto de 8192 tokens) y construye un índice FAISS (`IndexFlatIP`, búsqueda exacta por
similitud coseno vía producto interno sobre vectores normalizados).

### Ejecución

```bash
# Genera entrega/base_vectorial/encoder_bgem3/{index.faiss,metadata.jsonl}
python scripts/INDICE/generar_indice.py

# Sanity check manual: corre queries de prueba y muestra el top-5
python scripts/INDICE/sanity_check.py
```

Cachea los embeddings en `salida_v2/embeddings_bgem3.npy` para no recodificar si un paso
posterior falla; usar `--force-encode` para regenerar desde cero.

### Notas de diseño

- Corre en un solo proceso (no `ProcessPoolExecutor`): todo el cómputo es en la GPU vía
  MPS (Apple Silicon), paralelizar entre procesos no ayuda — competirían por el mismo
  dispositivo y cada uno cargaría el modelo (~2.3GB) por separado.
- `max_seq_length=2048`: cubre el p99 real del corpus (~1,138 tokens) con margen; solo
  trunca los pocos chunks de basura de extracción (JSON/CSV sin estructura de oraciones,
  hasta 47,795 tokens) que si no colapsan el buffer de atención de MPS (`batch_size × heads
  × seq² `). El texto completo se sigue entregando sin modificar en `metadata.jsonl` — el
  truncado solo afecta el embedding.
- `fenomeno` se asigna por institución/carpeta de origen (según ADL), no por tema
  semántico del documento — algunas instituciones publican contenido de un fenómeno
  distinto al que su carpeta indica. No usar `fenomeno` como proxy de relevancia temática
  al validar resultados de búsqueda.
- Si `faiss` y `torch` se importan en el mismo proceso (p.ej. corriendo toda la suite de
  tests junta), macOS puede abortar por runtimes de OpenMP duplicados
  (`Fatal Python error: Aborted`) — mitigado en `tests/conftest.py` con
  `KMP_DUPLICATE_LIB_OK=TRUE`.

### Testing

```bash
pytest tests/test_indice_core.py -v          # rápido, sin descargar el modelo
pytest tests/test_sanity_queries.py -m slow -v  # usa el índice y modelo reales
```
