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
