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
