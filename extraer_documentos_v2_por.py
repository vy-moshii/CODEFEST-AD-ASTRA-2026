#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CODEFEST AD ASTRA 2026 - Etapa 1
Preprocesamiento y extraccion de texto del corpus documental.

Convierte los 1826 archivos crudos del corpus en un DataFrame (una fila por
documento) con texto limpio y metadata, listo para la etapa de chunking.

NO se usa ningun modelo de lenguaje generativo (decoder). Toda la logica es
deterministica: parsers de formato, expresiones regulares y reglas de limpieza.

Uso:
    python3 extraer_documentos.py                 # corpus completo
    python3 extraer_documentos.py --limite 50     # prueba rapida
    python3 extraer_documentos.py --formatos PDF JSON
    python3 extraer_documentos.py --procesos 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = BASE_DIR / "CORPUS CODEFEST AD ASTRA 2026"
INVENTARIO = CORPUS_DIR / "Indice_Datos_Codefest.xlsx"
HOJA_INVENTARIO = "Inventario de Archivos"
SALIDA_DIR = BASE_DIR / "salida"

# Maximo de filas que se convierten a texto en archivos tabulares (CSV/XLSX y
# catalogos JSON). Los archivos bibliograficos de PubMed llegan a 111.775 filas;
# volcarlos completos generaria cientos de MB de citas casi identicas que
# dominarian el indice vectorial sin aportar valor semantico.
MAX_FILAS_TABULARES = None  # No truncar: se conservan todas las filas; el chunking se hará después.

# Un PDF con menos de este promedio de caracteres por pagina se considera
# escaneado / sin capa de texto y se marca para revision (posible OCR).
MIN_CHARS_POR_PAGINA = 100

# Longitud de texto usada para detectar el idioma.
MUESTRA_IDIOMA = 5000

# OCR: Tesseract debe estar instalado en Windows. Si no está en PATH,
# se puede indicar la ruta con la variable de entorno TESSERACT_CMD.
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)
OCR_LANG = "spa+eng+por"
OCR_DPI = 180
OCR_MIN_WORDS = 10

log = logging.getLogger("preproc")


# --------------------------------------------------------------------------
# Limpieza y normalizacion (Seccion 2.2 de la especificacion)
# --------------------------------------------------------------------------

# Caracteres de control excepto tab y salto de linea.
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Espacios "raros" (no-break space, espacios tipograficos, etc.).
RE_ESPACIOS_RAROS = re.compile(r"[   -   　]")
# Marcas invisibles de formato (zero-width, BOM, soft hyphen, bidi).
RE_INVISIBLES = re.compile(r"[­​-‏  ﻿‪-‮]")
# Lineas guia de indices y tablas de contenido: ". . . . . . . . 14"
RE_PUNTOS_GUIA = re.compile(r"(?:\.\s?){4,}")
RE_ESPACIOS = re.compile(r"[ \t]+")
RE_SALTOS = re.compile(r"\n{3,}")
RE_ESPACIO_ANTES_SALTO = re.compile(r"[ \t]+\n")


def limpiar_texto(texto: str) -> str:
    """Limpieza basica comun a todos los formatos.

    - Normaliza la codificacion a UTF-8 / Unicode NFC.
    - Elimina caracteres de control e invisibles.
    - Colapsa espacios redundantes conservando los saltos de parrafo.
    """
    if not texto:
        return ""
    # Normalizacion Unicode: unifica acentos compuestos vs. precompuestos, de
    # modo que "cion" con tilde sea siempre la misma secuencia de bytes.
    texto = unicodedata.normalize("NFC", texto)
    texto = RE_INVISIBLES.sub("", texto)
    texto = RE_ESPACIOS_RAROS.sub(" ", texto)
    texto = RE_CONTROL.sub("", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = RE_PUNTOS_GUIA.sub(" ", texto)
    texto = RE_ESPACIOS.sub(" ", texto)
    texto = RE_ESPACIO_ANTES_SALTO.sub("\n", texto)
    # Se conserva como maximo un salto doble: marca de parrafo para la etapa
    # de chunking del equipo.
    texto = RE_SALTOS.sub("\n\n", texto)
    return texto.strip()


def normalizar_para_comparar(linea: str) -> str:
    """Firma de una linea para detectar cabeceras/pies repetidos.

    Los numeros se reemplazan por '#' para que "Pagina 3" y "Pagina 47"
    compartan la misma firma.
    """
    s = linea.strip().lower()
    s = re.sub(r"\d+", "#", s)
    s = re.sub(r"\s+", " ", s)
    return s


RE_SOLO_NUMERO = re.compile(r"^\s*[-–—|]*\s*\d{1,4}\s*[-–—|]*\s*$")
RE_PAGINA = re.compile(
    r"^\s*(p[aá]g(ina)?\.?|page|pag\.)\s*\d+\s*(de|of|/)?\s*\d*\s*$", re.IGNORECASE
)


def es_numeracion(linea: str) -> bool:
    """True si la linea es solo un numero de pagina."""
    return bool(RE_SOLO_NUMERO.match(linea) or RE_PAGINA.match(linea))


def detectar_boilerplate(paginas: list[str], margen: int = 3) -> set[str]:
    """Detecta cabeceras y pies de pagina repetidos a lo largo del documento.

    Se miran las primeras y ultimas `margen` lineas de cada pagina; si una
    firma aparece en al menos la mitad de las paginas (y en 3 como minimo),
    se considera boilerplate sin valor informativo.
    """
    if len(paginas) < 4:
        return set()
    from collections import Counter

    conteo: Counter = Counter()
    for pag in paginas:
        lineas = [l for l in pag.split("\n") if l.strip()]
        if not lineas:
            continue
        candidatas = lineas[:margen] + lineas[-margen:]
        # set() para no contar dos veces una linea repetida dentro de la misma
        # pagina (documentos cortos con 1-2 lineas).
        for l in set(candidatas):
            firma = normalizar_para_comparar(l)
            if 0 < len(firma) <= 120:
                conteo[firma] += 1

    umbral = max(3, int(len(paginas) * 0.5))
    return {firma for firma, n in conteo.items() if n >= umbral}


# Guion de corte de palabra al final de linea: "innova-\ncion" -> "innovacion".
# Solo se une cuando la siguiente linea empieza en minuscula, para no fusionar
# guiones legitimos ("Franco-Alemana") ni comienzos de nombre propio.
RE_GUION_CORTE = re.compile(
    r"([A-Za-zÁÉÍÓÚÑÜáéíóúñü])[-‐‑]\n([a-záéíóúñü])"
)


def reconstruir_parrafos(texto_pagina: str) -> str:
    """Une las lineas de una pagina en parrafos.

    pdfplumber devuelve un salto de linea por cada linea fisica del PDF. Para
    que la etapa de chunking pueda detectar limites de oracion correctamente,
    se unen las lineas de un mismo bloque con espacios y se conservan los
    saltos dobles como separadores de parrafo.
    """
    # Primero se reparan las palabras cortadas por guion al final de linea.
    texto_pagina = RE_GUION_CORTE.sub(r"\1\2", texto_pagina)
    bloques = re.split(r"\n\s*\n", texto_pagina)
    salida = []
    for bloque in bloques:
        lineas = [l.strip() for l in bloque.split("\n") if l.strip()]
        if lineas:
            salida.append(" ".join(lineas))
    return "\n\n".join(salida)


# --------------------------------------------------------------------------
# Extractores por formato (Seccion 2.1 de la especificacion)
# --------------------------------------------------------------------------


def _obtener_idiomas_ocr(pytesseract):
    """Detecta qué modelos OCR (spa, eng, por) están instalados."""
    requeridos = ("spa", "eng", "por")
    try:
        disponibles = set(pytesseract.get_languages(config=""))
    except Exception as e:
        log.warning("No se pudieron consultar los idiomas de Tesseract: %s", e)
        disponibles = set()
    idiomas = [idioma for idioma in requeridos if idioma in disponibles]
    if not idiomas:
        raise RuntimeError(
            "Tesseract está instalado, pero no hay modelos spa/eng/por. "
            "Ejecuta 'tesseract --list-langs'."
        )
    return "+".join(idiomas), idiomas


def _configurar_tesseract():
    """Configura Tesseract si está instalado en una ruta conocida."""
    import pytesseract
    cmd = Path(TESSERACT_CMD)
    if cmd.exists():
        pytesseract.pytesseract.tesseract_cmd = str(cmd)
    elif not shutil.which("tesseract"):
        raise FileNotFoundError(f"No se encontró Tesseract en {cmd} ni en PATH.")

    global OCR_LANG
    OCR_LANG, idiomas = _obtener_idiomas_ocr(pytesseract)
    log.info("Idiomas OCR disponibles: %s", ", ".join(idiomas))
    return pytesseract


def _ocr_pagina_pdf(page, pytesseract) -> str:
    """Renderiza una página PDF y aplica OCR solo cuando hace falta."""
    from PIL import Image
    import fitz

    pix = page.get_pixmap(
        matrix=fitz.Matrix(OCR_DPI / 72.0, OCR_DPI / 72.0),
        alpha=False,
    )
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG)
    finally:
        img.close()


def extraer_pdf(ruta: Path) -> dict:
    """Extrae PDF con pdfplumber y activa OCR como fallback.

    Primero se intenta extracción de texto normal. Si la capa de texto es
    inexistente o demasiado pobre, se renderizan las páginas con PyMuPDF y se
    aplica Tesseract. Así no se OCRizan innecesariamente los PDF normales.
    """
    import pdfplumber

    paginas_crudas: list[str] = []
    with pdfplumber.open(ruta) as pdf:
        n_paginas = len(pdf.pages)
        metadatos_pdf = pdf.metadata or {}
        for pagina in pdf.pages:
            try:
                paginas_crudas.append(pagina.extract_text() or "")
            except Exception as exc:
                log.debug("pagina ilegible en %s: %s", ruta.name, exc)
                paginas_crudas.append("")
            finally:
                pagina.flush_cache()

    boilerplate = detectar_boilerplate(paginas_crudas)
    paginas_limpias = []
    for pag in paginas_crudas:
        lineas = []
        for linea in pag.split("\n"):
            if not linea.strip():
                lineas.append("")
                continue
            if es_numeracion(linea):
                continue
            if normalizar_para_comparar(linea) in boilerplate:
                continue
            lineas.append(linea)
        texto_pag = reconstruir_parrafos("\n".join(lineas))
        if texto_pag.strip():
            paginas_limpias.append(texto_pag)

    texto = "\n\n".join(paginas_limpias)
    n_chars = len(texto)
    promedio = n_chars / max(n_paginas, 1)
    metodo = "pdfplumber"
    detalle = None

    # OCR únicamente para PDF sin capa de texto o con extracción claramente pobre.
    if n_paginas and promedio < MIN_CHARS_POR_PAGINA:
        try:
            pytesseract = _configurar_tesseract()
            import fitz

            paginas_ocr = []
            with fitz.open(ruta) as pdf_ocr:
                for page in pdf_ocr:
                    texto_pagina = page.get_text("text") or ""
                    # Si la página ya tiene texto razonable, conservarlo.
                    if len(texto_pagina.strip()) >= MIN_CHARS_POR_PAGINA:
                        paginas_ocr.append(texto_pagina)
                    else:
                        paginas_ocr.append(_ocr_pagina_pdf(page, pytesseract))

            texto_ocr = "\n\n".join(
                reconstruir_parrafos(t) for t in paginas_ocr if t and t.strip()
            )
            if len(texto_ocr.strip()) >= max(50, n_paginas * 10):
                texto = texto_ocr
                n_chars = len(texto)
                metodo = "tesseract_ocr"
                detalle = "OCR aplicado por baja cantidad de texto extraible"
            else:
                metodo = "pdfplumber_sin_texto"
                detalle = "OCR no produjo texto suficiente"
        except ImportError as exc:
            metodo = "sin_texto_extraible"
            detalle = f"OCR no disponible: {exc}"
        except Exception as exc:
            metodo = "sin_texto_extraible"
            detalle = f"fallo OCR: {type(exc).__name__}: {exc}"[:300]

    estado = "ok" if texto.strip() else "sin_texto_extraible"
    if metodo == "tesseract_ocr" and not texto.strip():
        estado = "sin_texto_extraible"

    meta = {}
    if isinstance(metadatos_pdf, dict):
        titulo = metadatos_pdf.get("Title")
        if isinstance(titulo, str):
            titulo = titulo.strip()
            if 3 < len(titulo) < 300 and not re.fullmatch(r"[\w\-. ]+\.(indd|pdf|docx?|qxd)", titulo, re.I):
                meta["titulo"] = titulo

    return {
        "texto": texto,
        "n_paginas": n_paginas,
        "estado": estado,
        "meta": meta,
        "boilerplate_eliminado": len(boilerplate),
        "detalle": detalle,
        "metodo_extraccion": metodo,
    }

def _texto_de_lista(valor) -> str:
    """Convierte una lista de parrafos/strings en texto separado por parrafos."""
    if not isinstance(valor, list):
        return ""
    partes = [str(v).strip() for v in valor if isinstance(v, (str, int, float)) and str(v).strip()]
    return "\n\n".join(partes)


def _filas_a_texto(df: pd.DataFrame, max_filas: int | None = None) -> tuple[str, int, bool]:
    """Convierte un DataFrame en texto 'columna: valor | columna: valor'.

    Cada fila es una linea independiente, de modo que la etapa de chunking
    puede tratarla como unidad de fragmentacion. Las celdas vacias se omiten
    para no generar ruido.
    """
    total = len(df)
    truncado = max_filas is not None and total > max_filas
    if truncado:
        df = df.head(max_filas)

    # Se descartan las columnas de indice sin nombre que pandas crea al leer
    # CSV exportados con el indice incluido.
    columnas = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    df = df[columnas]

    lineas = []
    for fila in df.itertuples(index=False, name=None):
        pares = []
        for col, val in zip(columnas, fila):
            if val is None:
                continue
            if isinstance(val, float) and pd.isna(val):
                continue
            s = str(val).strip()
            if not s or s.lower() in ("nan", "nat", "none"):
                continue
            pares.append(f"{col}: {s}")
        if pares:
            lineas.append(" | ".join(pares))
    return "\n".join(lineas), total, truncado


def extraer_json(ruta: Path) -> dict:
    """Extrae texto de un JSON seleccionando explicitamente los campos de texto.

    El corpus tiene 20 esquemas JSON distintos. En lugar de codificar uno por
    observatorio, se despacha segun los campos presentes:

    - Articulos web: title + body_paragraphs (body_text se ignora porque es
      exactamente la concatenacion de body_paragraphs; se verifico en los 485
      archivos donde aparecen ambos).
    - CEEEP: no trae cuerpo, solo abstract academico.
    - CENIA: sections = [{heading, paragraphs}] + lists.
    - SWF report-data: content = {sections: {titulo: texto}}.
    - Catalogos/manifiestos: listas de registros -> se tratan como tabla.

    Los campos descriptivos (url, date, authors, tags) se devuelven como
    metadata del documento y no se mezclan con el cuerpo del texto.
    """
    with open(ruta, "r", encoding="utf-8") as fh:
        datos = json.load(fh)

    meta: dict = {}
    partes: list[str] = []
    subtipo = "desconocido"
    filas_totales = None
    truncado = False

    if isinstance(datos, list):
        # Catalogo / manifiesto de descargas: lista de registros homogeneos.
        subtipo = "catalogo_lista"
        if datos and isinstance(datos[0], dict):
            texto, filas_totales, truncado = _filas_a_texto(
                pd.DataFrame(datos), MAX_FILAS_TABULARES
            )
            partes.append(texto)
        elif not datos:
            subtipo = "catalogo_vacio"

    elif isinstance(datos, dict):
        titulo = datos.get("title") or datos.get("titulo")
        if isinstance(titulo, str) and titulo.strip():
            meta["titulo"] = titulo.strip()
            partes.append(titulo.strip())

        for campo_meta, destino in (
            ("url", "url"),
            ("source_url", "url"),
            ("date", "fecha"),
            ("doi", "doi"),
            ("issue", "issue"),
        ):
            val = datos.get(campo_meta)
            if isinstance(val, (str, int, float)) and str(val).strip():
                meta.setdefault(destino, str(val).strip())

        for campo_lista, destino in (
            ("authors", "autores"),
            ("tags", "etiquetas"),
            ("topics", "etiquetas"),
            ("categories", "etiquetas"),
            ("keywords", "etiquetas"),
        ):
            val = datos.get(campo_lista)
            if isinstance(val, list) and val:
                previos = meta.get(destino, "")
                nuevos = "; ".join(str(v).strip() for v in val if str(v).strip())
                meta[destino] = f"{previos}; {nuevos}".strip("; ") if previos else nuevos

        # --- Cuerpo del documento, por orden de preferencia ---
        if isinstance(datos.get("body_paragraphs"), list) and datos["body_paragraphs"]:
            subtipo = "articulo"
            partes.append(_texto_de_lista(datos["body_paragraphs"]))
        elif isinstance(datos.get("body_text"), str) and datos["body_text"].strip():
            subtipo = "articulo"
            partes.append(datos["body_text"].strip())
        elif isinstance(datos.get("sections"), list) and datos["sections"]:
            # CENIA: secciones con encabezado. El encabezado se conserva porque
            # es una senal estructural util para el chunking jerarquico.
            subtipo = "pagina_secciones"
            for sec in datos["sections"]:
                if not isinstance(sec, dict):
                    continue
                head = str(sec.get("heading") or "").strip()
                cuerpo = _texto_de_lista(sec.get("paragraphs"))
                bloque = f"{head}\n\n{cuerpo}" if head and cuerpo else (head or cuerpo)
                if bloque.strip():
                    partes.append(bloque.strip())
            listas = _texto_de_lista(datos.get("lists"))
            if listas:
                partes.append(listas)
        elif isinstance(datos.get("abstract"), str) and datos["abstract"].strip():
            # CEEEP: articulo de revista sin cuerpo, solo resumen academico.
            subtipo = "resumen_academico"
            partes.append(datos["abstract"].strip())
        elif isinstance(datos.get("content"), dict):
            # SWF report-data: {"sections": {"titulo": "texto", ...}}
            subtipo = "reporte_secciones"
            secciones = datos["content"].get("sections")
            if isinstance(secciones, dict):
                for head, cuerpo in secciones.items():
                    if isinstance(cuerpo, str) and cuerpo.strip():
                        partes.append(f"{head}\n\n{cuerpo.strip()}")
            meta_int = datos.get("metadata")
            if isinstance(meta_int, dict):
                if meta_int.get("title"):
                    meta.setdefault("titulo", str(meta_int["title"]).strip())
                if meta_int.get("description"):
                    partes.insert(1, str(meta_int["description"]).strip())
        elif isinstance(datos.get("excerpt"), str) and datos["excerpt"].strip():
            subtipo = "solo_extracto"
            partes.append(datos["excerpt"].strip())
        else:
            subtipo = "sin_cuerpo"

        # Alertas Tempranas: alerta_meta trae campos que si son contenido
        # (tipo de alerta, municipios en riesgo) y no meros descriptores. Se
        # conservan como metadata y ademas se anexan al texto porque describen
        # el objeto del documento y son muy buscables.
        am = datos.get("alerta_meta")
        if isinstance(am, dict) and am:
            subtipo = "alerta_temprana"
            for k in ("codigo", "tipo", "fecha_emision", "municipios"):
                if am.get(k):
                    meta[f"alerta_{k}"] = str(am[k]).strip()
            if am.get("fecha_emision"):
                meta.setdefault("fecha", str(am["fecha_emision"]).strip())
            contexto = []
            if am.get("tipo"):
                contexto.append(f"Tipo de alerta: {am['tipo']}")
            if am.get("municipios"):
                contexto.append(f"Municipios: {am['municipios']}")
            if contexto:
                partes.insert(1 if len(partes) > 1 else len(partes), ". ".join(contexto) + ".")

    texto = "\n\n".join(p for p in partes if p and p.strip())
    return {
        "texto": texto,
        "estado": "ok" if texto.strip() else "vacio",
        "meta": meta,
        "subtipo": subtipo,
        "filas_totales": filas_totales,
        "truncado": truncado,
    }


def _leer_csv_robusto(ruta: Path) -> pd.DataFrame:
    """Lee un CSV probando codificaciones y separadores habituales."""
    import chardet

    with open(ruta, "rb") as fh:
        muestra = fh.read(200_000)
    detectado = (chardet.detect(muestra) or {}).get("encoding") or "utf-8"

    intentos = []
    for enc in (detectado, "utf-8", "utf-8-sig", "latin-1"):
        if enc and enc not in intentos:
            intentos.append(enc)

    ultimo_error: Exception | None = None
    for enc in intentos:
        try:
            return pd.read_csv(
                ruta,
                encoding=enc,
                sep=None,             # deja que pandas infiera el delimitador
                engine="python",
                on_bad_lines="skip",  # una fila malformada no invalida el archivo
                dtype=str,
            )
        except Exception as exc:
            ultimo_error = exc
    raise RuntimeError(f"no se pudo leer el CSV con {intentos}: {ultimo_error}")


def extraer_csv(ruta: Path) -> dict:
    df = _leer_csv_robusto(ruta)
    texto, total, truncado = _filas_a_texto(df, MAX_FILAS_TABULARES)
    return {
        "texto": texto,
        "estado": "ok" if texto.strip() else "vacio",
        "filas_totales": total,
        "truncado": truncado,
    }


def extraer_xlsx(ruta: Path) -> dict:
    """Extrae todas las hojas; cada hoja se prefija con su nombre."""
    xl = pd.ExcelFile(ruta)
    bloques = []
    total_filas = 0
    truncado = False
    for hoja in xl.sheet_names:
        df = xl.parse(hoja, dtype=str)
        texto, total, trunc = _filas_a_texto(df, MAX_FILAS_TABULARES)
        total_filas += total
        truncado = truncado or trunc
        if texto.strip():
            bloques.append(f"Hoja: {hoja}\n{texto}" if len(xl.sheet_names) > 1 else texto)
    texto = "\n\n".join(bloques)
    return {
        "texto": texto,
        "estado": "ok" if texto.strip() else "vacio",
        "filas_totales": total_filas,
        "truncado": truncado,
    }


def extraer_txt(ruta: Path) -> dict:
    """Lee un .txt plano y elimina la cabecera de scraping si existe."""
    import chardet

    datos = ruta.read_bytes()
    enc = (chardet.detect(datos[:200_000]) or {}).get("encoding") or "utf-8"
    try:
        texto = datos.decode(enc, errors="replace")
    except LookupError:
        texto = datos.decode("utf-8", errors="replace")

    # Cabecera "SOURCE: ... / SCRAPED: ... / ====" que anade el scraper.
    meta: dict = {}
    m = re.match(
        r"\s*SOURCE:\s*(\S+)\s*\n\s*SCRAPED:\s*(\S+)\s*\n=+\s*\n", texto, re.IGNORECASE
    )
    if m:
        meta["url"] = m.group(1)
        meta["fecha_scraping"] = m.group(2)
        texto = texto[m.end():]

    lineas = texto.split("\n")

    # Menus de navegacion: lineas cortas repetidas consecutivamente.
    sin_repetidas, previa = [], None
    for l in lineas:
        s = l.strip()
        if s and s == previa and len(s) < 60:
            continue  # entrada de menu duplicada
        sin_repetidas.append(l)
        previa = s

    # Boilerplate de cabecera de sitio web: los volcados de pagina empiezan con
    # decenas de entradas de menu (lineas de pocas palabras, sin puntuacion
    # final) antes del contenido real. Se descarta ese bloque inicial hasta la
    # primera linea con aspecto de prosa, y solo si lo descartado son en su
    # mayoria lineas cortas: asi un .txt que ya empieza en prosa queda intacto.
    def es_prosa(linea: str) -> bool:
        s = linea.strip()
        return len(s.split()) >= 12 and s.endswith((".", "!", "?", ":", ";"))

    idx = next((i for i, l in enumerate(sin_repetidas) if es_prosa(l)), None)
    if idx:
        encabezado = [l for l in sin_repetidas[:idx] if l.strip()]
        cortas = sum(1 for l in encabezado if len(l.split()) < 8)
        if encabezado and cortas / len(encabezado) >= 0.8:
            meta["lineas_boilerplate_eliminadas"] = len(encabezado)
            sin_repetidas = sin_repetidas[idx:]

    texto = "\n".join(sin_repetidas)
    return {"texto": texto, "estado": "ok" if texto.strip() else "vacio", "meta": meta}


def extraer_imagen(ruta: Path) -> dict:
    """Aplica OCR sobre imágenes con texto potencialmente relevante."""
    try:
        import pytesseract
        from PIL import Image
        if ruta.suffix.lower() == ".avif":
            try:
                import pillow_avif  # noqa: F401
            except ImportError:
                pass
    except ImportError:
        return {"texto": "", "estado": "pendiente_ocr", "detalle": "instala pytesseract y pillow"}

    try:
        pytesseract = _configurar_tesseract()
        pytesseract.get_tesseract_version()
    except Exception:
        return {"texto": "", "estado": "pendiente_ocr", "detalle": f"Tesseract no disponible en {TESSERACT_CMD}"}

    try:
        img = Image.open(ruta)
        if img.mode not in ("RGB", "L"):
            convertida = img.convert("RGB")
            img.close()
            img = convertida
        texto = pytesseract.image_to_string(img, lang=OCR_LANG)
    except Exception as exc:
        return {"texto": "", "estado": "error", "detalle": f"OCR imagen: {exc}"[:300]}
    finally:
        try:
            img.close()
        except Exception:
            pass

    util = len(texto.split()) >= OCR_MIN_WORDS
    return {
        "texto": texto if util else "",
        "estado": "ok" if util else "sin_texto_relevante",
    }

def extraer_pbf(ruta: Path, claves_vistas: set | None = None) -> dict:
    """Decodifica un vector tile (Mapbox Vector Tile) y vuelca sus atributos.

    Segun la especificacion, el mismo elemento se repite en varios niveles de
    zoom; se conserva una sola version de cada entidad. `claves_vistas` acumula
    las entidades ya emitidas por tiles anteriores.
    """
    import mapbox_vector_tile as mvt

    if claves_vistas is None:
        claves_vistas = set()

    datos = mvt.decode(ruta.read_bytes())
    lineas = []
    n_total = 0
    n_nuevas = 0
    for nombre_capa, capa in datos.items():
        for feat in capa.get("features", []):
            props = feat.get("properties") or {}
            n_total += 1
            identificador = props.get("b_ID_concatenated") or props.get("fid")
            if identificador is not None and str(identificador).strip():
                clave = (nombre_capa, str(identificador).strip())
                if clave in claves_vistas:
                    continue
                claves_vistas.add(clave)
            else:
                # Sin identificador estable NO se colapsan todas las features de
                # una capa bajo la misma clave. Se conserva la feature para evitar
                # perdida silenciosa de datos; una deduplicacion semantica posterior
                # puede hacerse durante la construccion del indice.
                clave = None
            n_nuevas += 1
            pares = []
            for k, v in props.items():
                if v is None:
                    continue
                s = str(v).strip()
                if not s or s.lower() in ("nan", "none"):
                    continue
                pares.append(f"{k}: {s}")
            if pares:
                lineas.append(" | ".join(pares))

    texto = "\n".join(lineas)
    if not texto.strip():
        estado = "duplicado_pbf" if n_total else "vacio"
    else:
        estado = "ok"
    return {
        "texto": texto,
        "estado": estado,
        "features_totales": n_total,
        "features_nuevas": n_nuevas,
    }


# --------------------------------------------------------------------------
# Idioma
# --------------------------------------------------------------------------

_identificador_idioma = None


def detectar_idioma(texto: str) -> str:
    """Idioma predominante del documento (Seccion 2.2)."""
    global _identificador_idioma
    muestra = texto[:MUESTRA_IDIOMA].strip()
    if len(muestra) < 30:
        return "desconocido"
    try:
        import langid

        if _identificador_idioma is None:
            # langid es deterministico: mismo texto -> mismo resultado siempre.
            _identificador_idioma = langid.langid.LanguageIdentifier.from_modelstring(
                langid.langid.model, norm_probs=True
            )
        codigo, prob = _identificador_idioma.classify(muestra)
        # Los archivos tabulares (listas de autores, codigos, banderas
        # booleanas) no tienen prosa suficiente y langid devuelve etiquetas
        # arbitrarias con baja confianza. Se prefiere "desconocido" antes que
        # un idioma inventado que el equipo pudiera usar como filtro.
        return codigo if prob >= 0.60 else "desconocido"
    except Exception:
        return "desconocido"


# --------------------------------------------------------------------------
# Procesamiento de un documento
# --------------------------------------------------------------------------

MAPA_FENOMENO = {"F1": 1, "F2": 2, "F3": 3}


def procesar_documento(registro: dict) -> dict:
    """Procesa un archivo del inventario. Nunca lanza excepcion.

    Cualquier fallo se captura y se devuelve como estado='error' con el motivo,
    para que un archivo corrupto no interrumpa la corrida completa.
    """
    ruta = Path(registro["ruta_absoluta"])
    tipo = registro["tipo_adl"]
    ext = ruta.suffix.lower().lstrip(".")

    salida = {
        "doc_id": registro["doc_id"],
        "fuente": registro["fuente"],
        "ruta_relativa": registro["ruta_relativa"],
        "formato": ext,
        "tipo_adl": tipo,
        "fenomeno": registro["fenomeno"],
        "observatorio": registro["observatorio"],
        "codigo_observatorio": registro["codigo_observatorio"],
        "titulo": None,
        "url": None,
        "fecha": None,
        "autores": None,
        "etiquetas": None,
        "idioma": "desconocido",
        "texto": "",
        "n_caracteres": 0,
        "n_palabras": 0,
        "n_paginas": None,
        "filas_totales": None,
        "truncado": False,
        "estado": "error",
        "detalle": None,
        "subtipo_json": None,
        "bytes_archivo": None,
        "hash_sha1": None,
        "segundos": 0.0,
    }

    inicio = time.time()
    try:
        salida["bytes_archivo"] = ruta.stat().st_size

        if tipo == "PDF" or ext == "pdf":
            res = extraer_pdf(ruta)
            salida["n_paginas"] = res.get("n_paginas")
            for k, v in (res.get("meta") or {}).items():
                if k in salida:
                    salida[k] = v
        elif tipo == "JSON" or ext == "json":
            res = extraer_json(ruta)
            salida["subtipo_json"] = res.get("subtipo")
            for k, v in (res.get("meta") or {}).items():
                if k in salida:
                    salida[k] = v
        elif tipo == "CSV" or ext == "csv":
            res = extraer_csv(ruta)
        elif tipo == "Excel" or ext in ("xlsx", "xls"):
            res = extraer_xlsx(ruta)
        elif tipo == "Texto" or ext == "txt":
            res = extraer_txt(ruta)
            for k, v in (res.get("meta") or {}).items():
                if k in salida:
                    salida[k] = v
        elif ext in ("jpg", "jpeg", "png", "avif", "webp", "gif", "tif", "tiff"):
            res = extraer_imagen(ruta)
        elif ext == "pbf":
            # La deduplicacion global entre tiles se resuelve en una segunda
            # pasada secuencial (ver procesar_pbf_secuencial).
            res = {"texto": "", "estado": "pendiente_pbf"}
        else:
            res = {"texto": "", "estado": "formato_no_soportado", "detalle": ext}

        texto = limpiar_texto(res.get("texto", ""))
        salida["texto"] = texto
        salida["n_caracteres"] = len(texto)
        salida["n_palabras"] = len(texto.split())
        salida["estado"] = res.get("estado", "ok")
        if res.get("detalle"):
            salida["detalle"] = res["detalle"]
        elif res.get("metodo_extraccion") and res.get("metodo_extraccion") != "pdfplumber":
            salida["detalle"] = f"metodo={res['metodo_extraccion']}"
        if res.get("filas_totales") is not None:
            salida["filas_totales"] = res["filas_totales"]
        if res.get("truncado"):
            salida["truncado"] = True
        if texto:
            salida["idioma"] = detectar_idioma(texto)
            salida["hash_sha1"] = hashlib.sha1(texto.encode("utf-8")).hexdigest()
        if salida["estado"] == "ok" and not texto:
            salida["estado"] = "vacio"

    except Exception as exc:
        salida["estado"] = "error"
        salida["detalle"] = f"{type(exc).__name__}: {exc}"[:300]

    salida["segundos"] = round(time.time() - inicio, 2)
    return salida


# --------------------------------------------------------------------------
# Escritura incremental y reanudacion
# --------------------------------------------------------------------------


def limite_memoria_worker(gib: float) -> None:
    """Inicializador de cada worker: le pone un techo de memoria virtual.

    Sin este techo un solo archivo patologico puede crecer hasta agotar la RAM
    de la maquina y provocar que el kernel mate procesos al azar (fue lo que
    ocurrio en la corrida anterior). Con el techo, ese archivo levanta
    MemoryError, se registra como error y la corrida continua.
    """
    import resource

    bytes_max = int(gib * 1024 ** 3)
    try:
        blando, duro = resource.getrlimit(resource.RLIMIT_AS)
        nuevo_duro = duro if duro != resource.RLIM_INFINITY else bytes_max
        resource.setrlimit(resource.RLIMIT_AS, (bytes_max, nuevo_duro))
    except (ValueError, OSError):
        pass  # si el SO no lo permite seguimos: el reciclado de workers ya acota


def sanear_jsonl(ruta_jsonl: Path) -> set[str]:
    """Repara el JSONL parcial y devuelve los doc_id ya procesados.

    Permite reanudar una corrida interrumpida sin repetir trabajo. Un kill puede
    dejar la ultima linea a medias y sin salto final; si nos limitaramos a abrir
    en modo append, el primer resultado nuevo se pegaria a esa linea rota y se
    perderian los dos. Por eso el archivo se reescribe dejando solo las lineas
    completas y con su salto de linea final.
    """
    if not ruta_jsonl.exists():
        return set()

    validas: list[str] = []
    descartadas = 0
    with open(ruta_jsonl, "r", encoding="utf-8") as fh:
        for linea in fh:
            desnuda = linea.strip()
            if not desnuda:
                continue
            try:
                json.loads(desnuda)["doc_id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                descartadas += 1
                continue
            validas.append(desnuda)

    texto_saneado = "".join(l + "\n" for l in validas)
    if descartadas or ruta_jsonl.read_text(encoding="utf-8") != texto_saneado:
        temporal = ruta_jsonl.with_suffix(".jsonl.tmp")
        temporal.write_text(texto_saneado, encoding="utf-8")
        temporal.replace(ruta_jsonl)
        if descartadas:
            log.warning("%d lineas incompletas del JSONL parcial descartadas", descartadas)

    return {json.loads(l)["doc_id"] for l in validas}


def doc_ids_en(ruta_jsonl: Path) -> set[str]:
    """doc_id presentes en el JSONL, sin cargar el contenido en memoria."""
    if not ruta_jsonl.exists():
        return set()
    vistos: set[str] = set()
    with open(ruta_jsonl, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            try:
                vistos.add(json.loads(linea)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return vistos


def doc_ids_sin_memoria(ruta_jsonl: Path) -> set[str]:
    """doc_id que fallaron por falta de memoria y merecen un reintento aislado.

    Se distinguen de un PDF corrupto (que fallaria igual con cualquier techo)
    porque el motivo es MemoryError o el fallo de reserva de pdfminer.
    """
    if not ruta_jsonl.exists():
        return set()
    SENALES = ("MemoryError", "Unable to allocate")
    vistos: dict[str, bool] = {}
    with open(ruta_jsonl, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            try:
                fila = json.loads(linea)
            except json.JSONDecodeError:
                continue
            detalle = fila.get("detalle") or ""
            # Vale el ultimo estado de cada doc_id: si un reintento previo ya lo
            # resolvio, no hay que volver a intentarlo.
            vistos[fila.get("doc_id")] = (
                fila.get("estado") == "error" and any(s in detalle for s in SENALES)
            )
    return {d for d, fallo in vistos.items() if fallo}


class EscritorIncremental:
    """Escribe un resultado por linea y lo baja a disco inmediatamente.

    El flush + fsync por documento es lo que garantiza que un kill por memoria
    no se lleve por delante el trabajo ya hecho.
    """

    def __init__(self, ruta: Path):
        self.ruta = ruta
        self.fh = open(ruta, "a", encoding="utf-8")
        self.escritos = 0

    def escribir(self, fila: dict) -> None:
        self.fh.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.escritos += 1

    def close(self) -> None:
        self.fh.close()

    def __enter__(self) -> "EscritorIncremental":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def construir_parquet_desde_jsonl(ruta_jsonl: Path, salida: Path, columnas: list[str]) -> int:
    """Convierte el JSONL a parquet leyendolo por lotes, sin cargarlo entero.

    El corpus completo son cientos de MB de texto; materializarlo en un unico
    DataFrame para escribirlo de golpe volveria a arriesgar la memoria justo al
    final de la corrida, con todo el trabajo ya hecho.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # El esquema se declara explicitamente en vez de deducirlo del primer lote:
    # columnas como `detalle` van vacias en los primeros cientos de documentos y
    # el tipo deducido alli (null) hace fallar cualquier lote posterior que si
    # traiga texto.
    ENTEROS = {"n_caracteres", "n_palabras", "n_paginas", "filas_totales",
               "bytes_archivo", "fenomeno"}
    esquema = pa.schema([
        pa.field(c,
                 pa.int64() if c in ENTEROS
                 else pa.bool_() if c == "truncado"
                 else pa.float64() if c == "segundos"
                 else pa.string())
        for c in columnas
    ])

    TAM_LOTE = 200
    escritor = None
    total = 0
    lote: list[dict] = []

    # Un doc_id puede aparecer mas de una vez (los PBF se reprocesan enteros al
    # reanudar). Vale la ultima version escrita, asi que primero se localiza en
    # que linea quedo cada doc_id; solo se guardan indices, no contenido.
    ultima_linea: dict[str, int] = {}
    with open(ruta_jsonl, "r", encoding="utf-8") as fh:
        for i, linea in enumerate(fh):
            linea = linea.strip()
            if not linea:
                continue
            try:
                ultima_linea[json.loads(linea)["doc_id"]] = i
            except (json.JSONDecodeError, KeyError):
                continue
    vigentes = set(ultima_linea.values())
    lineas_totales = sum(1 for _ in open(ruta_jsonl, encoding="utf-8"))
    if lineas_totales > len(vigentes):
        log.info(
            "%d filas repetidas descartadas (se conserva la ultima de cada doc_id)",
            lineas_totales - len(vigentes),
        )

    def volcar(lote: list[dict]) -> None:
        nonlocal escritor, total
        if not lote:
            return
        columnas_datos = {
            c: [fila.get(c) for fila in lote] for c in columnas
        }
        tabla = pa.table(
            {c: pa.array(v, type=esquema.field(c).type) for c, v in columnas_datos.items()},
            schema=esquema,
        )
        if escritor is None:
            escritor = pq.ParquetWriter(salida, esquema, compression="snappy")
        escritor.write_table(tabla)
        total += len(lote)

    try:
        with open(ruta_jsonl, "r", encoding="utf-8") as fh:
            for i, linea in enumerate(fh):
                linea = linea.strip()
                if not linea or i not in vigentes:
                    continue
                try:
                    lote.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
                if len(lote) >= TAM_LOTE:
                    volcar(lote)
                    lote = []
        volcar(lote)
    finally:
        if escritor is not None:
            escritor.close()
    return total


def procesar_pbf_secuencial(registros: list[dict]):
    """Procesa los vector tiles en serie para poder deduplicar entre archivos.

    Se ordenan por ruta para que la asignacion de cada entidad al primer tile
    que la contiene sea estable y reproducible entre corridas. Es un generador:
    va entregando cada resultado para que main() lo escriba al vuelo.

    OJO: la deduplicacion es incremental, asi que una reanudacion parcial de los
    PBF daria un reparto distinto de entidades. Por eso main() los reprocesa
    todos o ninguno.
    """
    claves: set = set()
    for reg in sorted(registros, key=lambda r: r["ruta_relativa"]):
        base = procesar_documento(reg)
        try:
            res = extraer_pbf(Path(reg["ruta_absoluta"]), claves)
            texto = limpiar_texto(res["texto"])
            base["texto"] = texto
            base["n_caracteres"] = len(texto)
            base["n_palabras"] = len(texto.split())
            base["estado"] = res["estado"]
            base["filas_totales"] = res.get("features_totales")
            base["detalle"] = (
                f"features={res.get('features_totales')} nuevas={res.get('features_nuevas')}"
            )
            if texto:
                base["idioma"] = detectar_idioma(texto)
                base["hash_sha1"] = hashlib.sha1(texto.encode("utf-8")).hexdigest()
        except Exception as exc:
            base["estado"] = "error"
            base["detalle"] = f"{type(exc).__name__}: {exc}"[:300]
        yield base


# --------------------------------------------------------------------------
# Inventario
# --------------------------------------------------------------------------


def cargar_inventario(formatos: list[str] | None, limite: int | None) -> list[dict]:
    """Lee el inventario oficial de ADL, que es la fuente de verdad.

    El campo `fenomeno` se toma del inventario y no de los nombres de carpeta,
    igual que `DOC_ID`, que ya viene asignado por la organizacion.
    """
    df = pd.read_excel(INVENTARIO, sheet_name=HOJA_INVENTARIO)
    df["ruta_relativa"] = (
        df["Carpeta"].astype(str).str.rstrip("/") + "/" + df["Nombre estandarizado"].astype(str)
    )

    if formatos:
        df = df[df["Tipo"].isin(formatos)]
    if limite:
        # Muestra estratificada: algunos documentos de cada tipo.
        df = df.groupby("Tipo", group_keys=False).head(max(1, limite // max(df["Tipo"].nunique(), 1)))
        df = df.head(limite)

    registros = []
    for _, fila in df.iterrows():
        registros.append(
            {
                "doc_id": str(fila["DOC_ID"]),
                "fuente": str(fila["Nombre estandarizado"]),
                "ruta_relativa": fila["ruta_relativa"],
                "ruta_absoluta": str(CORPUS_DIR / fila["ruta_relativa"]),
                "tipo_adl": str(fila["Tipo"]),
                "fenomeno": MAPA_FENOMENO.get(str(fila["Fenómeno"]).strip()),
                "observatorio": str(fila["Observatorio"]),
                "codigo_observatorio": str(fila["Código Observatorio"]),
            }
        )
    return registros


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocesamiento del corpus CODEFEST 2026")
    ap.add_argument("--limite", type=int, default=None, help="procesar solo N documentos")
    ap.add_argument("--formatos", nargs="*", default=None,
                    help="filtrar por tipo del inventario: PDF JSON CSV Excel Imagen Texto Otro")
    ap.add_argument("--procesos", type=int, default=4,
                    help="workers para archivos normales (cada uno puede llegar a ~2 GB)")
    ap.add_argument("--procesos-grandes", type=int, default=2,
                    help="workers para archivos grandes; se reciclan tras cada archivo")
    ap.add_argument("--umbral-grande", type=float, default=15.0,
                    help="MB a partir de los cuales un archivo va a la pasada de grandes")
    ap.add_argument("--tareas-por-worker", type=int, default=10,
                    help="documentos tras los cuales se recicla el worker (0 = nunca)")
    # El techo es un ultimo recurso contra un archivo desbocado, no el mecanismo
    # principal de control (ese es reciclar workers). Ajustado demasiado bajo, el
    # interprete puede abortar en mitad de una reserva en vez de levantar
    # MemoryError, que es justo lo que rompio la primera corrida de esta version.
    ap.add_argument("--memoria-worker", type=float, default=4.5,
                    help="techo de memoria virtual por worker, en GiB")
    ap.add_argument("--memoria-worker-grande", type=float, default=7.0,
                    help="techo de memoria por worker en la pasada de archivos grandes")
    ap.add_argument("--cada", type=int, default=25, help="cada cuantos documentos se reporta avance")
    ap.add_argument("--reiniciar", action="store_true",
                    help="ignora el JSONL parcial y reprocesa el corpus completo")
    ap.add_argument("--salida", type=Path, default=SALIDA_DIR)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    if not INVENTARIO.exists():
        log.error("no se encuentra el inventario: %s", INVENTARIO)
        return 1

    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_jsonl = args.salida / "documentos.jsonl"

    registros = cargar_inventario(args.formatos, args.limite)
    log.info("inventario cargado: %d documentos", len(registros))

    # ---- Reanudacion ----
    if args.reiniciar and ruta_jsonl.exists():
        respaldo = ruta_jsonl.with_suffix(".jsonl.bak")
        ruta_jsonl.replace(respaldo)
        log.info("--reiniciar: JSONL anterior movido a %s", respaldo.name)

    ya_hechos = sanear_jsonl(ruta_jsonl)
    if ya_hechos:
        log.info("reanudando: %d documentos ya estaban en %s", len(ya_hechos), ruta_jsonl.name)

    pendientes = [r for r in registros if r["doc_id"] not in ya_hechos]
    if not pendientes:
        log.info("no queda nada pendiente; solo se reconstruyen los agregados")

    es_pbf = lambda r: r["ruta_relativa"].lower().endswith(".pbf")
    pbf = [r for r in pendientes if es_pbf(r)]
    # La deduplicacion entre tiles es incremental: si quedo a medias hay que
    # rehacer los PBF completos, no solo los que faltan.
    if pbf and any(es_pbf(r) and r["doc_id"] in ya_hechos for r in registros):
        pbf = [r for r in registros if es_pbf(r)]
        log.info("los PBF se reprocesan completos (%d) por la deduplicacion entre tiles", len(pbf))

    umbral_bytes = args.umbral_grande * 1024 ** 2

    def tam(reg: dict) -> int:
        try:
            return Path(reg["ruta_absoluta"]).stat().st_size
        except OSError:
            return 0

    no_pbf = [r for r in pendientes if not es_pbf(r)]
    grandes = [r for r in no_pbf if tam(r) >= umbral_bytes]
    normales = [r for r in no_pbf if tam(r) < umbral_bytes]
    # Los mas pesados primero: si algo va a reventar, que reviente temprano y no
    # con la corrida casi terminada.
    grandes.sort(key=tam, reverse=True)

    log.info(
        "pendientes: %d normales (<%.0f MB), %d grandes, %d PBF",
        len(normales), args.umbral_grande, len(grandes), len(pbf),
    )

    inicio = time.time()
    contadores: dict[str, int] = {}
    hechos = 0
    total_pendiente = len(normales) + len(grandes) + len(pbf)

    def registrar(escritor: EscritorIncremental, fila: dict, etapa: str) -> None:
        nonlocal hechos
        escritor.escribir(fila)
        hechos += 1
        contadores[fila["estado"]] = contadores.get(fila["estado"], 0) + 1
        if hechos % args.cada == 0 or hechos == total_pendiente:
            tasa = hechos / max(time.time() - inicio, 1e-9)
            log.info(
                "  [%s] %d/%d (%.1f docs/s) ultimo=%s",
                etapa, hechos, total_pendiente, tasa, fila["doc_id"],
            )

    def fila_de_error(reg: dict, motivo: str) -> dict:
        return {
            "doc_id": reg["doc_id"],
            "fuente": reg["fuente"],
            "ruta_relativa": reg["ruta_relativa"],
            "formato": Path(reg["ruta_relativa"]).suffix.lower().lstrip("."),
            "tipo_adl": reg["tipo_adl"],
            "fenomeno": reg["fenomeno"],
            "observatorio": reg["observatorio"],
            "codigo_observatorio": reg["codigo_observatorio"],
            "texto": "",
            "n_caracteres": 0,
            "n_palabras": 0,
            "estado": "error",
            "detalle": motivo[:300],
            "idioma": "desconocido",
            "truncado": False,
        }

    def ejecutar_bloque(lote, etapa, workers, mem_gib, tareas_por_worker) -> list[dict]:
        """Procesa `lote` escribiendo cada resultado al vuelo.

        Los futures se envian con una ventana deslizante en vez de encolar los
        1800 de golpe: asi el proceso padre nunca sostiene mas de unos pocos
        resultados en memoria.

        Devuelve los registros que quedaron sin procesar. Si un worker muere de
        golpe, el pool entero queda inservible y hay que rehacerlo: en ese caso
        se devuelve lo que estaba en vuelo mas lo que no llego a despacharse.
        """
        kwargs = {
            "max_workers": workers,
            "initializer": limite_memoria_worker,
            "initargs": (mem_gib,),
        }
        if tareas_por_worker:
            kwargs["max_tasks_per_child"] = tareas_por_worker

        ventana = max(1, workers * 2)
        en_vuelo: dict = {}
        perdidos: list[dict] = []
        i = 0
        try:
            with ProcessPoolExecutor(**kwargs) as pool:
                while i < len(lote) or en_vuelo:
                    while i < len(lote) and len(en_vuelo) < ventana:
                        en_vuelo[pool.submit(procesar_documento, lote[i])] = lote[i]
                        i += 1
                    terminados, _ = wait(list(en_vuelo), return_when=FIRST_COMPLETED)
                    for fut in terminados:
                        reg_hecho = en_vuelo.pop(fut)
                        try:
                            fila = resultado_de(fut, reg_hecho)
                        except BrokenProcessPool:
                            # Ya salio de `en_vuelo`: sin esto el documento no
                            # volveria a intentarse y desapareceria de la salida.
                            perdidos.append(reg_hecho)
                            raise
                        registrar(escritor, fila, etapa)
        except BrokenProcessPool:
            return perdidos + list(en_vuelo.values()) + lote[i:]
        return []

    def correr_pasada(lote, etapa, workers, mem_gib, tareas_por_worker):
        """Ejecuta un grupo de documentos, rehaciendo el pool si se rompe.

        Cuando un worker muere de golpe no hay forma de saber cual de los
        documentos en vuelo lo mato, asi que los supervivientes se reintentan de
        a uno (un worker, un documento por proceso). Ese modo aislado si
        identifica al culpable, que se anota como error para que la corrida
        pueda continuar en vez de quedarse trabada en el.
        """
        if not lote:
            return
        log.info(
            "%s: %d documentos, %d workers, reciclado cada %s tareas, techo %.1f GiB",
            etapa, len(lote), workers, tareas_por_worker or "inf", mem_gib,
        )
        por_hacer = list(lote)
        aislado = False
        while por_hacer:
            if aislado:
                sobrantes = ejecutar_bloque(por_hacer[:1], etapa, 1, mem_gib, 1)
                if sobrantes:
                    culpable = sobrantes[0]
                    log.error(
                        "%s tumba al worker incluso aislado; se anota como error",
                        culpable["fuente"],
                    )
                    registrar(
                        escritor,
                        fila_de_error(culpable, "el worker murio procesando este archivo"),
                        etapa,
                    )
                por_hacer = por_hacer[1:]
                # Solo el primero es sospechoso: el resto vuelve al modo normal.
                aislado = False
                continue

            sobrantes = ejecutar_bloque(por_hacer, etapa, workers, mem_gib, tareas_por_worker)
            if not sobrantes:
                return
            log.warning(
                "pool roto en '%s': %d documentos se reintentan aislados",
                etapa, len(sobrantes),
            )
            por_hacer = sobrantes
            aislado = True

    def resultado_de(fut, reg):
        try:
            return fut.result()
        except BrokenProcessPool:
            raise  # lo maneja correr_pasada rehaciendo el pool
        except Exception as exc:
            # Fallo aislado del worker: se anota el documento y la corrida sigue.
            log.warning("worker fallo en %s: %s: %s", reg["fuente"], type(exc).__name__, exc)
            return fila_de_error(reg, f"worker: {type(exc).__name__}: {exc}")

    with EscritorIncremental(ruta_jsonl) as escritor:
        correr_pasada(normales, "normales", args.procesos, args.memoria_worker,
                      args.tareas_por_worker)
        # Un archivo grande puede dejar >1 GB retenido en el worker aunque se
        # cierre bien; con un worker nuevo por archivo esa memoria vuelve al SO.
        correr_pasada(grandes, "grandes", args.procesos_grandes, args.memoria_worker_grande, 1)

        if pbf:
            log.info("PBF: %d vector tiles (secuencial, con deduplicacion)", len(pbf))
            for fila in procesar_pbf_secuencial(pbf):
                registrar(escritor, fila, "pbf")

        # El tamano del archivo no predice cuanta memoria pide al parsearse: hay
        # PDF de 5 MB que se expanden a varios GB. Los que chocaron contra el
        # techo se reintentan solos y con mas margen; las filas nuevas pisan a
        # las viejas al construir el parquet (gana la ultima de cada doc_id).
        por_doc = {r["doc_id"]: r for r in registros}
        sin_memoria = [
            por_doc[d] for d in doc_ids_sin_memoria(ruta_jsonl) if d in por_doc
        ]
        if sin_memoria:
            log.info(
                "%d documentos se quedaron sin memoria; se reintentan aislados con %.1f GiB",
                len(sin_memoria), args.memoria_worker_grande,
            )
            correr_pasada(sin_memoria, "reintento-memoria", 1,
                          args.memoria_worker_grande, 1)

        # Red de seguridad: ningun documento del inventario puede quedar fuera de
        # la salida sin dejar rastro. Si algo se perdio (un pool roto en mal
        # momento), se reintenta aislado y, si vuelve a fallar, se anota.
        escritos = ya_hechos | doc_ids_en(ruta_jsonl)
        faltantes = [r for r in registros if r["doc_id"] not in escritos]
        if faltantes:
            log.warning("faltan %d documentos tras las pasadas; se reintentan aislados",
                        len(faltantes))
            correr_pasada(faltantes, "rescate", 1, args.memoria_worker_grande, 1)
            escritos = ya_hechos | doc_ids_en(ruta_jsonl)
            perdidos = [r for r in registros if r["doc_id"] not in escritos]
            for reg in perdidos:
                registrar(escritor, fila_de_error(reg, "no se pudo procesar en ninguna pasada"),
                          "rescate")

    log.info("extraccion terminada: %d documentos escritos en %s", hechos, ruta_jsonl.name)

    # ---- Agregados finales, leyendo el JSONL por lotes ----
    columnas = [
        "doc_id", "fuente", "ruta_relativa", "formato", "tipo_adl", "fenomeno",
        "observatorio", "codigo_observatorio", "titulo", "url", "fecha", "autores",
        "etiquetas", "idioma", "texto", "n_caracteres", "n_palabras", "n_paginas",
        "filas_totales", "truncado", "estado", "detalle", "subtipo_json",
        "bytes_archivo", "hash_sha1", "segundos",
    ]
    ruta_parquet = args.salida / "documentos.parquet"
    n_parquet = construir_parquet_desde_jsonl(ruta_jsonl, ruta_parquet, columnas)
    log.info("parquet escrito: %d filas -> %s", n_parquet, ruta_parquet.name)

    # La metadata sin la columna de texto sí cabe holgadamente en memoria.
    meta = pd.read_parquet(ruta_parquet, columns=[c for c in columnas if c != "texto"])
    meta = meta.sort_values("doc_id").reset_index(drop=True)
    meta.to_csv(args.salida / "documentos_metadata.csv", index=False, encoding="utf-8")
    meta[~meta["estado"].isin(["ok"])].to_csv(
        args.salida / "errores.csv", index=False, encoding="utf-8"
    )

    total = len(meta)
    ok = int((meta["estado"] == "ok").sum())
    transcurrido = time.time() - inicio
    resumen = {
        "documentos_totales": total,
        "documentos_ok": ok,
        "documentos_no_ok": total - ok,
        "palabras_totales": int(meta["n_palabras"].sum()),
        "caracteres_totales": int(meta["n_caracteres"].sum()),
        "segundos": round(transcurrido, 1),
        "por_estado": meta["estado"].value_counts().to_dict(),
        "por_formato": meta["formato"].value_counts().to_dict(),
        "ok_por_formato": meta[meta["estado"] == "ok"]["formato"].value_counts().to_dict(),
        "por_fenomeno": meta["fenomeno"].value_counts().sort_index().to_dict(),
        "por_idioma": meta[meta["estado"] == "ok"]["idioma"].value_counts().head(12).to_dict(),
        "truncados": int(meta["truncado"].fillna(False).sum()),
    }
    with open(args.salida / "resumen.json", "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2, default=str)

    log.info("=" * 64)
    log.info("procesados %d documentos en %.1f s", total, transcurrido)
    log.info("estado: %s", resumen["por_estado"])
    log.info("ok por formato: %s", resumen["ok_por_formato"])
    log.info("palabras extraidas: %s", f"{resumen['palabras_totales']:,}")
    log.info("salida -> %s", ruta_parquet)
    log.info("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
