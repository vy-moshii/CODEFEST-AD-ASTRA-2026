"""Sentence segmentation by language using pysbd with fallbacks."""
import re
from typing import Callable, Optional

import pysbd


# Detect pysbd's supported languages at import time
_PYSBD_SUPPORTED = set(pysbd.languages.LANGUAGE_CODES.keys())

# Map corpus language codes to pysbd language codes (may differ)
# pysbd uses ISO 639-1 codes, corpus metadata uses similar but not identical
_LANGUAGE_MAP = {
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": None,  # Portuguese not in pysbd, will use fallback
    "ru": "ru",
    "ar": "ar",
    "zh": "zh",
    "ja": "ja",
    "ko": None,  # Korean not in pysbd, will use fallback
    "hi": "hi",
    "nl": "nl",
    "pl": "pl",
    "el": "el",
    "da": "da",
    "fa": "fa",
    "ur": "ur",
    "hy": "hy",
    "am": "am",
    "bg": "bg",
    "kk": "kk",
    "mr": "mr",
    "my": "my",
    # Fallback for languages not in pysbd
    "an": None,  # Aragonese
    "id": None,  # Indonesian
    "wa": None,  # Walloon
    "lt": None,  # Lithuanian
    "af": None,  # Afrikaans
    "oc": None,  # Occitan
    "is": None,  # Icelandic
    "ms": None,  # Malay
    "qu": None,  # Quechua
    "desconocido": None,
}

# Regex segmenters for fallback languages (ASCII + some special punctuation)
_GENERIC_SENTENCE_RE = r"(?<=[.!?…\n])\s+"
_CHINESE_SENTENCE_RE = r"(?<=[。！？\n])\s*"


def _segmenter_pysbd(language_code: str) -> Callable[[str], list[str]]:
    """Return a pysbd sentence segmenter for the given language."""
    seg = pysbd.Segmenter(language=language_code, clean=False)
    return lambda text: [s.strip() for s in seg.segment(text) if s.strip()]


def _segmenter_regex_generic(text: str) -> list[str]:
    """Generic regex-based fallback for ASCII-punctuation languages."""
    if not text or not text.strip():
        return []
    # Split on sentence-ending punctuation followed by whitespace
    parts = re.split(_GENERIC_SENTENCE_RE, text)
    return [p.strip() for p in parts if p.strip()]


def _segmenter_regex_chinese(text: str) -> list[str]:
    """Chinese-specific regex fallback using Chinese punctuation."""
    if not text or not text.strip():
        return []
    parts = re.split(_CHINESE_SENTENCE_RE, text)
    return [p.strip() for p in parts if p.strip()]


def _segmenter_noop(text: str) -> list[str]:
    """Fallback: treat entire text as a single 'sentence' if segmentation fails."""
    return [text] if text and text.strip() else []


def get_sentence_segmenter(idioma: str) -> Callable[[str], list[str]]:
    """
    Return a sentence segmenter function for the given language code.

    Priority:
    1. If idioma is in pysbd's supported languages, use pysbd.
    2. If idioma is Chinese (zh), use Chinese-specific regex.
    3. Otherwise, use generic ASCII-punctuation regex (fallback for pt, ko, ms, qu, etc.).
    4. If all else fails, return the entire text as a single 'sentence'.

    Args:
        idioma: Language code (e.g., 'en', 'es', 'pt', 'zh', 'desconocido').

    Returns:
        A callable that takes text (str) and returns a list of sentences.
    """
    pysbd_lang = _LANGUAGE_MAP.get(idioma)

    # Check if pysbd supports this language
    if pysbd_lang and pysbd_lang in _PYSBD_SUPPORTED:
        return _segmenter_pysbd(pysbd_lang)

    # Chinese special case
    if idioma == "zh":
        return _segmenter_regex_chinese

    # Generic ASCII-punctuation fallback for all other cases
    if idioma in ("pt", "ko", "ms", "qu", "an", "id", "wa", "lt", "af", "oc", "is", "desconocido"):
        return _segmenter_regex_generic

    # Absolute fallback (should not reach here with current corpus, but defensive)
    return _segmenter_noop


def clean_text(text: str) -> str:
    """Normalize text: line breaks, whitespace, multiple newlines."""
    if not isinstance(text, str):
        return ""

    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce multiple newlines to max 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
