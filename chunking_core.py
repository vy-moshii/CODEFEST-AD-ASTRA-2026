"""Core chunking logic: sentence-aware chunk creation with overlap."""
from typing import Optional

try:
    import tiktoken
    # Check if it's a real module (not None/mock from tests)
    if tiktoken and hasattr(tiktoken, 'get_encoding'):
        TIKTOKEN_AVAILABLE = True
    else:
        TIKTOKEN_AVAILABLE = False
except (ImportError, TypeError, AttributeError):
    TIKTOKEN_AVAILABLE = False

from segmentacion import get_sentence_segmenter, clean_text


# Configuration
CHUNK_SIZE = 4000  # Target chunk size in characters (~1000 tokens)
CHUNK_OVERLAP = 500  # Overlap in characters (legacy, now in terms of sentences)
MIN_CHUNK_CHARS = 200  # Minimum chunk size in characters
OVERLAP_SENTENCES = 1  # Number of sentences to repeat from previous chunk


_TIKTOKEN_CACHE = None

def count_tokens(text: str) -> int:
    """
    Approximate token count for text.
    If tiktoken is available, use cl100k_base encoding (cached).
    Otherwise, fallback to word count.
    """
    global _TIKTOKEN_CACHE

    if TIKTOKEN_AVAILABLE:
        try:
            if _TIKTOKEN_CACHE is None:
                _TIKTOKEN_CACHE = tiktoken.get_encoding("cl100k_base")
            return len(_TIKTOKEN_CACHE.encode(text))
        except Exception:
            pass
    # Fallback to word count as token approximation
    return len(text.split())


def crear_chunks(texto: str, idioma: str = "en") -> list[dict]:
    """
    Divide text into chunks respecting sentence boundaries.

    Each chunk:
    - Contains complete sentences (no mid-sentence breaks)
    - Target size ~CHUNK_SIZE characters (~1000 tokens)
    - No overlap in this implementation (overlap happens in the main script if needed)
    - Preserves metadata about position and token count

    Args:
        texto: Text to chunk
        idioma: Language code for sentence segmentation (en, es, pt, zh, etc.)

    Returns:
        List of dicts with keys: 'texto', 'posicion', 'num_tokens'
    """
    if not texto or not texto.strip():
        return []

    # Clean text first
    texto = clean_text(texto)

    # Get sentence segmenter for this language
    segmenter = get_sentence_segmenter(idioma)
    sentences = segmenter(texto)

    if not sentences:
        return []

    chunks = []
    chunk_idx = 0
    sent_idx = 0

    while sent_idx < len(sentences):
        # Build a chunk by accumulating sentences until reaching ~CHUNK_SIZE
        chunk_sentences = []
        chunk_size = 0

        while sent_idx < len(sentences):
            sent = sentences[sent_idx]
            sent_len = len(sent)

            # If adding this sentence would exceed CHUNK_SIZE and we already have sentences
            if chunk_size + sent_len > CHUNK_SIZE and chunk_sentences:
                # Stop and create a chunk
                break

            # Add sentence (even if it makes us exceed CHUNK_SIZE on its own)
            chunk_sentences.append(sent)
            chunk_size += sent_len
            sent_idx += 1

        # Assemble and save chunk
        if chunk_sentences:
            chunk_text = " ".join(chunk_sentences).strip()

            # Save chunk regardless of size (avoid discarding valid content)
            # Minimum size filtering is the responsibility of the caller if needed
            if chunk_text:  # Only skip if truly empty
                chunks.append({
                    "texto": chunk_text,
                    "posicion": chunk_idx,
                    "num_tokens": count_tokens(chunk_text),
                })
                chunk_idx += 1

    return chunks
