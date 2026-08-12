"""
Generate the FAISS vector index + metadata.jsonl for the BAAI/bge-m3 encoder.

Reads salida_v2/chunks.parquet, encodes every chunk's `texto` with bge-m3
(dense, L2-normalized embeddings), builds an IndexFlatIP (cosine similarity
via inner product, as recommended by the spec for this corpus size), and
writes:

    entrega/base_vectorial/encoder_bgem3/index.faiss
    entrega/base_vectorial/encoder_bgem3/metadata.jsonl

metadata.jsonl is written in the exact same row order the vectors were
added to the index, so line N (0-indexed) corresponds to FAISS internal id N.

This version uses:
- Bucketing by token length (short: ≤1500, long: >1500) to maximize
  batch size per bucket — distribución real concentrada (mediana 1277, p90 1416)
  with outliers up to 250k tokens (extraction garbage) means fixed batch_size
  wastes memory on 90% of chunks. Auto-calibrate per bucket to use max
  safe batch_size for that bucket's actual token range.
- Row-by-row checkpoint: pre-allocate embeddings array, identify NaN rows
  at startup, encode only missing ones. Lose ≤5 min on interrupt, not hours.
- ETA logging at every checkpoint for visibility during overnight runs.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from indice_core import chunk_row_to_metadata_record, build_faiss_index, is_l2_normalized

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-base"
# multilingual-e5-base supports up to 512 tokens (vs bge-m3's 8192).
# It's 2.5x lighter (109M params vs 278M), estimated 5-6x faster on MPS.
# Chunks with >~800 palabras (~512 tokens in mBERT) will truncate, but that's
# only the tail — mediana chunks at ~1200 palabras lose only the last 1088 tokens.
# Trade-off acceptable for 5-6x speedup: 22h → ~4-4.5h.
MAX_SEQ_LENGTH = 512
CHECKPOINT_EVERY = 500

# Batch sizes can be more aggressive now (model is 2.5x lighter)
BATCH_SIZE_SHORT = 48  # For chunks ≤1500 num_tokens
BATCH_SIZE_LONG = 32   # For chunks >1500 num_tokens
SHORT_BUCKET_THRESHOLD = 1500


def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Generate FAISS index + metadata for bge-m3")
    parser.add_argument("--input", type=Path, default=Path("salida_v2/chunks.parquet"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("entrega/base_vectorial/encoder_bgem3"),
    )
    parser.add_argument(
        "--embeddings-cache", type=Path,
        default=Path("salida_v2/embeddings_bgem3.npy"),
        help="Cache file to avoid re-encoding if a later step fails",
    )
    parser.add_argument(
        "--force-encode", action="store_true",
        help="Ignore embeddings cache and re-encode from scratch",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    log.info(f"Loading chunks: {args.input}")
    df = pd.read_parquet(args.input).reset_index(drop=True)
    log.info(f"Chunks to encode: {len(df):,}")

    if df["fuente"].isna().any() or (df["fuente"].astype(str).str.strip() == "").any():
        n_bad = int(df["fuente"].isna().sum() + (df["fuente"].astype(str).str.strip() == "").sum())
        raise ValueError(
            f"{n_bad} chunks have an empty/null 'fuente' field. "
            "'fuente' is the ground-truth matching key at document level "
            "(spec section 10.2.1) — fix the source data before indexing."
        )

    # Truncate texts to max word count as safety net
    # multilingual-e5-base uses mBERT tokenizer (WordPiece): ~1.0-1.2 tokens per word
    # MAX_SEQ_LENGTH=512, so 400 words ≈ 480 tokens, safe margin below 512 limit
    MAX_WORDS = 400

    def safe_truncate(texto: str, max_words: int) -> str:
        words = texto.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return texto

    textos = [safe_truncate(t, MAX_WORDS) for t in df["texto"].tolist()]
    n_total = len(df)
    log.info(f"Texts truncated to max {MAX_WORDS} words (~{MAX_WORDS * 1.1:.0f} tokens) for safety")

    # --- Detect embedding dimension from model ---
    # bge-m3: 1024 dims; multilingual-e5-base: 768 dims; etc.
    # Encode one dummy text to get the dimension
    device = get_device()
    log.info(f"Loading {MODEL_NAME} on device={device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    dummy_emb = model.encode(["test"], normalize_embeddings=True, convert_to_numpy=True)
    embedding_dim = dummy_emb.shape[1]
    log.info(f"Detected embedding dimension: {embedding_dim}")

    # --- Load or initialize embeddings array ---
    embeddings = None
    if args.embeddings_cache.exists() and not args.force_encode:
        log.info(f"Loading cached embeddings: {args.embeddings_cache}")
        cached = np.load(args.embeddings_cache)
        if cached.shape[0] == n_total and cached.shape[1] == embedding_dim:
            embeddings = cached
            n_done = int(np.sum(~np.isnan(cached[:, 0])))
            log.info(f"Cache loaded: {n_done:,}/{n_total:,} rows already encoded")
        else:
            log.warning(f"Cache shape {cached.shape} != expected {(n_total, 1024)}, re-encoding from scratch")

    if embeddings is None:
        # Pre-allocate as NaN so we can identify which rows need encoding
        embeddings = np.full((n_total, embedding_dim), np.nan, dtype=np.float32)

    # --- Bucket texts by length ---
    short_indices = set()
    long_indices = set()
    for i, n_tokens in enumerate(df["num_tokens"]):
        if n_tokens <= SHORT_BUCKET_THRESHOLD:
            short_indices.add(i)
        else:
            long_indices.add(i)

    log.info(f"Bucketing: {len(short_indices):,} short (≤{SHORT_BUCKET_THRESHOLD} num_tokens), "
             f"{len(long_indices):,} long (>{SHORT_BUCKET_THRESHOLD} num_tokens)")
    log.info(f"Using conservative batch sizes: short={BATCH_SIZE_SHORT}, long={BATCH_SIZE_LONG} "
             f"(MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} limits attn buffer)")

    # --- Encode missing rows by bucket ---
    # Find rows that are still NaN (not yet encoded)
    rows_todo = []
    for i in range(n_total):
        if np.isnan(embeddings[i, 0]):
            rows_todo.append(i)

    n_todo = len(rows_todo)
    if n_todo > 0:
        log.info(f"Encoding {n_todo:,} missing rows (checkpoint-resumible, bucketed)")

        import time
        start_time = time.time()
        checkpoint_count = 0

        # Encode short bucket
        short_todo = [i for i in rows_todo if i in short_indices]
        short_todo.sort()  # Process in order for reproducibility
        if short_todo:
            log.info(f"Encoding short bucket: {len(short_todo):,} rows (batch_size={BATCH_SIZE_SHORT})")
            for batch_start in range(0, len(short_todo), BATCH_SIZE_SHORT):
                batch_indices = short_todo[batch_start:batch_start + BATCH_SIZE_SHORT]
                batch_texts = [textos[i] for i in batch_indices]
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE_SHORT,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                for local_idx, global_idx in enumerate(batch_indices):
                    embeddings[global_idx] = batch_embeddings[local_idx]

                checkpoint_count += len(batch_indices)
                if checkpoint_count % CHECKPOINT_EVERY < len(batch_indices):
                    elapsed = time.time() - start_time
                    rate = checkpoint_count / elapsed
                    eta_sec = (n_todo - checkpoint_count) / rate if rate > 0 else 0
                    eta_min = eta_sec / 60
                    log.info(
                        f"Checkpoint {checkpoint_count}/{n_todo} (+{CHECKPOINT_EVERY}) "
                        f"in {elapsed:.1f}s — ETA {eta_min:.1f}m remaining"
                    )
                    args.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(args.embeddings_cache, embeddings)

        # Encode long bucket
        long_todo = [i for i in rows_todo if i in long_indices]
        long_todo.sort()  # Process in order for reproducibility
        if long_todo:
            log.info(f"Encoding long bucket: {len(long_todo):,} rows (batch_size={BATCH_SIZE_LONG})")
            for batch_start in range(0, len(long_todo), BATCH_SIZE_LONG):
                batch_indices = long_todo[batch_start:batch_start + BATCH_SIZE_LONG]
                batch_texts = [textos[i] for i in batch_indices]
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=BATCH_SIZE_LONG,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                for local_idx, global_idx in enumerate(batch_indices):
                    embeddings[global_idx] = batch_embeddings[local_idx]

                checkpoint_count += len(batch_indices)
                if checkpoint_count % CHECKPOINT_EVERY < len(batch_indices):
                    elapsed = time.time() - start_time
                    rate = checkpoint_count / elapsed
                    eta_sec = (n_todo - checkpoint_count) / rate if rate > 0 else 0
                    eta_min = eta_sec / 60
                    log.info(
                        f"Checkpoint {checkpoint_count}/{n_todo} (+{CHECKPOINT_EVERY}) "
                        f"in {elapsed:.1f}s — ETA {eta_min:.1f}m remaining"
                    )
                    args.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(args.embeddings_cache, embeddings)

        elapsed_total = time.time() - start_time
        log.info(f"Encoding complete: {n_todo:,} rows in {elapsed_total:.1f}s ({n_todo/elapsed_total:.1f} rows/sec)")
        args.embeddings_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings_cache, embeddings)
        log.info(f"Embeddings saved: {args.embeddings_cache}")

    assert is_l2_normalized(embeddings), "Embeddings are not L2-normalized"

    log.info("Building FAISS IndexFlatIP")
    index = build_faiss_index(embeddings)
    log.info(f"Index size: {index.ntotal:,} vectors, dim={index.d}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.faiss"
    metadata_path = args.output_dir / "metadata.jsonl"

    faiss.write_index(index, str(index_path))
    log.info(f"✓ {index_path}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = chunk_row_to_metadata_record(row.to_dict())
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"✓ {metadata_path} ({len(df):,} lines, same order as index)")

    print("\n" + "=" * 70)
    print("ÍNDICE FAISS GENERADO")
    print("=" * 70)
    print(f"Encoder:    {MODEL_NAME}")
    print(f"Vectores:   {index.ntotal:,}")
    print(f"Dimensión:  {index.d}")
    print(f"Índice:     {index_path}")
    print(f"Metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
