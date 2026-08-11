"""
Sentence-aware chunking pipeline with paralelization and checkpoint support.

Rewrite of chunking.py:
- Segmenta por oraciones usando pysbd (multilingüe)
- Respeta límites de oración (completitud lingüística obligatoria)
- Paraleliza procesamiento con ProcessPoolExecutor
- Checkpoint JSONL resumible
- Genera chunks.parquet v2 con campos: doc_id, chunk_id, fuente, formato,
  fenomeno (int), posicion, num_tokens, texto
"""
import argparse
import csv
import json
import logging
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Add parent directory to path for module imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chunking_core import crear_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 4000
MIN_CHUNK_CHARS = 200
CHUNK_OVERLAP = 500
DEFAULT_PROCESOS = 6


def get_battery_status() -> Optional[bool]:
    """
    Check battery status on macOS.
    Returns True if on battery and low, False if plugged in or good, None if unsure.
    """
    try:
        # macOS: pmset -g batt gives "drawing from 'Battery Power'" and percentage
        result = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            timeout=2
        )
        output = result.stdout
        if "Battery Power" in output and ("%" in output):
            # Extract percentage (e.g., "76%")
            match = re.search(r"(\d+)%", output)
            if match:
                pct = int(match.group(1))
                return pct < 20  # True if low battery
        return False  # Plugged in or good battery
    except Exception:
        return None  # Unknown


def procesar_documento(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single document row: clean text, create chunks, add metadata.

    This function must be picklable for multiprocessing.
    """
    try:
        doc_id = row["doc_id"]
        texto = row.get("texto", "")
        idioma = row.get("idioma", "en")

        if not isinstance(texto, str) or not texto.strip():
            return []

        # Create chunks using chunking_core
        chunk_dicts = crear_chunks(texto, idioma)

        # Enrich chunks with metadata
        result = []
        for chunk_dict in chunk_dicts:
            registro = {
                "chunk_id": f"{doc_id}_chunk_{chunk_dict['posicion']:05d}",
                "doc_id": doc_id,
                "posicion": chunk_dict["posicion"],
                "texto": chunk_dict["texto"],
                "num_tokens": chunk_dict["num_tokens"],

                # Metadata from original document
                "observatorio": row.get("observatorio"),
                "fenomeno": int(row.get("fenomeno", 0)),  # Ensure int
                "fuente": row.get("fuente"),
                "formato": row.get("formato"),
                "idioma": idioma,
                "titulo": row.get("titulo"),
                "url": row.get("url"),
                "fecha": row.get("fecha"),

                # Statistics (for validation/diagnostics)
                "n_caracteres_chunk": len(chunk_dict["texto"]),
                "n_palabras_chunk": len(chunk_dict["texto"].split()),
            }
            result.append(registro)

        return result

    except Exception as e:
        # Log error but don't crash the entire pool
        log.error(f"Error processing doc {row.get('doc_id', 'UNKNOWN')}: {e}")
        return []


def load_processed_docs(checkpoint_path: Path) -> set:
    """Load set of doc_ids already processed from checkpoint JSONL."""
    processed = set()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        processed.add(obj["doc_id"])
            log.info(f"Loaded {len(processed)} already-processed docs from checkpoint")
        except Exception as e:
            log.warning(f"Could not load checkpoint: {e}")
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Sentence-aware chunking with paralelization"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("salida/documentos.parquet"),
        help="Input parquet file with documents"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("salida/chunks.parquet"),
        help="Output parquet file with chunks"
    )
    parser.add_argument(
        "--procesos",
        type=int,
        default=DEFAULT_PROCESOS,
        help=f"Number of worker processes (default {DEFAULT_PROCESOS})"
    )
    parser.add_argument(
        "--reiniciar",
        action="store_true",
        help="Remove checkpoint and start from scratch"
    )
    parser.add_argument(
        "--check-battery",
        action="store_true",
        help="Warn if battery is low (macOS only)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Check battery
    if args.check_battery:
        battery_status = get_battery_status()
        if battery_status is True:
            log.warning(
                "⚠️  BATTERY LOW: Consider plugging in before running full corpus "
                "(this job is quick but safer on AC power)"
            )

    # Paths
    checkpoint_path = args.output.parent / ".chunking_checkpoint.jsonl"
    errors_path = args.output.parent / "errores_chunking.csv"

    # Clean up old checkpoint if requested
    if args.reiniciar and checkpoint_path.exists():
        backup = checkpoint_path.with_suffix(".jsonl.bak")
        checkpoint_path.rename(backup)
        log.info(f"Previous checkpoint moved to {backup.name}")

    print("=" * 70)
    print("CHUNKING DEL CORPUS (sentence-aware, multilingüe)")
    print("=" * 70)

    # Load input data
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    log.info(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)

    # Filter to valid documents only
    df = df[df["estado"].eq("ok") & df["texto"].notna()].copy()
    log.info(f"Total valid documents: {len(df):,}")

    # Load already-processed docs from checkpoint
    processed_docs = load_processed_docs(checkpoint_path)
    docs_to_process = df[~df["doc_id"].isin(processed_docs)].copy()

    if len(docs_to_process) == 0:
        log.info("All documents already processed (checkpoint complete)")
        if checkpoint_path.exists():
            log.info("Finalizing from checkpoint...")
            # Just convert checkpoint to final parquet
            log.info("Loading checkpoint JSONL")
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            if records:
                chunks_df = pd.DataFrame(records)
                chunks_df.to_parquet(args.output, index=False, engine="pyarrow")
                log.info(f"Finalized: {args.output} ({len(chunks_df):,} chunks)")
        return

    log.info(f"Documents to process: {len(docs_to_process):,}")

    # Process documents with parallelization
    all_chunks = []
    all_errors = []

    with ProcessPoolExecutor(max_workers=args.procesos) as pool:
        # Submit all tasks
        futures = {
            pool.submit(procesar_documento, row.to_dict()): row["doc_id"]
            for _, row in docs_to_process.iterrows()
        }

        # Collect results as they complete
        with tqdm(total=len(futures), desc="Processing documents") as pbar:
            for future in as_completed(futures):
                doc_id = futures[future]
                try:
                    chunk_list = future.result()
                    all_chunks.extend(chunk_list)

                    # Write checkpoint immediately (one document at a time)
                    with open(checkpoint_path, "a", encoding="utf-8") as f:
                        for chunk in chunk_list:
                            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

                except Exception as e:
                    log.error(f"Failed to process {doc_id}: {e}")
                    all_errors.append((doc_id, str(e)))

                pbar.update(1)

    # Save errors
    if all_errors:
        with open(errors_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "error"])
            writer.writerows(all_errors)
        log.warning(f"Errors saved to {errors_path} ({len(all_errors)} docs)")

    # Finalize: read checkpoint and write parquet
    log.info("Finalizing: reading checkpoint and writing parquet")
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        final_records = [json.loads(line) for line in f if line.strip()]

    if final_records:
        chunks_df = pd.DataFrame(final_records)
        chunks_df.to_parquet(args.output, index=False, engine="pyarrow")
        log.info(f"✓ {args.output} written ({len(chunks_df):,} chunks)")

        # Print summary
        print("\n" + "=" * 70)
        print("CHUNKING COMPLETADO")
        print("=" * 70)
        print(f"Documentos procesados: {len(docs_to_process):,}")
        print(f"Chunks generados:      {len(chunks_df):,}")
        if len(chunks_df) > 0:
            print(
                f"Promedio chunks/doc:   {len(chunks_df) / len(docs_to_process):.2f}"
            )
            print(
                f"Caracteres promedio:   {chunks_df['n_caracteres_chunk'].mean():.0f}"
            )
            print(
                f"Palabras promedio:     {chunks_df['n_palabras_chunk'].mean():.0f}"
            )
            print(
                f"Tokens promedio (aprox): {chunks_df['num_tokens'].mean():.0f}"
            )

        print(f"\nSalida: {args.output}")

        # Clean up checkpoint on success
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            log.info("Checkpoint cleaned up")
    else:
        log.warning("No chunks were generated")


if __name__ == "__main__":
    main()
