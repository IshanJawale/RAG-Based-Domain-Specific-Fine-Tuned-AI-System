"""Resumable, bounded-memory ingestion of High Court PDFs into LanceDB.

PDF extraction runs in a bounded worker pool. Embedding and database writes
remain in the parent process, so CUDA is never initialized in a child process.
"""
import argparse
import concurrent.futures
import multiprocessing
import os
import signal
import sqlite3
import time
import uuid
from pathlib import Path

import lancedb
import pyarrow as pa
import pymupdf
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

pymupdf.TOOLS.mupdf_display_errors(False)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
HC_PDF_DIR = BASE_DIR / "law dataset" / "high_court"
DB_PATH = DATA_DIR / "lancedb"
PROGRESS_DB = DATA_DIR / "hc_ingest_progress.db"

# 500 words may exceed BGE-small's 512 token input limit and be truncated.
CHUNK_WORD_LIMIT = 220
BATCH_CHUNK_LIMIT = 2_000
# Keep GPU embedding throughput high. Restarting clears the allocation state
# from any previously orphaned processes; the OOM split/retry remains a guard.
EMBEDDING_BATCH_SIZE = 128
DEFAULT_WORKERS = 6
tokenizer = model = device = None
interrupted = False

schema = pa.schema([
    pa.field("chunk_id", pa.string()), pa.field("source_type", pa.string()),
    pa.field("case_id", pa.string()), pa.field("court", pa.string()),
    pa.field("year", pa.string()), pa.field("chunk_index", pa.int32()),
    pa.field("text", pa.string()), pa.field("vector", pa.list_(pa.float32(), 384)),
])


def handle_interrupt(_sig, _frame):
    """Request a safe single-process stop after the current operation."""
    global interrupted
    if interrupted:
        print("\n[Interrupt] Shutdown is already in progress; please wait...", flush=True)
        return
    interrupted = True
    print("\n[Interrupt] Stopping submissions; completed batches are already saved...", flush=True)


def init_pdf_worker():
    """Only the parent handles Ctrl+C; workers finish or are explicitly stopped."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def init_model():
    global tokenizer, model, device
    print("Loading embedding model (BGE-small)...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", local_files_only=True)
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Embedding on {device}.")


def get_embeddings(texts):
    try:
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(device)
        with torch.inference_mode():
            model_output = model(**encoded)
            embeddings = model_output[0][:, 0]
        vectors = torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy().tolist()
        # Release GPU tensor references before the next batch, which reduces
        # allocator pressure during a million-document run.
        del encoded, model_output, embeddings
        return vectors
    except RuntimeError as exc:
        if device.type != "cuda" or "out of memory" not in str(exc).lower():
            raise
        # A failed forward pass can leave local tensor references alive until
        # this frame returns; drop them before recursively retrying.
        if "encoded" in locals():
            del encoded
        if "model_output" in locals():
            del model_output
        if "embeddings" in locals():
            del embeddings
        torch.cuda.empty_cache()
        if len(texts) == 1:
            raise RuntimeError(
                "CUDA ran out of memory even for one passage. Close other GPU "
                "programs or run with CUDA_VISIBLE_DEVICES disabled."
            ) from exc
        # Retry recursively with smaller batches. This preserves order, so the
        # returned vectors still align with the chunks being assigned below.
        midpoint = len(texts) // 2
        print(f"\n[GPU memory] Retrying {len(texts)} embeddings as "
              f"{midpoint} + {len(texts) - midpoint}.", flush=True)
        return get_embeddings(texts[:midpoint]) + get_embeddings(texts[midpoint:])


def canonical_path(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def source_signature(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def format_duration(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}"


def display_name(path, limit=48):
    """Keep a current-PDF label useful without making tqdm wrap lines."""
    name = Path(path).name
    return name if len(name) <= limit else f"{name[:limit - 3]}..."


def init_progress_db():
    conn = sqlite3.connect(PROGRESS_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_progress (
                filepath TEXT PRIMARY KEY, size_bytes INTEGER, mtime_ns INTEGER,
                status TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file_progress_status ON file_progress(status)")
        # Retain previous progress from the original script instead of duplicating it.
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='completed_files'").fetchone():
            conn.execute("""
                INSERT OR IGNORE INTO file_progress (filepath, status, chunk_count)
                SELECT lower(filepath), 'completed', chunk_count FROM completed_files
            """)
        # v1 copied legacy paths without Windows case normalization.  The
        # initial repair could consequently leave two case-variant rows for
        # one source PDF. Keep the already-normalized row, then normalize any
        # remaining legacy-only rows.
        conn.execute("""
            DELETE FROM file_progress AS legacy
            WHERE legacy.filepath <> lower(legacy.filepath)
              AND EXISTS (
                  SELECT 1 FROM file_progress AS normalized
                  WHERE normalized.filepath = lower(legacy.filepath)
              )
        """)
        conn.execute("UPDATE file_progress SET filepath = lower(filepath) "
                     "WHERE filepath <> lower(filepath)")
        # Repair rows written by the initial bounded-loader revision, whose
        # status and chunk_count parameters were accidentally reversed.
        conn.execute("""
            UPDATE file_progress
            SET chunk_count = CAST(status AS INTEGER), status = 'completed'
            WHERE status GLOB '[0-9]*'
        """)
    return conn


def get_file_state(conn, filepath, size_bytes, mtime_ns):
    row = conn.execute("SELECT size_bytes, mtime_ns, status FROM file_progress WHERE filepath=?",
                       (filepath,)).fetchone()
    if row is None:
        return None
    old_size, old_mtime, status = row
    # Legacy rows have no signature and refer to data already in LanceDB.
    if old_size is None or (old_size == size_bytes and old_mtime == mtime_ns):
        return status
    return None


def set_file_states(conn, files, status, error=None):
    """Persist source-file state for normal completion or write recovery."""
    with conn:
        conn.executemany("""
            INSERT INTO file_progress
                (filepath,size_bytes,mtime_ns,status,chunk_count,last_error,updated_at)
            VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(filepath) DO UPDATE SET
                size_bytes=excluded.size_bytes, mtime_ns=excluded.mtime_ns,
                status=excluded.status, chunk_count=excluded.chunk_count,
                last_error=excluded.last_error, updated_at=CURRENT_TIMESTAMP
        """, [(filepath, size_bytes, mtime_ns, status, chunk_count, error)
               for filepath, size_bytes, mtime_ns, chunk_count in files])


def get_progress_summary(conn):
    """Return durable progress counts without walking the PDF corpus again."""
    counts = {"completed": 0, "skipped": 0, "failed": 0, "writing": 0}
    for status, count in conn.execute(
        "SELECT status, COUNT(*) FROM file_progress GROUP BY status"
    ):
        counts[status] = count
    return counts


def iter_pdf_paths(root):
    """Stream paths: retaining a million pathname strings wastes substantial RAM."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            yield path


def extract_chunks_from_pdf(pdf_path_str):
    """Extract and chunk one PDF in the main process."""
    pdf_path = Path(pdf_path_str)
    try:
        with pymupdf.open(pdf_path) as doc:
            words = []
            for page in doc:
                words.extend(page.get_text().split())
        if len(words) < 50:
            return pdf_path_str, [], None

        year = "unknown"
        if pdf_path.parent.name.isdigit():
            year = pdf_path.parent.name
        elif pdf_path.parent.parent.name.isdigit():
            year = pdf_path.parent.parent.name
        case_id = pdf_path.stem
        chunks = []
        for chunk_index, start in enumerate(range(0, len(words), CHUNK_WORD_LIMIT)):
            text = " ".join(words[start:start + CHUNK_WORD_LIMIT])
            # Stable IDs make a retry after a crash idempotent with merge_insert.
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                      f"high-court:{pdf_path_str}:{chunk_index}"))
            chunks.append({"chunk_id": chunk_id, "source_type": "high_court",
                           "case_id": case_id, "court": "High Court of India",
                           "year": year, "chunk_index": chunk_index,
                           "text": f"[High Court of India - {year}] {case_id}\n\n{text}",
                           "vector": None})
        return pdf_path_str, chunks, None
    except Exception as exc:
        return pdf_path_str, None, f"{type(exc).__name__}: {exc}"


def upsert_records(table, records):
    """Recover a batch whose prior LanceDB write may have completed."""
    try:
        (table.merge_insert("chunk_id").when_matched_update_all()
         .when_not_matched_insert_all().execute(records))
    except AttributeError as exc:
        raise RuntimeError("LanceDB >= 0.17.0 with merge_insert is required for safe resume.") from exc


def append_records(table, records):
    """Fast path for a new batch; each record has a deterministic chunk ID."""
    table.add(records)


def stop_executor(executor, immediate):
    """Reap only this pool's children; never touch unrelated Python processes."""
    if not immediate:
        executor.shutdown(wait=True, cancel_futures=True)
        return

    processes = list(getattr(executor, "_processes", {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 5
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
    executor.shutdown(wait=False, cancel_futures=True)


def embed_chunks(chunks, embedding_batch_size):
    texts = [chunk["text"] for chunk in chunks]
    for start in range(0, len(texts), embedding_batch_size):
        embeddings = get_embeddings(texts[start:start + embedding_batch_size])
        for offset, embedding in enumerate(embeddings):
            chunks[start + offset]["vector"] = embedding


def process_hc_cases(args):
    global interrupted
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HC_PDF_DIR.is_dir():
        raise FileNotFoundError(f"High Court PDF directory does not exist: {HC_PDF_DIR}")
    conn = init_progress_db()
    db = lancedb.connect(DB_PATH)
    # `list_tables()` returns a pagination response in current LanceDB versions,
    # not a plain collection of names.  `exist_ok` is version-stable and avoids
    # a race between checking and creating the table.
    table = db.create_table("high_court", schema=schema, mode="create", exist_ok=True)

    saved = get_progress_summary(conn)
    previously_handled = saved["completed"] + saved["skipped"]
    print("Resume status from SQLite:")
    print(f"  {saved['completed']:,} PDFs embedded and saved")
    print(f"  {saved['skipped']:,} PDFs skipped (too little/no extractable text)")
    print(f"  {saved['failed']:,} PDFs previously failed "
          f"({'will retry' if args.retry_failed else 'will remain skipped; use --retry-failed to retry'})")
    if saved["writing"]:
        print(f"  {saved['writing']:,} PDFs need one crash-recovery upsert")
    print(f"  {previously_handled:,} PDFs already handled in total")
    print("Counting PDFs for an accurate progress percentage and ETA...")
    # This is a separate metadata-only traversal. Show it because a million-file
    # walk can take minutes and otherwise looks like a frozen terminal.
    count_progress = tqdm(desc="Counting PDFs", unit="pdf", dynamic_ncols=True,
                          mininterval=0.5)
    try:
        total_pdfs = 0
        for _ in iter_pdf_paths(HC_PDF_DIR):
            total_pdfs += 1
            count_progress.update(1)
    finally:
        count_progress.close()
    if not total_pdfs:
        print("No High Court PDFs found.")
        conn.close()
        return
    print(f"Found {total_pdfs:,} PDFs. Starting with {args.workers} PDF workers "
          f"and at most {args.inflight} PDFs in flight.")
    records, record_files = [], []
    batch_requires_upsert = False
    errors = skipped = processed = chunks_written = 0
    pending = {}
    paths = iter(iter_pdf_paths(HC_PDF_DIR))
    exhausted = False
    scanned = 0
    resume_path_printed = False
    started_at = time.monotonic()
    progress = tqdm(
        total=total_pdfs,
        desc="High Court PDFs",
        unit="pdf",
        dynamic_ncols=True,
        mininterval=0.5,
        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"),
    )

    def refresh_progress(force=False):
        # Updating terminal text for every PDF is surprisingly expensive at
        # million-file scale. Refresh it periodically instead.
        if force or scanned % 100 == 0:
            progress.set_postfix_str(
                f"already={skipped:,}  in-flight={len(pending)}  failures={errors:,}"
            )

    def flush_records():
        nonlocal records, record_files, chunks_written, batch_requires_upsert
        if records:
            progress.set_postfix_str(f"writing={len(records):,} chunks to LanceDB")
            progress.refresh()
            write_started = time.monotonic()
            # Make the batch recoverable before appending it. If the process is
            # interrupted after `add` commits but before SQLite is updated, the
            # next run sees `writing` and upserts this one batch safely.
            set_file_states(conn, record_files, "writing")
            use_upsert = batch_requires_upsert
            batch_requires_upsert = True
            if use_upsert:
                upsert_records(table, records)
            else:
                append_records(table, records)
            set_file_states(conn, record_files, "completed")
            chunks_written += len(records)
            progress.write(f"LanceDB batch saved: {len(records):,} chunks in "
                           f"{format_duration(time.monotonic() - write_started)}"
                           f" ({'recovery upsert' if use_upsert else 'append'}).")
            records, record_files = [], []
            batch_requires_upsert = False

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, initializer=init_pdf_worker
    )
    try:
        while pending or not exhausted:
            while not interrupted and not exhausted and len(pending) < args.inflight:
                try:
                    path = next(paths)
                    filepath = canonical_path(path)
                    size_bytes, mtime_ns = source_signature(path)
                    scanned += 1
                except StopIteration:
                    exhausted = True
                    break
                except OSError as exc:
                    errors += 1
                    tqdm.write(f"[Scan error] {exc}")
                    continue

                state = get_file_state(conn, filepath, size_bytes, mtime_ns)
                if state in {"completed", "skipped"} or (state == "failed" and not args.retry_failed):
                    skipped += 1
                    progress.update(1)
                    refresh_progress()
                    continue
                if not resume_path_printed:
                    progress.write("Resume point (first eligible PDF found):")
                    progress.write(f"  {filepath}")
                    resume_path_printed = True
                progress.set_postfix_str(f"extracting={display_name(filepath)}")
                progress.refresh()
                future = executor.submit(extract_chunks_from_pdf, filepath)
                pending[future] = (filepath, size_bytes, mtime_ns, state == "writing")
                refresh_progress()

            if interrupted or not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, timeout=0.25,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                filepath, size_bytes, mtime_ns, needs_upsert = pending.pop(future)
                processed += 1
                progress.update(1)
                refresh_progress()
                if interrupted:
                    break
                try:
                    returned_path, chunks, error = future.result()
                    if returned_path != filepath:
                        raise RuntimeError("extractor returned a result for the wrong PDF")
                except Exception as exc:
                    chunks, error = None, f"WorkerFailure: {type(exc).__name__}: {exc}"

                item = (filepath, size_bytes, mtime_ns, 0)
                if error:
                    errors += 1
                    set_file_states(conn, [item], "failed", error)
                    if errors <= 10:
                        progress.write(f"[PDF error] {filepath}: {error}")
                    continue
                if not chunks:
                    set_file_states(conn, [item], "skipped")
                    continue
                if needs_upsert:
                    batch_requires_upsert = True
                progress.set_postfix_str(
                    f"embedding={len(chunks)} chunks; pdf={display_name(filepath)}"
                )
                progress.refresh()
                embed_chunks(chunks, args.embedding_batch_size)
                if interrupted:
                    break
                records.extend(chunks)
                record_files.append((filepath, size_bytes, mtime_ns, len(chunks)))
                if len(records) >= args.batch_chunks:
                    flush_records()
    finally:
        try:
            if interrupted:
                # The progress row is written only after LanceDB succeeds, so
                # discarding this bounded in-memory batch is safe: it will be
                # retried next run. Avoiding an upsert here makes Ctrl+C prompt.
                if records:
                    progress.write(
                        f"Interrupt: leaving {len(records):,} uncommitted chunks for the next run."
                    )
            else:
                flush_records()
        finally:
            stop_executor(executor, immediate=interrupted)
            refresh_progress(force=True)
            progress.close()
            conn.close()

    elapsed = format_duration(time.monotonic() - started_at)
    if interrupted:
        print(f"Stopped safely after {elapsed}. {chunks_written:,} chunks saved; rerun to resume.")
    else:
        print(f"Done. Scanned {scanned:,} PDFs; processed {processed:,}; already handled {skipped:,}; recorded {errors:,} "
              f"failures, and wrote {chunks_written:,} chunks in {elapsed}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--inflight", type=int, default=DEFAULT_WORKERS * 2)
    parser.add_argument("--batch-chunks", type=int, default=BATCH_CHUNK_LIMIT)
    parser.add_argument("--embedding-batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry PDFs whose previous extraction failed")
    args = parser.parse_args()
    if min(args.workers, args.inflight, args.batch_chunks, args.embedding_batch_size) < 1:
        parser.error("all numeric limits must be positive")
    return args


if __name__ == "__main__":
    multiprocessing.freeze_support()
    signal.signal(signal.SIGINT, handle_interrupt)
    init_model()
    process_hc_cases(parse_args())
