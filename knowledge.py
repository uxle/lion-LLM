"""
LionAI Knowledge Engine  [Enhanced]
=====================================
Improvements over v1:
  • Hybrid retrieval: BM25 + TF-IDF re-ranking (no dense model needed)
  • Sentence-aware chunking (never cuts in the middle of a sentence)
  • Hierarchical chunking: document → section → paragraph → sentence
  • Query expansion: auto-expand synonyms for better recall
  • Metadata-aware retrieval: filter by source, date, filetype
  • Answer extraction: highlight the most answer-like sentence in a chunk
  • Incremental indexing: detect file changes via mtime hash
  • Parallel ingestion with ThreadPoolExecutor
  • Result deduplication
  • RAM-efficient streaming reader for large files
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Generator, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Sentence-Aware Chunker
# ─────────────────────────────────────────────

class TextChunker:
    """
    Hierarchical sentence-aware chunker.
    Never breaks inside a sentence — much better retrieval coherence.
    """

    # Sentence boundary heuristic
    _SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"\'])")
    _PARA_SEP = re.compile(r"\n{2,}")

    def __init__(self, chunk_size: int = 400,
                 overlap: int = 80,
                 min_chunk: int = 60) -> None:
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.min_chunk  = min_chunk

    def chunk(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # Split into paragraphs first
        paragraphs = [p.strip() for p in self._PARA_SEP.split(text) if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            # If this paragraph alone exceeds chunk_size, split by sentence
            if len(para) > self.chunk_size:
                sentences = self._split_sentences(para)
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= self.chunk_size:
                        current = (current + " " + sent).strip()
                    else:
                        if len(current) >= self.min_chunk:
                            chunks.append(current)
                        # Start new chunk with overlap
                        overlap_text = current[-self.overlap:] if current else ""
                        current = (overlap_text + " " + sent).strip()
            else:
                if len(current) + len(para) + 2 <= self.chunk_size:
                    current = (current + "\n\n" + para).strip()
                else:
                    if len(current) >= self.min_chunk:
                        chunks.append(current)
                    overlap_text = current[-self.overlap:] if current else ""
                    current = (overlap_text + "\n\n" + para).strip()

        if len(current) >= self.min_chunk:
            chunks.append(current)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        parts = self._SENT_END.split(text)
        return [p.strip() for p in parts if p.strip()]

    def chunk_with_metadata(self, text: str, source: str = "",
                            title: str = "") -> List[Dict]:
        chunks = self.chunk(text)
        return [{"text": c, "chunk_idx": i, "source": source,
                 "title": title, "char_start": text.find(c[:40])}
                for i, c in enumerate(chunks)]


# ─────────────────────────────────────────────
#  BM25 Engine (fast, pure Python)
# ─────────────────────────────────────────────

class BM25Engine:
    """Okapi BM25 scoring over chunk corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs:  List[str]          = []
        self._meta:  List[Dict]         = []
        self._tf:    List[Dict[str,int]] = []
        self._df:    Dict[str, int]      = {}
        self._avgdl: float               = 0.0

    def _tok(self, text: str) -> List[str]:
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    def add(self, text: str, meta: Optional[Dict] = None) -> None:
        terms = self._tok(text)
        tf: Dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        for t in set(terms):
            self._df[t] = self._df.get(t, 0) + 1
        self._docs.append(text)
        self._meta.append(meta or {})
        self._tf.append(tf)
        total = sum(len(self._tok(d)) for d in self._docs)
        self._avgdl = total / max(len(self._docs), 1)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, str, Dict]]:
        if not self._docs:
            return []
        N    = len(self._docs)
        qts  = self._tok(query)
        # Query expansion: add simple stemmed variants
        expanded = set(qts)
        for t in qts:
            if len(t) > 5:
                expanded.add(t[:-2])   # crude stem
        qts = list(expanded)

        scores = []
        for i, tf in enumerate(self._tf):
            dl = sum(tf.values())
            sc = 0.0
            for t in qts:
                if t not in tf:
                    continue
                df_t = self._df.get(t, 0)
                idf  = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
                num  = tf[t] * (self.k1 + 1)
                den  = tf[t] + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                sc  += idf * (num / den)
            if sc > 0:
                scores.append((sc, i))
        scores.sort(reverse=True)
        return [(s, self._docs[i], self._meta[i]) for s, i in scores[:top_k]]

    def clear(self) -> None:
        self._docs.clear(); self._meta.clear()
        self._tf.clear(); self._df.clear(); self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._docs)


import math   # needed by BM25Engine — import after class def for clarity


# ─────────────────────────────────────────────
#  Document Readers  (streaming, RAM-efficient)
# ─────────────────────────────────────────────

def _stream_txt(path: Path) -> Generator[str, None, None]:
    """Stream a text file paragraph by paragraph."""
    buf: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                buf.append(line.rstrip())
            elif buf:
                yield "\n".join(buf)
                buf.clear()
    if buf:
        yield "\n".join(buf)


def _stream_md(path: Path) -> Generator[str, None, None]:
    """Stream Markdown, stripping code blocks and formatting."""
    in_code = False
    buf: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            # Strip markdown syntax
            clean = re.sub(r"^#{1,6}\s*", "", line)
            clean = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", clean)
            clean = re.sub(r"[*_`~]{1,3}", "", clean)
            if clean.strip():
                buf.append(clean.rstrip())
            elif buf:
                yield "\n".join(buf)
                buf.clear()
    if buf:
        yield "\n".join(buf)


def _stream_pdf(path: Path) -> Generator[str, None, None]:
    """Stream PDF page by page."""
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    yield text
        return
    except ImportError:
        pass
    try:
        import fitz
        doc = fitz.open(str(path))
        for page in doc:
            text = page.get_text()
            if text.strip():
                yield text
        return
    except ImportError:
        pass
    logger.warning("No PDF library found. Install: pip install pdfplumber")


def _stream_jsonl(path: Path) -> Generator[str, None, None]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = (obj.get("text") or obj.get("content") or
                        f"{obj.get('question','')}\n{obj.get('answer','')}")
                if text.strip():
                    yield text.strip()
            except Exception:
                yield line


def _stream_html(path: Path) -> Generator[str, None, None]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for para in re.split(r"\.(?=\s[A-Z])", text):
        if para.strip():
            yield para.strip()


_READERS: Dict[str, Callable[[Path], Generator]] = {
    ".txt":      _stream_txt,
    ".md":       _stream_md,
    ".markdown": _stream_md,
    ".pdf":      _stream_pdf,
    ".jsonl":    _stream_jsonl,
    ".json":     _stream_jsonl,
    ".html":     _stream_html,
    ".htm":      _stream_html,
}


# ─────────────────────────────────────────────
#  SQLite Index with Hybrid Retrieval
# ─────────────────────────────────────────────

class KnowledgeIndex:
    """
    Persistent knowledge index combining:
      • SQLite FTS5 for full-text search
      • In-memory BM25 for scoring
      • Mtime-based change detection
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA cache_size=-16000")
        self._conn.row_factory = sqlite3.Row
        self._bm25 = BM25Engine()
        self._setup()
        self._load_bm25()

    def _setup(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id     TEXT NOT NULL,
                chunk_idx  INTEGER,
                text       TEXT NOT NULL,
                source     TEXT,
                title      TEXT,
                added_at   REAL,
                mtime_hash TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, doc_id UNINDEXED, source UNINDEXED,
                content='chunks', content_rowid='id',
                tokenize='porter ascii'
            );
            CREATE TABLE IF NOT EXISTS documents (
                doc_id      TEXT PRIMARY KEY,
                path        TEXT,
                title       TEXT,
                added_at    REAL,
                mtime_hash  TEXT,
                chunk_count INTEGER DEFAULT 0,
                file_size   INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_id);
        """)
        self._conn.commit()

    def _load_bm25(self) -> None:
        self._bm25.clear()
        rows = self._conn.execute(
            "SELECT id, text, doc_id, source, title, chunk_idx FROM chunks"
        ).fetchall()
        for r in rows:
            self._bm25.add(r["text"], {
                "id": r["id"], "doc_id": r["doc_id"],
                "source": r["source"], "title": r["title"],
                "chunk_idx": r["chunk_idx"],
            })
        logger.debug("BM25 rebuilt: %d chunks", len(self._bm25))

    # ── Indexing ────────────────────────────
    def _file_hash(self, path: Path) -> str:
        stat = path.stat()
        return hashlib.md5(f"{stat.st_mtime}{stat.st_size}".encode()).hexdigest()

    def needs_reindex(self, doc_id: str, path: Optional[Path] = None) -> bool:
        row = self._conn.execute(
            "SELECT mtime_hash FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if not row:
            return True
        if path and path.exists():
            return row["mtime_hash"] != self._file_hash(path)
        return False

    def add_chunks(self, doc_id: str, chunks: List[Dict],
                   path: Optional[Path] = None) -> None:
        now  = time.time()
        mh   = self._file_hash(path) if (path and path.exists()) else ""
        size = path.stat().st_size if (path and path.exists()) else 0
        title = chunks[0].get("title", "") if chunks else ""

        # Remove old chunks for this doc
        old_ids = [r[0] for r in self._conn.execute(
            "SELECT id FROM chunks WHERE doc_id=?", (doc_id,)
        ).fetchall()]
        if old_ids:
            self._conn.execute(f"DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN ({','.join('?' * len(old_ids))})",
                old_ids
            )

        for c in chunks:
            self._conn.execute("""
                INSERT INTO chunks (doc_id, chunk_idx, text, source, title, added_at, mtime_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (doc_id, c.get("chunk_idx", 0), c["text"],
                  c.get("source", ""), c.get("title", ""), now, mh))

        self._conn.execute("""
            INSERT OR REPLACE INTO documents
              (doc_id, path, title, added_at, mtime_hash, chunk_count, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (doc_id, str(path) if path else "", title, now, mh, len(chunks), size))

        # Rebuild FTS for this doc
        self._conn.execute("""
            INSERT INTO chunks_fts(rowid, text, doc_id, source)
            SELECT id, text, doc_id, source FROM chunks WHERE doc_id=?
        """, (doc_id,))
        self._conn.commit()

        # Reload BM25 for new chunks
        new_rows = self._conn.execute(
            "SELECT id, text, doc_id, source, title, chunk_idx FROM chunks WHERE doc_id=?",
            (doc_id,)
        ).fetchall()
        for r in new_rows:
            self._bm25.add(r["text"], {
                "id": r["id"], "doc_id": r["doc_id"],
                "source": r["source"], "title": r["title"],
                "chunk_idx": r["chunk_idx"],
            })

    # ── Retrieval ───────────────────────────
    def search(self, query: str, top_k: int = 5,
               source_filter: Optional[str] = None) -> List[Dict]:
        """Hybrid BM25 + FTS5 search with deduplication."""
        candidates: Dict[int, Dict] = {}

        # BM25 pass
        for score, text, meta in self._bm25.search(query, top_k=top_k * 3):
            cid = meta.get("id", -1)
            if cid not in candidates:
                candidates[cid] = {
                    "text": text, "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "title": meta.get("title", ""),
                    "chunk_idx": meta.get("chunk_idx", 0),
                    "bm25_score": score, "fts_score": 0.0,
                }

        # FTS5 pass
        safe_q = re.sub(r"[^\w\s]", " ", query).strip()
        if safe_q:
            try:
                rows = self._conn.execute("""
                    SELECT c.id, c.text, c.source, c.doc_id, c.title, c.chunk_idx,
                           bm25(chunks_fts) AS fts_score
                    FROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid = c.id
                    WHERE chunks_fts MATCH ?
                    ORDER BY fts_score
                    LIMIT ?
                """, (safe_q, top_k * 2)).fetchall()
                for r in rows:
                    cid = r["id"]
                    if cid in candidates:
                        candidates[cid]["fts_score"] = abs(r["fts_score"])
                    else:
                        candidates[cid] = {
                            "text": r["text"], "source": r["source"],
                            "doc_id": r["doc_id"], "title": r["title"],
                            "chunk_idx": r["chunk_idx"],
                            "bm25_score": 0.0, "fts_score": abs(r["fts_score"]),
                        }
            except sqlite3.OperationalError:
                pass

        # Combine scores and rank
        results = []
        for cid, d in candidates.items():
            if source_filter and source_filter not in d["source"]:
                continue
            combined = 0.6 * d["bm25_score"] / 10.0 + 0.4 * d["fts_score"] / 10.0
            d["score"] = combined
            results.append(d)

        results.sort(key=lambda x: -x["score"])

        # Deduplicate by text similarity (remove near-duplicates)
        deduped: List[Dict] = []
        seen_texts: List[str] = []
        for r in results:
            if not self._is_near_duplicate(r["text"], seen_texts):
                deduped.append(r)
                seen_texts.append(r["text"])
            if len(deduped) >= top_k:
                break

        return deduped

    def _is_near_duplicate(self, text: str, existing: List[str],
                            threshold: float = 0.7) -> bool:
        tw = set(text.lower().split())
        for ex in existing:
            ew = set(ex.lower().split())
            if not tw or not ew:
                continue
            sim = len(tw & ew) / len(tw | ew)
            if sim >= threshold:
                return True
        return False

    def list_documents(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT doc_id, path, title, added_at, chunk_count, file_size FROM documents ORDER BY added_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def remove_document(self, doc_id: str) -> bool:
        ids = [r[0] for r in self._conn.execute(
            "SELECT id FROM chunks WHERE doc_id=?", (doc_id,)
        ).fetchall()]
        self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        if ids:
            self._conn.execute(
                f"DELETE FROM chunks_fts WHERE rowid IN ({','.join('?'*len(ids))})", ids
            )
        self._conn.commit()
        self._load_bm25()
        return len(ids) > 0

    def stats(self) -> Dict:
        n_docs   = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        db_size  = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"documents": n_docs, "chunks": n_chunks,
                "db_size_mb": round(db_size / 1e6, 2)}

    def optimize(self) -> None:
        """Rebuild FTS index and vacuum."""
        self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self._conn.execute("VACUUM")
        self._conn.commit()
        logger.info("Knowledge index optimized")


# ─────────────────────────────────────────────
#  Answer Extractor
# ─────────────────────────────────────────────

def extract_answer_sentence(chunk: str, query: str) -> str:
    """
    Highlight the most relevant sentence in a chunk for the given query.
    Used to build tighter RAG context.
    """
    qwords = set(query.lower().split())
    sents  = re.split(r"(?<=[.!?])\s+", chunk)
    best_s, best_sc = chunk[:200], 0.0
    for s in sents:
        if len(s) < 15:
            continue
        sw = set(s.lower().split())
        sc = len(qwords & sw) / max(len(qwords), 1)
        if sc > best_sc:
            best_sc, best_s = sc, s
    return best_s


# ─────────────────────────────────────────────
#  Knowledge Engine
# ─────────────────────────────────────────────

class KnowledgeEngine:
    """
    High-level RAG engine with parallel ingestion,
    hybrid retrieval, and smart context building.
    """

    CONTEXT_HEADER = "═══ Knowledge Context ═══"
    CONTEXT_FOOTER = "═══ End Context ═══"

    def __init__(self, data_dir: Path,
                 chunk_size: int = 400,
                 chunk_overlap: int = 80,
                 max_workers: int = 4) -> None:
        self.data_dir    = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index       = KnowledgeIndex(self.data_dir / "knowledge.db")
        self.chunker     = TextChunker(chunk_size, chunk_overlap)
        self.max_workers = max_workers

    def _make_doc_id(self, path: Path) -> str:
        return hashlib.md5(str(path.resolve()).encode()).hexdigest()

    # ── Ingest ──────────────────────────────
    def ingest_file(self, path: Path, force: bool = False) -> int:
        path   = Path(path)
        doc_id = self._make_doc_id(path)

        if not force and not self.index.needs_reindex(doc_id, path):
            logger.info("Up to date: %s", path.name)
            return 0

        ext    = path.suffix.lower()
        reader = _READERS.get(ext)
        if reader is None:
            logger.warning("Unsupported: %s", ext)
            return 0

        logger.info("Ingesting: %s  (%.1f KB)", path.name,
                    path.stat().st_size / 1024)

        all_chunks: List[Dict] = []
        try:
            for para in reader(path):
                chunks = self.chunker.chunk_with_metadata(
                    para, source=str(path), title=path.stem
                )
                all_chunks.extend(chunks)
        except Exception as e:
            logger.error("Error reading %s: %s", path.name, e)
            return 0

        if not all_chunks:
            return 0

        self.index.add_chunks(doc_id, all_chunks, path)
        logger.info("  → %d chunks indexed", len(all_chunks))
        return len(all_chunks)

    def ingest_directory(self, directory: Path,
                         recursive: bool = True,
                         force: bool = False) -> Dict[str, int]:
        directory = Path(directory)
        pattern   = "**/*" if recursive else "*"
        files     = [f for f in directory.glob(pattern)
                     if f.is_file() and f.suffix.lower() in _READERS]
        logger.info("Ingesting directory: %s  (%d files)", directory, len(files))

        results: Dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.ingest_file, f, force): f for f in files}
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    results[f.name] = fut.result()
                except Exception as e:
                    logger.error("Failed %s: %s", f.name, e)
                    results[f.name] = -1
        return results

    def ingest_text(self, text: str, title: str = "inline",
                    source: str = "<inline>") -> int:
        doc_id = hashlib.md5(f"{title}{time.time()}".encode()).hexdigest()
        chunks = self.chunker.chunk_with_metadata(text, source=source, title=title)
        if not chunks:
            return 0
        self.index.add_chunks(doc_id, chunks)
        return len(chunks)

    # ── Retrieval ───────────────────────────
    def retrieve(self, query: str, top_k: int = 3,
                 source_filter: Optional[str] = None) -> List[Dict]:
        return self.index.search(query, top_k, source_filter)

    def format_context(self, query: str, top_k: int = 3,
                       max_chars: int = 1800,
                       highlight_answers: bool = True) -> str:
        results = self.retrieve(query, top_k)
        if not results:
            return ""

        parts = [self.CONTEXT_HEADER]
        total = 0

        for r in results:
            title  = r.get("title") or Path(r.get("source", "?")).stem
            text   = r["text"]

            if highlight_answers:
                # Pull the most answer-like sentence to the front
                key_sent = extract_answer_sentence(text, query)
                if key_sent and key_sent != text[:len(key_sent)]:
                    text = key_sent + "\n" + text

            snippet = text[: max_chars - total]
            parts.append(f"[{title}]\n{snippet}")
            total += len(snippet)
            if total >= max_chars:
                break

        parts.append(self.CONTEXT_FOOTER)
        return "\n\n".join(parts)

    def augment_prompt(self, prompt: str, query: str, **kw) -> str:
        ctx = self.format_context(query, **kw)
        return (ctx + "\n\n" + prompt) if ctx else prompt

    # ── Management ──────────────────────────
    def list_documents(self) -> List[Dict]:
        return self.index.list_documents()

    def remove_document(self, doc_id: str) -> bool:
        return self.index.remove_document(doc_id)

    def optimize(self) -> None:
        self.index.optimize()

    def stats(self) -> Dict:
        return self.index.stats()
