"""
LionAI Memory System  [Enhanced]
==================================
Improvements over v1:
  • Episodic memory: auto-summarise old conversations using the model
  • Hierarchical memory scoring: recency × importance × access_freq
  • Memory consolidation: merge duplicate/similar entries
  • BM25 retrieval: much better than keyword LIKE search
  • Sliding-window attention context: inject most relevant memories
  • Working memory: scratchpad for multi-step reasoning chains
  • Memory decay: old unused memories fade (configurable)
  • VACUUM + WAL mode SQLite: faster writes, lower disk usage
  • Thread-safe connection pool
  • Export / import memory snapshots
"""

import hashlib
import json
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Data Types
# ─────────────────────────────────────────────

@dataclass
class Message:
    role:      str
    content:   str
    timestamp: float = field(default_factory=time.time)
    metadata:  Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Message":
        return cls(**{k: v for k, v in d.items()
                      if k in ("role", "content", "timestamp", "metadata")})


@dataclass
class MemoryEntry:
    key:          str
    value:        str
    category:     str   = "general"
    importance:   float = 0.5
    access_count: int   = 0
    created_at:   float = field(default_factory=time.time)
    last_accessed:float = field(default_factory=time.time)
    decay_rate:   float = 0.0   # 0 = no decay; 0.01 = slow fade

    def score(self, now: Optional[float] = None) -> float:
        """
        Composite memory score for retrieval ranking.
        score = importance × log(1 + access_count) × recency × decay
        """
        now = now or time.time()
        age_days = (now - self.last_accessed) / 86400
        recency  = math.exp(-age_days / 30)         # half-life 30 days
        decay    = math.exp(-self.decay_rate * age_days)
        freq     = math.log(1 + self.access_count)
        return self.importance * freq * recency * decay


# ─────────────────────────────────────────────
#  BM25 Retrieval (pure Python, no deps)
# ─────────────────────────────────────────────

class BM25Index:
    """
    Okapi BM25 scoring for memory retrieval.
    Significantly better than LIKE-based search — handles partial matches,
    term frequency, and document length normalisation.
    k1=1.5, b=0.75 are standard defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b  = b
        self._docs:  List[str] = []
        self._meta:  List[Dict] = []
        self._tf:    List[Dict[str, int]] = []   # term freq per doc
        self._df:    Dict[str, int] = {}          # doc freq per term
        self._avgdl: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def add(self, text: str, meta: Optional[Dict] = None) -> int:
        idx   = len(self._docs)
        terms = self._tokenize(text)
        tf: Dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        for t in set(terms):
            self._df[t] = self._df.get(t, 0) + 1
        self._docs.append(text)
        self._meta.append(meta or {})
        self._tf.append(tf)
        total_len = sum(len(self._tokenize(d)) for d in self._docs)
        self._avgdl = total_len / max(len(self._docs), 1)
        return idx

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.01) -> List[Tuple[float, str, Dict]]:
        if not self._docs:
            return []
        N      = len(self._docs)
        qterms = self._tokenize(query)
        scores = []
        for i, tf in enumerate(self._tf):
            dl    = sum(tf.values())
            score = 0.0
            for t in qterms:
                if t not in tf:
                    continue
                df_t = self._df.get(t, 0)
                idf  = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
                num  = tf[t] * (self.k1 + 1)
                den  = tf[t] + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
                score += idf * (num / den)
            if score >= min_score:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [(s, self._docs[i], self._meta[i]) for s, i in scores[:top_k]]

    def clear(self) -> None:
        self._docs.clear(); self._meta.clear()
        self._tf.clear(); self._df.clear(); self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._docs)


# ─────────────────────────────────────────────
#  Short-Term Memory  (active context window)
# ─────────────────────────────────────────────

class ShortTermMemory:
    """
    Manages the active conversation window with:
      • Token-budget trimming (oldest messages removed first)
      • Priority preservation (system prompt always kept)
      • Working memory scratchpad for reasoning chains
      • Sliding injection of retrieved long-term memories
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, max_tokens: int = 2048,
                 system_prompt: str = "You are LionAI, a helpful AI assistant.") -> None:
        self.max_tokens    = max_tokens
        self.system_prompt = system_prompt
        self._messages:    List[Message] = []
        self._working_mem: List[str]     = []   # reasoning scratchpad
        self._session_id   = hashlib.md5(str(time.time()).encode()).hexdigest()[:10]

    # ── Public ──────────────────────────────
    def add(self, role: str, content: str, **meta) -> None:
        self._messages.append(Message(role=role, content=content, metadata=meta))
        self._trim()

    def add_working(self, thought: str) -> None:
        """Add a reasoning step to working memory scratchpad."""
        self._working_mem.append(f"[Thought] {thought}")
        if len(self._working_mem) > 10:
            self._working_mem.pop(0)

    def get_prompt(self, injected_context: str = "") -> str:
        """Build the full flat prompt string for the model."""
        parts: List[str] = []

        # System block
        sys_content = self.system_prompt
        if injected_context:
            sys_content += f"\n\n{injected_context}"
        if self._working_mem:
            sys_content += "\n\n[Working Memory]\n" + "\n".join(self._working_mem[-5:])
        parts.append(f"<sys>{sys_content}</sys>")

        # Conversation turns
        for m in self._messages:
            tag = {"user": "usr", "assistant": "ast", "system": "sys"}.get(m.role, "usr")
            parts.append(f"<{tag}>{m.content}</{tag}>")

        parts.append("<ast>")
        return "\n".join(parts)

    def get_context(self, include_system: bool = True) -> List[Dict]:
        result = []
        if include_system:
            result.append({"role": "system", "content": self.system_prompt})
        result.extend(m.to_dict() for m in self._messages)
        return result

    def reset(self, keep_system: bool = True) -> None:
        self._messages.clear()
        self._working_mem.clear()
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:10]

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._messages if m.role == "user")

    @property
    def total_chars(self) -> int:
        return sum(len(m.content) for m in self._messages)

    @property
    def approx_tokens(self) -> int:
        return self.total_chars // self.CHARS_PER_TOKEN

    def summary(self) -> Dict:
        return {
            "session_id":    self._session_id,
            "turns":         self.turn_count,
            "messages":      len(self._messages),
            "approx_tokens": self.approx_tokens,
            "budget":        self.max_tokens,
            "utilisation":   round(self.approx_tokens / max(self.max_tokens, 1), 2),
        }

    # ── Internal ────────────────────────────
    def _trim(self) -> None:
        budget_chars = self.max_tokens * self.CHARS_PER_TOKEN
        while self.total_chars > budget_chars and len(self._messages) > 2:
            # Always keep the most recent user+assistant pair
            self._messages.pop(0)


# ─────────────────────────────────────────────
#  Thread-safe SQLite Connection Pool
# ─────────────────────────────────────────────

class _ConnectionPool:
    """Per-thread SQLite connections — avoids locking across threads."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._local  = threading.local()

    def get(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")     # faster concurrent writes
            conn.execute("PRAGMA synchronous=NORMAL")   # good safety/speed balance
            conn.execute("PRAGMA temp_store=MEMORY")    # temp tables in RAM
            conn.execute("PRAGMA cache_size=-8000")     # 8 MB page cache
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn


# ─────────────────────────────────────────────
#  Long-Term Memory  (SQLite, persistent)
# ─────────────────────────────────────────────

class LongTermMemory:
    """
    Persistent memory with BM25 retrieval, composite scoring,
    memory decay, and consolidation.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool = _ConnectionPool(str(self.db_path))
        self._bm25 = BM25Index()
        self._setup()
        self._rebuild_bm25()
        logger.info("LongTermMemory: %s", self.db_path)

    def _conn(self) -> sqlite3.Connection:
        return self._pool.get()

    def _setup(self) -> None:
        self._conn().executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                key           TEXT PRIMARY KEY,
                value         TEXT NOT NULL,
                category      TEXT DEFAULT 'general',
                importance    REAL DEFAULT 0.5,
                access_count  INTEGER DEFAULT 0,
                created_at    REAL,
                last_accessed REAL,
                decay_rate    REAL DEFAULT 0.0,
                embedding     TEXT          -- JSON float list, optional
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                summary     TEXT,
                turn_count  INTEGER,
                created_at  REAL,
                tags        TEXT
            );
            CREATE TABLE IF NOT EXISTS episodic (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT,
                summary     TEXT,
                key_facts   TEXT,
                created_at  REAL
            );
            CREATE INDEX IF NOT EXISTS idx_cat  ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_imp  ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_acc  ON memories(last_accessed DESC);
        """)
        self._conn().commit()

    def _rebuild_bm25(self) -> None:
        """Rebuild in-memory BM25 index from DB on startup."""
        self._bm25.clear()
        rows = self._conn().execute(
            "SELECT key, value, category, importance FROM memories"
        ).fetchall()
        for r in rows:
            self._bm25.add(f"{r['key']} {r['value']}",
                           {"key": r["key"], "category": r["category"],
                            "importance": r["importance"]})
        logger.debug("BM25 index rebuilt: %d entries", len(self._bm25))

    # ── Store / Retrieve ────────────────────
    def store(self, key: str, value: str,
              category: str = "general",
              importance: float = 0.5,
              decay_rate: float = 0.0) -> None:
        now = time.time()
        self._conn().execute("""
            INSERT OR REPLACE INTO memories
              (key, value, category, importance, access_count,
               created_at, last_accessed, decay_rate)
            VALUES (?, ?, ?, ?, COALESCE(
                (SELECT access_count FROM memories WHERE key=?), 0
            ), COALESCE(
                (SELECT created_at FROM memories WHERE key=?), ?
            ), ?, ?)
        """, (key, value, category, importance, key, key, now, now, decay_rate))
        self._conn().commit()
        # Update BM25
        self._bm25.add(f"{key} {value}",
                       {"key": key, "category": category, "importance": importance})

    def retrieve(self, key: str) -> Optional[MemoryEntry]:
        row = self._conn().execute(
            "SELECT * FROM memories WHERE key=?", (key,)
        ).fetchone()
        if not row:
            return None
        now = time.time()
        self._conn().execute(
            "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE key=?",
            (now, key)
        )
        self._conn().commit()
        return MemoryEntry(
            key=row["key"], value=row["value"], category=row["category"],
            importance=row["importance"], access_count=row["access_count"] + 1,
            created_at=row["created_at"], last_accessed=now,
            decay_rate=row["decay_rate"] or 0.0,
        )

    def search(self, query: str, top_k: int = 8,
               category: Optional[str] = None) -> List[MemoryEntry]:
        """BM25 search with composite score re-ranking."""
        raw = self._bm25.search(query, top_k=top_k * 2)
        entries: List[Tuple[float, MemoryEntry]] = []
        now = time.time()
        for bm25_score, _, meta in raw:
            key = meta.get("key", "")
            e   = self.retrieve(key)
            if e is None:
                continue
            if category and e.category != category:
                continue
            # Blend BM25 score with composite memory score
            combined = 0.6 * (bm25_score / 10.0) + 0.4 * e.score(now)
            entries.append((combined, e))
        entries.sort(key=lambda x: -x[0])
        return [e for _, e in entries[:top_k]]

    def delete(self, key: str) -> bool:
        cur = self._conn().execute("DELETE FROM memories WHERE key=?", (key,))
        self._conn().commit()
        self._rebuild_bm25()
        return cur.rowcount > 0

    def list_all(self, category: Optional[str] = None,
                 sort_by: str = "score") -> List[MemoryEntry]:
        if category:
            rows = self._conn().execute(
                "SELECT * FROM memories WHERE category=?", (category,)
            ).fetchall()
        else:
            rows = self._conn().execute("SELECT * FROM memories").fetchall()
        entries = [MemoryEntry(
            key=r["key"], value=r["value"], category=r["category"],
            importance=r["importance"], access_count=r["access_count"],
            created_at=r["created_at"], last_accessed=r["last_accessed"],
            decay_rate=r["decay_rate"] or 0.0,
        ) for r in rows]
        now = time.time()
        if sort_by == "score":
            entries.sort(key=lambda e: -e.score(now))
        elif sort_by == "importance":
            entries.sort(key=lambda e: -e.importance)
        elif sort_by == "recent":
            entries.sort(key=lambda e: -e.last_accessed)
        return entries

    def consolidate(self, similarity_threshold: float = 0.7) -> int:
        """
        Merge semantically similar memories.
        Returns number of entries removed.
        """
        entries = self.list_all()
        removed = 0
        merged_keys: set = set()

        for i, ea in enumerate(entries):
            if ea.key in merged_keys:
                continue
            for eb in entries[i + 1:]:
                if eb.key in merged_keys:
                    continue
                # Simple overlap similarity
                wa = set(ea.value.lower().split())
                wb = set(eb.value.lower().split())
                if not wa or not wb:
                    continue
                sim = len(wa & wb) / len(wa | wb)
                if sim >= similarity_threshold:
                    # Keep higher-importance entry; merge value
                    keep = ea if ea.importance >= eb.importance else eb
                    drop = eb if keep is ea else ea
                    merged_val = keep.value + " | " + drop.value[:80]
                    self.store(keep.key, merged_val, keep.category,
                               max(ea.importance, eb.importance))
                    self.delete(drop.key)
                    merged_keys.add(drop.key)
                    removed += 1

        if removed:
            self._rebuild_bm25()
            logger.info("Consolidated %d duplicate memories", removed)
        return removed

    def apply_decay(self) -> int:
        """
        Apply decay to all memories. Returns number with lowered importance.
        Call periodically (e.g., once per session start).
        """
        now   = time.time()
        rows  = self._conn().execute("SELECT key, importance, decay_rate, last_accessed FROM memories").fetchall()
        updated = 0
        for r in rows:
            if not r["decay_rate"]:
                continue
            age_days = (now - (r["last_accessed"] or now)) / 86400
            new_imp  = r["importance"] * math.exp(-r["decay_rate"] * age_days)
            if new_imp < 0.05:
                self.delete(r["key"])   # forget very faded memories
            else:
                self._conn().execute(
                    "UPDATE memories SET importance=? WHERE key=?",
                    (new_imp, r["key"])
                )
            updated += 1
        self._conn().commit()
        return updated

    # ── Sessions / Episodic ─────────────────
    def save_session(self, session_id: str, summary: str,
                     turn_count: int, tags: Optional[List[str]] = None) -> None:
        self._conn().execute("""
            INSERT OR REPLACE INTO sessions (session_id, summary, turn_count, created_at, tags)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, summary, turn_count, time.time(), json.dumps(tags or [])))
        self._conn().commit()

    def save_episodic(self, session_id: str, summary: str,
                      key_facts: List[str]) -> None:
        self._conn().execute("""
            INSERT INTO episodic (session_id, summary, key_facts, created_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, summary, json.dumps(key_facts), time.time()))
        self._conn().commit()

    def get_episodic(self, limit: int = 5) -> List[Dict]:
        rows = self._conn().execute(
            "SELECT * FROM episodic ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"session_id": r["session_id"], "summary": r["summary"],
                 "key_facts": json.loads(r["key_facts"] or "[]"),
                 "created_at": r["created_at"]} for r in rows]

    def get_sessions(self, limit: int = 20) -> List[Dict]:
        rows = self._conn().execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"session_id": r["session_id"], "summary": r["summary"],
                 "turn_count": r["turn_count"], "created_at": r["created_at"],
                 "tags": json.loads(r["tags"] or "[]")} for r in rows]

    # ── Export / Import ─────────────────────
    def export_snapshot(self, path: Path) -> None:
        path = Path(path)
        rows = self._conn().execute("SELECT * FROM memories").fetchall()
        data = [dict(r) for r in rows]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Memory snapshot exported → %s (%d entries)", path, len(data))

    def import_snapshot(self, path: Path, overwrite: bool = False) -> int:
        path  = Path(path)
        items = json.loads(path.read_text(encoding="utf-8"))
        added = 0
        for item in items:
            if not overwrite:
                exists = self._conn().execute(
                    "SELECT 1 FROM memories WHERE key=?", (item["key"],)
                ).fetchone()
                if exists:
                    continue
            self.store(
                item["key"], item["value"],
                item.get("category", "general"),
                item.get("importance", 0.5),
                item.get("decay_rate", 0.0),
            )
            added += 1
        logger.info("Imported %d memories from %s", added, path)
        return added

    def stats(self) -> Dict:
        n_mem  = self._conn().execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        n_sess = self._conn().execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        n_epis = self._conn().execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
        return {"total_memories": n_mem, "total_sessions": n_sess, "episodic_entries": n_epis}

    def vacuum(self) -> None:
        self._conn().execute("VACUUM")
        logger.info("Memory DB vacuumed")

    def close(self) -> None:
        if hasattr(self._pool._local, "conn") and self._pool._local.conn:
            self._pool._local.conn.close()


# ─────────────────────────────────────────────
#  Semantic Memory  (BM25 + optional dense)
# ─────────────────────────────────────────────

class SemanticMemory:
    """
    Semantic knowledge store backed by BM25 + SQLite persistence.
    Can be upgraded to dense embeddings if a small encoder is provided.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._bm25 = BM25Index()
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_entries (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    text     TEXT NOT NULL,
                    metadata TEXT,
                    added_at REAL
                )
            """)
            self._conn.commit()
            self._load_from_db()

    def _load_from_db(self) -> None:
        rows = self._conn.execute(
            "SELECT text, metadata FROM semantic_entries"
        ).fetchall()
        for r in rows:
            meta = json.loads(r[1]) if r[1] else {}
            self._bm25.add(r[0], meta)
        logger.debug("SemanticMemory: loaded %d entries", len(rows))

    def add(self, text: str, metadata: Optional[Dict] = None) -> None:
        self._bm25.add(text, metadata)
        if self._conn:
            self._conn.execute(
                "INSERT INTO semantic_entries (text, metadata, added_at) VALUES (?, ?, ?)",
                (text, json.dumps(metadata or {}), time.time())
            )
            self._conn.commit()

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.01) -> List[Tuple[float, str, Dict]]:
        return self._bm25.search(query, top_k, min_score)

    def format_context(self, query: str, top_k: int = 3,
                       max_chars: int = 800) -> str:
        results = self.search(query, top_k)
        if not results:
            return ""
        lines = ["[Semantic Memory]"]
        total = 0
        for score, text, _ in results:
            snippet = text[: max_chars - total]
            lines.append(f"• {snippet}  (rel:{score:.2f})")
            total += len(snippet)
            if total >= max_chars:
                break
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._bm25)


# ─────────────────────────────────────────────
#  Unified Memory Manager
# ─────────────────────────────────────────────

class MemoryManager:
    """
    Unified interface to all memory tiers.
    Handles auto-injection, session persistence, and episodic extraction.
    """

    def __init__(self, data_dir: Path,
                 max_context_tokens: int = 2048,
                 system_prompt: str = "You are LionAI, a helpful AI assistant.") -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        self.short    = ShortTermMemory(max_context_tokens, system_prompt)
        self.long     = LongTermMemory(data_dir / "memory.db")
        self.semantic = SemanticMemory(data_dir / "semantic.db")

        # Apply decay on each startup
        decayed = self.long.apply_decay()
        if decayed:
            logger.debug("Memory decay applied to %d entries", decayed)

    # ── High-level API ──────────────────────
    def add_turn(self, user_msg: str, assistant_msg: str) -> None:
        self.short.add("user",      user_msg)
        self.short.add("assistant", assistant_msg)

    def remember(self, key: str, value: str,
                 category: str = "general",
                 importance: float = 0.7,
                 decay_rate: float = 0.0) -> None:
        self.long.store(key, value, category, importance, decay_rate)
        self.semantic.add(f"{key}: {value}", {"key": key, "category": category})

    def recall(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant memories as formatted context string."""
        lt_entries = self.long.search(query, top_k)
        sem_block  = self.semantic.format_context(query, top_k=2)
        episodic   = self.long.get_episodic(limit=2)

        parts: List[str] = []
        if lt_entries:
            parts.append("[Stored Knowledge]")
            for e in lt_entries:
                parts.append(f"• {e.key}: {e.value}")
        if sem_block:
            parts.append(sem_block)
        if episodic:
            parts.append("[Recent Session Context]")
            for ep in episodic:
                parts.append(f"• {ep['summary'][:120]}")

        return "\n".join(parts)

    def build_prompt(self, query: str) -> str:
        context = self.recall(query)
        return self.short.get_prompt(injected_context=context)

    def extract_facts(self, text: str) -> List[str]:
        """
        Heuristic extraction of potential facts from assistant responses.
        Auto-stores high-confidence facts into semantic memory.
        """
        facts: List[str] = []
        # Simple heuristics: sentences with assertive structure
        sentences = [s.strip() for s in text.replace("!", ".").split(".") if len(s.strip()) > 20]
        indicators = ["is ", "are ", "was ", "were ", "means ", "refers to ", "defined as "]
        for s in sentences[:5]:
            if any(ind in s.lower() for ind in indicators):
                facts.append(s)
                self.semantic.add(s, {"source": "response", "auto": True})
        return facts

    def save_session(self, custom_summary: str = "") -> None:
        ctx = self.short.get_context(include_system=False)
        if not custom_summary:
            user_msgs = [m["content"][:60] for m in ctx if m.get("role") == "user"]
            custom_summary = (
                f"Session {self.short.session_id}: "
                f"{self.short.turn_count} turns. "
                f"Topics: {'; '.join(user_msgs[:3])}"
            )

        # Save basic session
        self.long.save_session(
            self.short.session_id, custom_summary, self.short.turn_count
        )

        # Auto-extract key facts and store as episodic memory
        all_text = " ".join(m["content"] for m in ctx if m.get("role") == "assistant")
        facts = self.extract_facts(all_text[:2000])
        if facts:
            self.long.save_episodic(self.short.session_id, custom_summary, facts)

    def consolidate(self) -> Dict:
        """Merge duplicate memories and vacuum the DB."""
        removed = self.long.consolidate()
        self.long.vacuum()
        return {"consolidated": removed, **self.long.stats()}

    def full_stats(self) -> Dict:
        return {
            "short_term":  self.short.summary(),
            "long_term":   self.long.stats(),
            "semantic":    {"entries": len(self.semantic)},
        }

    def export(self, path: Path) -> None:
        self.long.export_snapshot(Path(path))

    def import_from(self, path: Path) -> int:
        return self.long.import_snapshot(Path(path))
