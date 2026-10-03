"""
rag_engine.py — Incremental RAG engine with page-level citations

Key ideas
---------
1. SQLite is the source of truth (chunks + metadata). Incremental O(D) inserts.
2. FAISS uses IndexIDMap2 so the FAISS id == the SQLite chunk_id.
   Adding a document never touches existing vectors.
3. BM25 rebuild + FAISS snapshot happen on a BACKGROUND thread with
   version counters, so /upload latency is O(new chunks), not O(corpus).
4. Retrieval returns dicts with filename + page, not raw strings.

PDF
 │
 ▼
Extract text
 │
 ▼
Split into small chunks
 │
 ▼
Create embeddings
 │
 ├───────────────► FAISS
 │                  Semantic search
 │
 └───────────────► SQLite
                    Metadata + original text

User Query
 │
 ▼
"What is QPS vs concurrency?"
 │
 ├──► FAISS ──► semantic matches
 │
 ├──► BM25 ───► keyword matches
 │
 ▼
RRF
 │
 ▼
Combine results
 │
 ▼
Cross-Encoder
 │
 ▼
Best chunks
 │
 ▼
LLM
 │
 ▼
Answer
"""

import os
import re
import time
import sqlite3
import hashlib
import threading
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
import requests
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
OLLAMA_BASE      = "http://localhost:11434"
EMBED_BATCH_URL  = f"{OLLAMA_BASE}/api/embed"        # batch endpoint (newer Ollama)
EMBED_SINGLE_URL = f"{OLLAMA_BASE}/api/embeddings"   # single-text fallback
EMBED_MODEL      = "nomic-embed-text" # Convert text to vector embeddings for semantic search It is designed to capture the semantic meaning of the text, allowing for more accurate retrieval of relevant information based on the content rather than just keyword matching.
VECTOR_DIM       = 768   # 768-dimensional vector

DB_PATH    = "rag.db"
FAISS_FILE = "faiss.index"
FAISS_TMP  = "faiss.index.tmp"

CHUNK_SIZE       = 300
CHUNK_OVERLAP    = 60
EMBED_BATCH_SIZE = 32
RERANK_MODEL     = "BAAI/bge-reranker-v2-m3"
RRF_K            = 20

TOKEN_RE = re.compile(r"[a-z0-9]+")

# "Redis is FAST!" => "Redis is FAST!", This is used for BM25 (keyword-based search algorithm)
def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

# Chunking + overlap preserves context while keeping retrieval manageable.
def chunk_page(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Word-based chunking with overlap. Keeps page boundaries intact."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    out = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        if piece:
            out.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return out


# ======================================================================
# ENGINE
# ======================================================================
class RAGEngine:

    def __init__(self):
        self._lock = threading.RLock() # Reentrant Lock 

        # ---- SQLite -------------------------------------------------
        self._db = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

        # ---- FAISS --------------------------------------------------
        self.index = self._load_or_create_index()
        self._sync_index_with_db()

        # ---- BM25 (lazy, background-rebuilt) ------------------------
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[int] = []
        self._bm25_version = 0        # bumped on every write
        self._bm25_built = -1         # version currently in RAM

        # ---- FAISS snapshot bookkeeping -----------------------------
        self._index_version = 0
        self._index_saved = -1

        # ---- Reranker (lazy) ----------------------------------------
        self._reranker = None
        self._reranker_lock = threading.Lock()

        # ---- Background worker --------------------------------------
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._background_loop, daemon=True)
        self._worker.start()

        # Warm BM25 on boot
        self._rebuild_bm25()

    # ------------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------------
    def _init_schema(self):
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            sha256      TEXT    UNIQUE NOT NULL,
            uploaded_at REAL    NOT NULL,
            num_chunks  INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id      INTEGER NOT NULL,
            page        INTEGER,
            chunk_index INTEGER NOT NULL,
            text        TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """)
        self._db.commit()

    # ------------------------------------------------------------------
    # FAISS helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _new_index():
        # Cosine similarity = inner product on L2-normalised vectors.
        """
        L2 is basically a way of calculating the length/size of a vector.
        For a normal number: 5 For a normal number: 5, But a vector has multiple numbers: [3, 4] => square root of (3^2 + 4^2) = 5
        normalize_L2()
        Original vector:
        [3, 4]
        Length = 5

        After L2 normalization:
        [3/5, 4/5] = [0.6, 0.8]
        
        Now its length is: √(0.6² + 0.8²) = 1
        """
        
        base = faiss.IndexFlatIP(VECTOR_DIM)
        return faiss.IndexIDMap2(base)

        # ---- SCALE-UP: swap the line above for HNSW -----------------
        # base = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
        # base.hnsw.efConstruction = 200
        # base.hnsw.efSearch = 64
        # return faiss.IndexIDMap2(base)

    def _load_or_create_index(self):
        if os.path.exists(FAISS_FILE):
            try:
                idx = faiss.read_index(FAISS_FILE)
                print(f"[rag] FAISS index loaded: {idx.ntotal} vectors")
                return idx
            except Exception as e:
                print(f"[rag] could not load FAISS index ({e}); starting empty")
        return self._new_index()

    def _index_ids(self) -> set:
        if not hasattr(self.index, "id_map"):
            return set()
        return {int(i) for i in faiss.vector_to_array(self.index.id_map)}

    def _sync_index_with_db(self):
        """If the DB has chunks the index doesn't (crash / deleted file),
        re-embed only the missing ones — never the whole corpus."""
        have = self._index_ids()
        rows = self._db.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        missing = [(cid, txt) for cid, txt in rows if cid not in have]
        if not missing:
            return
        print(f"[rag] re-embedding {len(missing)} chunks missing from index …")
        for i in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[i:i + EMBED_BATCH_SIZE]
            mat = np.asarray(self._embed_batch([t for _, t in batch]), dtype="float32")
            faiss.normalize_L2(mat)
            ids = np.asarray([cid for cid, _ in batch], dtype="int64")
            self.index.add_with_ids(mat, ids)
        self._save_index()

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Try the batch endpoint first; fall back to one-by-one."""
        try:
            r = requests.post(
                EMBED_BATCH_URL,
                json={"model": EMBED_MODEL, "input": texts},
                timeout=180,
            )
            if r.status_code == 200:
                embs = r.json().get("embeddings")
                if embs and len(embs) == len(texts):
                    return embs
        except Exception:
            pass

        out = []
        for t in texts:
            r = requests.post(
                EMBED_SINGLE_URL,
                json={"model": EMBED_MODEL, "prompt": t},
                timeout=180,
            )
            r.raise_for_status()
            out.append(r.json()["embedding"])
        return out

    # ------------------------------------------------------------------
    # WRITE PATH
    # ------------------------------------------------------------------
    def add_document(self,
                     pages: List[Tuple[Optional[int], str]],
                     filename: str,
                     content_hash: Optional[str] = None) -> Dict[str, Any]:
        """
        pages : [(page_number, text), ...]   page_number may be None for .txt
        Returns a summary dict.
        """
        joined = "\n".join(t for _, t in pages)
        digest = content_hash or hashlib.sha256(joined.encode()).hexdigest()

        # --- dedup -------------------------------------------------
        row = self._db.execute(
            "SELECT doc_id, num_chunks FROM documents WHERE sha256 = ?", (digest,)
        ).fetchone()
        if row:
            return {"doc_id": row[0], "chunks_added": 0, "duplicate": True}

        # --- chunk (page-aware) -------------------------------------
        chunks: List[Tuple[Optional[int], int, str]] = []
        for page_no, page_text in pages:
            for i, c in enumerate(chunk_page(page_text)):
                chunks.append((page_no, i, c))

        if not chunks:
            return {"doc_id": None, "chunks_added": 0, "duplicate": False,
                    "warning": "no extractable text"}

        # --- embed (outside the lock, batched) ----------------------
        vectors: List[List[float]] = []
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            vectors.extend(self._embed_batch([c[2] for c in batch]))

        mat = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(mat)

        # --- persist metadata + get ids -----------------------------
        with self._lock:
            cur = self._db.cursor()
            cur.execute(
                "INSERT INTO documents(filename, sha256, uploaded_at, num_chunks)"
                " VALUES (?,?,?,?)",
                (filename, digest, time.time(), len(chunks)),
            )
            doc_id = cur.lastrowid

            cur.executemany(
                "INSERT INTO chunks(doc_id, page, chunk_index, text) VALUES (?,?,?,?)",
                [(doc_id, p, i, t) for (p, i, t) in chunks],
            )
            self._db.commit()

            # AUTOINCREMENT => ids come back in insertion order
            ids = [r[0] for r in cur.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id=? ORDER BY chunk_id",
                (doc_id,),
            )]

            # --- incremental FAISS add: only the new vectors --------
            self.index.add_with_ids(mat, np.asarray(ids, dtype="int64"))

            # --- mark background work -------------------------------
            self._bm25_version += 1
            self._index_version += 1

        print(f"[rag] +{len(chunks)} chunks from '{filename}' (doc_id={doc_id})")

        return {"doc_id": doc_id, "chunks_added": len(chunks),
                "duplicate": False, "filename": filename}

    # ------------------------------------------------------------------
    # READ PATH
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 4, candidates: int = 50) -> List[Dict]:
        # 1) semantic
        qv = np.asarray(self._embed_batch([query]), dtype="float32")
        faiss.normalize_L2(qv)

        sem_ids: List[int] = []
        if self.index.ntotal > 0:
            _, I = self.index.search(qv, candidates)
            sem_ids = [int(i) for i in I[0] if i != -1]

        # 2) lexical
        lex_ids = self._bm25_search(query, candidates)

        # 3) fuse (RRF — no score normalisation headaches)
        fused = self._rrf([sem_ids, lex_ids])
        cand_ids = [cid for cid, _ in fused[:candidates]]
        if not cand_ids:
            return []

        # 4) hydrate with metadata
        # Hydration => By id, we need actual data from db
        rows = self._fetch_chunks(cand_ids)

        # 5) cross-encoder rerank => The model evaluates how relevant this chunk is to the query.
        
        return self._rerank(query, rows, top_k)

    def _fetch_chunks(self, ids: List[int]) -> List[Dict]:
        placeholders = ",".join("?" * len(ids))
        sql = f"""
            SELECT c.chunk_id, c.text, c.page, c.chunk_index,
                   d.doc_id, d.filename
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id IN ({placeholders})
        """
        rows = self._db.execute(sql, ids).fetchall()
        by_id = {
            r[0]: {
                "chunk_id": r[0],
                "text": r[1],
                "page": r[2],
                "chunk_index": r[3],
                "doc_id": r[4],
                "filename": r[5],
            }
            for r in rows
        }
        return [by_id[i] for i in ids if i in by_id]

    def _bm25_search(self, query: str, top_k: int) -> List[int]:
        with self._lock:
            bm25, ids = self._bm25, self._bm25_ids
        if bm25 is None or not ids:
            return []
        scores = bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [ids[i] for i in ranked if scores[i] > 0]

    """
Reciprocal Rank Fusion (RRF) 
- is a ranking algorithm that combines the scores of two ranking algorithms to produce a single ranking. 
- It is used to combine the scores of semantic search and keyword search to produce a final ranking of relevant documents.
    FAISS       BM25
  │           │
  └─────┬─────┘
        ▼
       RRF
        │
        ▼
Combined ranking
    """
    
    @staticmethod
    def _rrf(rankings: List[List[int]], k: int = RRF_K):
        scores: Dict[int, float] = {}
        for ranking in rankings:
            for rank, cid in enumerate(ranking):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def _get_reranker(self):
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    try:
                        from sentence_transformers import CrossEncoder
                        self._reranker = CrossEncoder(RERANK_MODEL)
                    except Exception as e:
                        print(f"[rag] reranker unavailable, skipping: {e}")
                        self._reranker = False  # sentinel — never retry
        return self._reranker or None

    def _rerank(self, query: str, rows: List[Dict], top_k: int) -> List[Dict]:
        if not rows:
            return []
        reranker = self._get_reranker()
        if reranker is None:
            return rows[:top_k]

        scores = reranker.predict([(query, r["text"]) for r in rows])
        for r, s in zip(rows, scores):
            r["rerank_score"] = float(s)
        rows.sort(key=lambda r: -r["rerank_score"])
        return rows[:top_k]

    # ------------------------------------------------------------------
    # BACKGROUND: BM25 rebuild + FAISS snapshot
    # ------------------------------------------------------------------
    def _rebuild_bm25(self):
        t0 = time.time()
        version = self._bm25_version

        rows = self._db.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        ids = [r[0] for r in rows]
        corpus = [tokenize(r[1]) for r in rows]
        bm25 = BM25Okapi(corpus) if corpus else None

        with self._lock:
            self._bm25 = bm25
            self._bm25_ids = ids
            self._bm25_built = version

        print(f"[rag] BM25 rebuilt: {len(ids)} chunks in {time.time() - t0:.2f}s")

    def _save_index(self):
        with self._lock:
            faiss.write_index(self.index, FAISS_TMP)
        os.replace(FAISS_TMP, FAISS_FILE)   # atomic — no torn file on crash

    def _background_loop(self):
        while not self._stop.is_set():
            time.sleep(1.0)             # debounce: bursts collapse into one rebuild
            try:
                if self._bm25_built != self._bm25_version:
                    self._rebuild_bm25()
                if self._index_saved != self._index_version:
                    self._save_index()
                    self._index_saved = self._index_version
            except Exception as e:
                print(f"[rag] background worker error: {e}")

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        docs = self._db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "documents": docs,
            "chunks_in_db": chunks,
            "vectors_in_index": int(self.index.ntotal),
            "bm25_in_sync": self._bm25_built == self._bm25_version,
        }

    def shutdown(self):
        self._stop.set()
        self._save_index()
        self._db.close()
        