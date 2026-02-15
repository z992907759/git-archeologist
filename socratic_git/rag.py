"""Embedding + LanceDB indexing and retrieval."""
import os

try:
    import lancedb
except Exception:  # pragma: no cover - dependency/runtime environment issues
    lancedb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - dependency/runtime environment issues
    SentenceTransformer = None

from socratic_git.miner import build_text

_EMBED_MODEL = None
_DB = None
_TABLE = None
_DB_PATH = "data/lancedb"
_TABLE_NAME = "commits"


def configure_table(table_name):
    """Set current table and connect DB."""
    global _DB, _TABLE, _TABLE_NAME
    _TABLE_NAME = table_name
    _TABLE = None
    if lancedb is None:
        _DB = None
    else:
        os.makedirs(_DB_PATH, exist_ok=True)
        _DB = lancedb.connect(_DB_PATH)


def _ensure_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("Missing dependency: sentence-transformers")
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def open_existing_table():
    """Open current table; raise if not found."""
    global _TABLE
    if _DB is None:
        raise RuntimeError("Missing dependency: lancedb")
    _TABLE = _DB.open_table(_TABLE_NAME)
    return _TABLE


def index_commits(commits):
    """Embed commit texts and overwrite current LanceDB table."""
    global _DB, _TABLE

    if SentenceTransformer is None or lancedb is None:
        raise RuntimeError("Missing dependency: sentence-transformers or lancedb")

    if _DB is None:
        os.makedirs(_DB_PATH, exist_ok=True)
        _DB = lancedb.connect(_DB_PATH)

    _ensure_embed_model()

    texts = [build_text(commit) for commit in commits]
    vectors = _EMBED_MODEL.encode(texts).tolist() if texts else []

    rows = []
    for commit, text, vector in zip(commits, texts, vectors):
        rows.append(
            {
                "vector": vector,
                "hash": commit.get("hash", ""),
                "author": commit.get("author", ""),
                "date": commit.get("date", ""),
                "message": commit.get("message", ""),
                "diff": commit.get("diff", ""),
                "text": text,
            }
        )

    _TABLE = _DB.create_table(_TABLE_NAME, data=rows, mode="overwrite")

    records = []
    for commit, text in zip(commits, texts):
        records.append({"commit": commit, "text": text})
    return records


def append_commits(commits):
    """Embed and append commits to current LanceDB table (create if missing)."""
    global _DB, _TABLE

    if SentenceTransformer is None or lancedb is None:
        raise RuntimeError("Missing dependency: sentence-transformers or lancedb")

    if _DB is None:
        os.makedirs(_DB_PATH, exist_ok=True)
        _DB = lancedb.connect(_DB_PATH)

    _ensure_embed_model()

    texts = [build_text(commit) for commit in commits]
    vectors = _EMBED_MODEL.encode(texts).tolist() if texts else []
    rows = []
    for commit, text, vector in zip(commits, texts, vectors):
        rows.append(
            {
                "vector": vector,
                "hash": commit.get("hash", ""),
                "author": commit.get("author", ""),
                "date": commit.get("date", ""),
                "message": commit.get("message", ""),
                "diff": commit.get("diff", ""),
                "text": text,
            }
        )

    if not rows:
        return []

    if _TABLE is None:
        try:
            _TABLE = _DB.open_table(_TABLE_NAME)
        except Exception:
            _TABLE = _DB.create_table(_TABLE_NAME, data=rows, mode="overwrite")
            return [{"commit": c, "text": t} for c, t in zip(commits, texts)]

    _TABLE.add(rows)
    return [{"commit": c, "text": t} for c, t in zip(commits, texts)]


def retrieve(query, topk=3):
    """Retrieve top-k related commits by vector similarity."""
    if not query or _TABLE is None:
        return []

    _ensure_embed_model()

    query_vector = _EMBED_MODEL.encode(query).tolist()
    rows = _TABLE.search(query_vector).limit(topk).to_list()

    results = []
    for row in rows:
        results.append(
            {
                "commit": {
                    "hash": row.get("hash", ""),
                    "author": row.get("author", ""),
                    "date": row.get("date", ""),
                    "message": row.get("message", ""),
                    "diff": row.get("diff", ""),
                },
                "text": row.get("text", ""),
                "score": row.get("_distance"),
            }
        )
    return results
