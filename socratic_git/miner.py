"""Git mining and diff cleaning utilities."""
import os
import re

try:
    from git import NULL_TREE, Repo
    from git.exc import InvalidGitRepositoryError, NoSuchPathError
except Exception:  # pragma: no cover - dependency/runtime environment issues
    NULL_TREE = None
    Repo = None
    InvalidGitRepositoryError = Exception
    NoSuchPathError = Exception


def extract_commits(repo_path, n=50):
    """Read latest commits and include raw diff text."""
    if n <= 0:
        return []
    if Repo is None:
        return []

    try:
        repo = Repo(repo_path)
    except (NoSuchPathError, InvalidGitRepositoryError):
        return []

    commits = []
    for commit in repo.iter_commits(max_count=n):
        diff_text = ""
        try:
            if commit.parents:
                diffs = commit.diff(commit.parents[0], create_patch=True)
            else:
                diffs = commit.diff(NULL_TREE, create_patch=True)

            parts = []
            for d in diffs:
                raw = d.diff
                if isinstance(raw, bytes):
                    parts.append(raw.decode("utf-8", errors="ignore"))
                else:
                    parts.append(str(raw or ""))
            diff_text = "\n".join(parts)
        except Exception:
            diff_text = ""

        commits.append(
            {
                "hash": commit.hexsha,
                "author": commit.author.name,
                "date": commit.committed_datetime.date().isoformat(),
                "message": commit.message.strip(),
                "diff": diff_text,
            }
        )
    return commits


def compress_diff(diff_text):
    """Simple truncation-based diff compression."""
    if not diff_text:
        return ""

    max_lines = 300
    max_chars = 20000

    lines = diff_text.splitlines()
    compressed = "\n".join(lines[:max_lines])
    if len(compressed) > max_chars:
        compressed = compressed[:max_chars]
    return compressed


def clean_diff(diff_text):
    """Clean diff by dropping noisy files and limiting block/line volume."""
    if not diff_text:
        return ""

    excluded_exact = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
    excluded_ext = (".png", ".jpg", ".pdf", ".ipynb")

    max_lines_per_block = 40
    max_blocks = 10
    max_total_lines = 400

    lines = diff_text.splitlines()
    blocks = []
    current = []

    for line in lines:
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            current = [line]
        else:
            if current:
                current.append(line)
    if current:
        blocks.append(current)

    kept = []
    total = 0
    kept_blocks = 0

    for block in blocks:
        header = block[0]
        parts = header.split()
        path = ""
        if len(parts) >= 4 and parts[3].startswith("b/"):
            path = parts[3][2:]

        lower_path = path.lower()
        base_name = lower_path.rsplit("/", 1)[-1]
        if base_name in excluded_exact or lower_path.endswith(excluded_ext):
            continue

        trimmed = block[:max_lines_per_block]
        if total + len(trimmed) > max_total_lines:
            remain = max_total_lines - total
            if remain <= 0:
                break
            trimmed = trimmed[:remain]

        kept.extend(trimmed)
        total += len(trimmed)
        kept_blocks += 1

        if kept_blocks >= max_blocks or total >= max_total_lines:
            break

    return "\n".join(kept)


def build_text(commit):
    """Build a single text chunk used for vector indexing."""
    diff_summary = clean_diff(commit.get("diff", ""))
    return (
        f"hash={commit.get('hash', '')}; "
        f"author={commit.get('author', '')}; "
        f"date={commit.get('date', '')}; "
        f"message={commit.get('message', '')}; "
        f"diff={diff_summary}"
    )


def extract_changed_files(diff_text):
    """Extract changed file paths from git diff headers."""
    if not diff_text:
        return []

    files = []
    seen = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/") and parts[3].startswith("b/"):
                path = parts[3][2:]
                if path not in seen:
                    seen.add(path)
                    files.append(path)
    return files


def get_file_blame(repo_path, file_path):
    """Return a short blame summary for a file."""
    if Repo is None:
        return ""

    try:
        repo = Repo(repo_path)
    except (NoSuchPathError, InvalidGitRepositoryError):
        return ""

    target = _resolve_file_in_repo(repo_path, file_path)
    if target is None:
        return ""

    try:
        blamed = repo.blame("HEAD", target)
        latest_commit = None
        for commit, _lines in blamed:
            if latest_commit is None or commit.committed_datetime > latest_commit.committed_datetime:
                latest_commit = commit
        if latest_commit is None:
            return ""
        return (
            f"file={target}; "
            f"latest_commit={latest_commit.hexsha}; "
            f"author={latest_commit.author.name}; "
            f"date={latest_commit.committed_datetime.date().isoformat()}"
        )
    except Exception:
        return ""


def get_symbol_or_lines_from_query(query: str) -> dict:
    """Extract file path, optional line number, and optional symbol from query text."""
    text = query or ""
    result = {}

    file_match = re.search(r"([A-Za-z0-9_./\\-]+\.(?:py|js|ts|java|cpp|c|cc|hpp|h))", text)
    if file_match:
        result["file"] = file_match.group(1)

    line_match = re.search(r"\bline\s+(\d+)\b", text, flags=re.IGNORECASE)
    if line_match is None:
        line_match = re.search(r"\bL(\d+)\b", text)
    if line_match is None:
        line_match = re.search(r":(\d+)\b", text)
    if line_match:
        try:
            line_no = int(line_match.group(1))
            if line_no > 0:
                result["line"] = line_no
        except Exception:
            pass

    symbol = None
    patterns = [
        r"\b(?:was|is|for|about)\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:added|introduced|changed)\b",
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(\)",
    ]
    for pattern in patterns:
        flags = re.IGNORECASE if "function" in pattern or "was|is|for|about" in pattern else 0
        m = re.search(pattern, text, flags=flags)
        if m:
            symbol = m.group(1)
            break

    if symbol:
        result["symbol"] = symbol

    return result


def get_blame_for_line(repo_path: str, file_path: str, line_no: int) -> dict:
    """Run blame for a single line and return structured fields."""
    if Repo is None or line_no <= 0:
        return {}
    try:
        repo = Repo(repo_path)
    except (NoSuchPathError, InvalidGitRepositoryError):
        return {}

    target = _resolve_file_in_repo(repo_path, file_path)
    if target is None:
        return {}

    try:
        out = repo.git.blame(
            "-L",
            f"{line_no},{line_no}",
            "--date=short",
            "HEAD",
            "--",
            target,
        )
        line = out.splitlines()[0] if out else ""
        m = re.match(
            r"^([0-9a-f^]{7,40}) \((.+?)\s+(\d{4}-\d{2}-\d{2})(?: [0-9:+\- ]+)?\s+\d+\)\s?(.*)$",
            line,
        )
        if m:
            return {
                "file": target,
                "line": line_no,
                "hash": m.group(1).lstrip("^"),
                "author": m.group(2).strip(),
                "date": m.group(3),
                "line_content": m.group(4).strip(),
            }
        parts = line.split()
        return {
            "file": target,
            "line": line_no,
            "hash": (parts[0].lstrip("^") if parts else ""),
            "author": "",
            "date": "",
            "line_content": line,
        }
    except Exception:
        return {}


def get_introducing_commit(repo_path: str, commit_hash: str) -> dict:
    """Load commit metadata and cleaned diff by hash."""
    if Repo is None or not commit_hash:
        return {}
    try:
        repo = Repo(repo_path)
        commit = repo.commit(commit_hash)
    except Exception:
        return {}

    changed_files = []
    try:
        name_only = repo.git.show("--name-only", "--pretty=format:", commit.hexsha)
        for line in name_only.splitlines():
            s = line.strip()
            if s:
                changed_files.append(s)
    except Exception:
        changed_files = []

    diff_text = ""
    try:
        raw_show = repo.git.show("--pretty=fuller", commit.hexsha)
        diff_text = clean_diff(raw_show or "")
    except Exception:
        diff_text = ""

    return {
        "hash": commit.hexsha,
        "author": commit.author.name,
        "date": commit.committed_datetime.date().isoformat(),
        "message": commit.message.strip(),
        "diff": diff_text,
        "changed_files": changed_files,
    }


def find_symbol_definition(repo_path: str, file_path: str, symbol: str):
    """Find likely definition line (1-based) for a symbol in a file."""
    if not symbol:
        return None

    target = _resolve_file_in_repo(repo_path, file_path)
    if target is None:
        return None

    abs_path = os.path.join(repo_path, target)
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None

    escaped = re.escape(symbol)
    patterns = [
        re.compile(rf"^\s*def\s+{escaped}\s*\("),  # Python function
        re.compile(rf"^\s*class\s+{escaped}\b"),  # Python class
        re.compile(rf"\bfunction\s+{escaped}\b"),  # JS/TS function
        re.compile(rf"\bconst\s+{escaped}\s*=\s*\("),  # JS/TS arrow style
        re.compile(rf"\b{escaped}\s*=\s*function\b"),  # JS/TS function expr
        re.compile(rf"^\s*(?:public|private|protected)?\s*[\w<>\[\],\s*&:]+\b{escaped}\s*\("),  # Java/C++
    ]

    for idx, line in enumerate(lines, start=1):
        for pat in patterns:
            if pat.search(line):
                return idx
    # Fallback: first symbol occurrence in file (call/use site) when definition is not present.
    any_use = re.compile(rf"\b{escaped}\b")
    for idx, line in enumerate(lines, start=1):
        if any_use.search(line):
            return idx
    # Fallback: partial token match (e.g. "hybrid_search" -> "hybrid").
    tokens = [t for t in re.split(r"[_\W]+", symbol) if len(t) >= 4]
    for token in sorted(tokens, key=len, reverse=True):
        token_pat = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
        for idx, line in enumerate(lines, start=1):
            if token_pat.search(line):
                return idx
    return None


def find_symbol_in_repo(repo_path: str, symbol: str, preferred_file: str | None = None):
    """Find symbol in repo; prefer the hinted file, then fallback to other source files."""
    exts = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".cc", ".hpp", ".h"}

    checked = set()
    if preferred_file:
        line_no = find_symbol_definition(repo_path, preferred_file, symbol)
        checked.add(preferred_file)
        if line_no:
            resolved = _resolve_file_in_repo(repo_path, preferred_file) or preferred_file
            return resolved, line_no

    for root, dirs, files in os.walk(repo_path):
        if ".git" in dirs:
            dirs.remove(".git")
        for name in files:
            _, ext = os.path.splitext(name)
            if ext.lower() not in exts:
                continue
            abs_path = os.path.join(root, name)
            try:
                rel_path = os.path.relpath(abs_path, repo_path).replace("\\", "/")
            except Exception:
                continue
            if rel_path in checked:
                continue
            line_no = find_symbol_definition(repo_path, rel_path, symbol)
            if line_no:
                return rel_path, line_no

    return None, None


def get_local_history_window(repo_path: str, file_path: str, n: int = 3):
    """Get nearby file history commits for extra local evidence."""
    if Repo is None or n <= 0:
        return []
    try:
        repo = Repo(repo_path)
    except (NoSuchPathError, InvalidGitRepositoryError):
        return []

    target = _resolve_file_in_repo(repo_path, file_path)
    if target is None:
        return []

    try:
        commits = list(repo.iter_commits(paths=target, max_count=n))
    except Exception:
        return []

    rows = []
    for c in commits:
        rows.append(
            {
                "hash": c.hexsha,
                "date": c.committed_datetime.date().isoformat(),
                "author": c.author.name,
                "message": c.message.strip(),
                "file": target,
            }
        )
    return rows


def get_file_content_at_commit(repo_path: str, commit_hash: str, file_path: str) -> str:
    """Read file content at a specific commit via `git show <hash>:<path>`."""
    if Repo is None or not commit_hash:
        return ""
    try:
        repo = Repo(repo_path)
    except (NoSuchPathError, InvalidGitRepositoryError):
        return ""

    target = _resolve_file_in_repo(repo_path, file_path) or (file_path or "").strip().replace("\\", "/")
    if not target:
        return ""
    try:
        return repo.git.show(f"{commit_hash}:{target}")
    except Exception:
        return ""


def find_introducing_commit_by_predicate(commits, predicate_fn):
    """Find first commit (oldest -> newest) where predicate becomes true."""
    if not commits:
        return None
    for commit_hash in reversed(commits):
        try:
            if predicate_fn(commit_hash):
                return commit_hash
        except Exception:
            continue
    return None


def find_symbol_definition_in_text(text: str, symbol: str):
    """Find likely symbol definition line in provided text content (1-based)."""
    if not text or not symbol:
        return None
    lines = text.splitlines()
    escaped = re.escape(symbol)
    patterns = [
        re.compile(rf"^\s*def\s+{escaped}\s*\("),
        re.compile(rf"^\s*class\s+{escaped}\b"),
        re.compile(rf"\bfunction\s+{escaped}\b"),
        re.compile(rf"\bconst\s+{escaped}\s*=\s*\("),
        re.compile(rf"\b{escaped}\s*=\s*function\b"),
        re.compile(rf"^\s*(?:public|private|protected)?\s*[\w<>\[\],\s*&:]+\b{escaped}\s*\("),
    ]
    for idx, line in enumerate(lines, start=1):
        for pat in patterns:
            if pat.search(line):
                return idx

    any_use = re.compile(rf"\b{escaped}\b")
    for idx, line in enumerate(lines, start=1):
        if any_use.search(line):
            return idx
    return None


def _resolve_file_in_repo(repo_path, file_path):
    normalized = (file_path or "").strip().replace("\\", "/")
    candidates = [normalized, normalized.lstrip("./")]

    if os.path.isabs(normalized):
        try:
            rel = os.path.relpath(normalized, repo_path).replace("\\", "/")
            candidates.append(rel)
        except Exception:
            pass

    for cand in candidates:
        if not cand:
            continue
        abs_path = os.path.join(repo_path, cand)
        if os.path.isfile(abs_path):
            return cand
    return None
