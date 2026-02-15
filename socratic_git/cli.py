"""Socratic Git CLI entrypoint."""

import argparse
from datetime import datetime
import json
import os
import re
import subprocess
import time
from urllib.parse import urlparse

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from socratic_git.miner import (
    InvalidGitRepositoryError,
    Repo,
    extract_commits,
    get_file_blame,
    get_blame_for_line,
    get_introducing_commit,
    get_symbol_or_lines_from_query,
    find_symbol_definition,
    find_symbol_in_repo,
    extract_changed_files,
    clean_diff,
    get_local_history_window,
    get_file_content_at_commit,
    find_introducing_commit_by_predicate,
    find_symbol_definition_in_text,
)
from socratic_git.rag import (
    SentenceTransformer,
    configure_table,
    index_commits,
    lancedb,
    open_existing_table,
    retrieve,
)
from socratic_git.utils import print_evidence, table_name_for_repo


def _generate_answer(*args, **kwargs):
    """Lazy-load LLM backend to avoid MLX init on non-LLM commands."""
    from socratic_git.llm import generate_answer

    return generate_answer(*args, **kwargs)


def _llm_disabled():
    """Allow opt-out in constrained environments without changing default behavior."""
    return os.environ.get("SOCRATIC_SKIP_LLM", "").strip() == "1"


def _tokenize_query(text):
    return [t for t in re.split(r"[^A-Za-z0-9_]+", (text or "").lower()) if t]


def _keyword_index_path(repo_path):
    os.makedirs("data/keyword_index", exist_ok=True)
    return os.path.join("data", "keyword_index", f"{table_name_for_repo(repo_path)}.jsonl")


def has_index(repo_path):
    """Return whether vector/keyword index exists for the repo."""
    table = table_name_for_repo(repo_path)
    vector_path = os.path.join("data", "lancedb", f"{table}.lance")
    keyword_path = _keyword_index_path(repo_path)
    vector_exists = os.path.isdir(vector_path)
    keyword_exists = os.path.isfile(keyword_path)
    return vector_exists, keyword_exists


def _build_keyword_rows(commits):
    rows = []
    for c in commits:
        text = ((c.get("message", "") or "") + "\n" + clean_diff(c.get("diff", "") or "")).strip()
        rows.append(
            {
                "hash": c.get("hash", ""),
                "author": c.get("author", ""),
                "date": c.get("date", ""),
                "message": c.get("message", ""),
                "text": text,
                "files": extract_changed_files(c.get("diff", "")),
            }
        )
    return rows


def write_keyword_index(repo_path, commits):
    """Write keyword index JSONL for offline retrieval."""
    path = _keyword_index_path(repo_path)
    rows = _build_keyword_rows(commits)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path, len(rows)


def _load_keyword_index(repo_path):
    path = _keyword_index_path(repo_path)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def keyword_retrieve(repo_path, query, topk=3, n=200):
    """Simple offline retrieval using keyword overlap on indexed text."""
    rows = _load_keyword_index(repo_path)
    if not rows:
        commits = extract_commits(repo_path, n=n)
        if not commits:
            return []
        rows = _build_keyword_rows(commits)

    q_tokens = _tokenize_query(query)
    if not q_tokens:
        q_tokens = ["why"]

    scored = []
    for r in rows:
        text = (r.get("text", "") or "").lower()
        score = sum(1 for tok in q_tokens if tok in text)
        scored.append((score, r))

    # Prefer high score; keep recent order for ties.
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, topk)]
    results = []
    for _, r in top:
        results.append(
            {
                "commit": {
                    "hash": r.get("hash", ""),
                    "author": r.get("author", ""),
                    "date": r.get("date", ""),
                    "message": r.get("message", ""),
                    "diff": "",
                },
                "text": r.get("text", ""),
            }
        )
    return results


def resolve_contexts(repo_path, question, topk, retrieval_mode, command_label):
    """Retrieve contexts with vector mode + automatic keyword fallback."""
    topk = max(1, topk)
    mode = retrieval_mode or "vector"
    if mode == "keyword":
        contexts = keyword_retrieve(repo_path, question, topk=topk)
        return contexts, "keyword"

    if lancedb is None or SentenceTransformer is None:
        print("Warning: Embedding model unavailable, falling back to keyword retrieval.")
        contexts = keyword_retrieve(repo_path, question, topk=topk)
        return contexts, "keyword"

    try:
        vector_exists, keyword_exists = has_index(repo_path)
        if vector_exists or keyword_exists:
            if vector_exists and keyword_exists:
                kind = "vector|keyword"
            elif vector_exists:
                kind = "vector"
            else:
                kind = "keyword"
            print(f"[{command_label}] found existing index ({kind})")
        else:
            print(f"[{command_label}] index not found, auto-indexing latest 50 commits...")
            t0 = time.perf_counter()
            commits = extract_commits(repo_path, n=50)
            if not commits:
                return [], "vector"
            index_commits(commits)
            t1 = time.perf_counter()
            print(f"[{command_label}] auto-index done in {t1 - t0:.2f}s")

        contexts = retrieve(question, topk=topk)
        return contexts, "vector"
    except Exception:
        print("Warning: Embedding model unavailable, falling back to keyword retrieval.")
        contexts = keyword_retrieve(repo_path, question, topk=topk)
        return contexts, "keyword"


def run_cmd_at_commit(repo, commit_hash, cmd):
    """Checkout commit in detached mode, run command, and restore original HEAD."""
    original_ref = None
    if repo.head.is_detached:
        original_ref = repo.head.commit.hexsha
    else:
        original_ref = repo.active_branch.name

    try:
        repo.git.checkout("--detach", commit_hash)
        proc = subprocess.run(
            cmd,
            cwd=repo.working_tree_dir,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        out_lines = (proc.stdout or "").splitlines()[-200:]
        err_lines = (proc.stderr or "").splitlines()[-200:]
        stdout_tail = "\n".join(out_lines)[-8000:]
        stderr_tail = "\n".join(err_lines)[-8000:]
        return {
            "returncode": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    finally:
        try:
            repo.git.checkout(original_ref)
        except Exception:
            pass


def bisect_search(commits_range, is_bad_fn, max_steps=30):
    """Binary search first bad commit in ordered commits_range (oldest->newest)."""
    steps = []
    tested = {}
    left, right = 0, len(commits_range) - 1
    first_bad = None

    while left <= right and len(steps) < max_steps:
        mid = (left + right) // 2
        commit_hash = commits_range[mid]
        if commit_hash in tested:
            result = tested[commit_hash]
        else:
            result = is_bad_fn(commit_hash)
            tested[commit_hash] = result

        is_bad = bool(result.get("is_bad", False))
        steps.append(
            {
                "commit": commit_hash,
                "status": "bad" if is_bad else "good",
                "returncode": result.get("returncode", -1),
            }
        )
        if is_bad:
            first_bad = commit_hash
            right = mid - 1
        else:
            left = mid + 1

    return first_bad, steps, tested


def parse_github_remote(repo_path):
    """Parse GitHub owner/repo from origin url."""
    try:
        repo = Repo(repo_path)
        origin = next((r for r in repo.remotes if r.name == "origin"), None)
        if origin is None:
            return None
        url = origin.url.strip()
    except Exception:
        return None

    # https://github.com/<owner>/<repo>.git
    if url.startswith("http://") or url.startswith("https://"):
        try:
            parsed = urlparse(url)
            if (parsed.hostname or "").lower() != "github.com":
                return None
            path = (parsed.path or "").strip("/")
            if path.endswith(".git"):
                path = path[:-4]
            parts = path.split("/")
            if len(parts) >= 2:
                return {"owner": parts[0], "repo": parts[1]}
        except Exception:
            return None

    # git@github.com:<owner>/<repo>.git
    m = re.match(r"^git@github\.com:([^/]+)/(.+?)(?:\.git)?$", url)
    if m:
        return {"owner": m.group(1), "repo": m.group(2)}

    return None


def extract_pr_issue_refs(text):
    """Extract PR refs from text; issue refs kept optional/unknown."""
    src = text or ""
    prs = set()
    issues = set()
    for pat in [
        r"Merge pull request\s+#(\d+)",
        r"\bPR\s*#(\d+)\b",
        r"\(#(\d+)\)",
        r"\B#(\d+)\b",
    ]:
        for m in re.finditer(pat, src, flags=re.IGNORECASE):
            try:
                prs.add(int(m.group(1)))
            except Exception:
                pass
    return {"prs": sorted(prs), "issues": sorted(issues)}


def get_commit_message_body(repo_path, commit_hash):
    """Get full commit message body."""
    if not commit_hash:
        return ""
    try:
        repo = Repo(repo_path)
        return repo.git.show("-s", "--format=%B", commit_hash)
    except Exception:
        return ""


def fetch_github_pr(repo_path, owner, repo_name, pr_number, token=None):
    """Fetch PR details with 7-day local cache."""
    cache_dir = os.path.join(repo_path, ".socratic_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"github_pr_{owner}_{repo_name}_{pr_number}.json")
    now = time.time()
    ttl = 7 * 24 * 3600

    if os.path.exists(cache_path):
        try:
            st = os.stat(cache_path)
            if now - st.st_mtime <= ttl:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
        except Exception:
            pass

    if requests is None:
        return None

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls/{pr_number}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None

    data = {
        "number": pr_number,
        "title": payload.get("title", ""),
        "body": (payload.get("body", "") or "")[:1200],
        "author": (payload.get("user") or {}).get("login", ""),
        "created_at": payload.get("created_at", ""),
        "merged_at": payload.get("merged_at", ""),
        "html_url": payload.get("html_url", ""),
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return data


def build_github_pr_evidence(repo_path, commit_hash, disabled=False):
    """Build optional PR evidence block from commit refs."""
    if disabled or not commit_hash:
        return None
    # Always ensure cache directory exists when enhancement is enabled.
    os.makedirs(os.path.join(repo_path, ".socratic_cache"), exist_ok=True)
    gh = parse_github_remote(repo_path)
    if gh is None:
        return None

    full_msg = get_commit_message_body(repo_path, commit_hash)
    refs = extract_pr_issue_refs(full_msg)
    prs = refs.get("prs", [])
    if not prs:
        return None

    token = os.environ.get("GITHUB_TOKEN", "").strip() or None
    pr_no = prs[0]
    if requests is None:
        return {
            "available": False,
            "pr_number": pr_no,
            "text": "PR evidence unavailable (requests not installed; pip install requests)",
            "title": "",
            "body": "",
            "url": "",
        }
    pr = fetch_github_pr(repo_path, gh["owner"], gh["repo"], pr_no, token=token)
    if pr is None:
        return {
            "available": False,
            "pr_number": pr_no,
            "text": "PR evidence unavailable (rate limit or not found)",
            "title": "",
            "body": "",
            "url": "",
        }

    body_snippet = (pr.get("body", "") or "").strip()
    text = (
        "## GitHub PR Evidence (Optional)\n"
        f"- pr: #{pr_no}\n"
        f"- title: {pr.get('title', '')}\n"
        f"- author: {pr.get('author', '')}\n"
        f"- created/merged: {pr.get('created_at', '')} / {pr.get('merged_at', '')}\n"
        f"- url: {pr.get('html_url', '')}\n"
        f"- body_snippet:\n{body_snippet if body_snippet else '(empty)'}"
    )
    return {
        "available": True,
        "pr_number": pr_no,
        "text": text,
        "title": pr.get("title", ""),
        "body": pr.get("body", ""),
        "url": pr.get("html_url", ""),
    }


def main():
    """Run CLI with index/ask subcommands."""
    parser = argparse.ArgumentParser(description="Socratic Git CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_index = subparsers.add_parser("index", help="Build/overwrite vector index")
    p_index.add_argument("--repo", required=True, help="Path to local git repository")
    p_index.add_argument("--n", type=int, default=200, help="Number of latest commits to index")
    p_index.add_argument("--mode", choices=["vector", "keyword"], default="vector", help="Index mode")

    p_ask = subparsers.add_parser("ask", help="Ask question against indexed commits")
    p_ask.add_argument("--repo", required=True, help="Path to local git repository")
    p_ask.add_argument("--q", required=True, help="Question")
    p_ask.add_argument("--topk", type=int, default=3, help="Number of contexts to retrieve")
    p_ask.add_argument("--retrieval", choices=["vector", "keyword"], default="vector", help="Retrieval mode")

    p_trace = subparsers.add_parser("trace", help="Generate evidence-driven markdown trace report")
    p_trace.add_argument("--repo", required=True, help="Path to local git repository")
    p_trace.add_argument("--q", required=True, help="Question")
    p_trace.add_argument("--topk", type=int, default=3, help="Number of contexts to retrieve")
    p_trace.add_argument("--retrieval", choices=["vector", "keyword"], default="vector", help="Retrieval mode")
    p_trace.add_argument("--out", default="", help="Output markdown path")
    p_trace.add_argument("--no-github", action="store_true", help="Disable optional GitHub PR evidence fetch")

    p_regress = subparsers.add_parser("regress", help="Find introducing commit for regression/bug signal")
    p_regress.add_argument("--repo", required=True, help="Path to local git repository")
    p_regress.add_argument("--file", required=True, help="Target file path in repo")
    p_regress.add_argument("--pattern", default="", help="Text or regex pattern to match")
    p_regress.add_argument("--symbol", default="", help="Symbol/function/class name")
    p_regress.add_argument("--out", default="", help="Output markdown path")
    p_regress.add_argument("--max", type=int, default=2000, help="Max commits to scan for the file")
    p_regress.add_argument("--no-github", action="store_true", help="Disable optional GitHub PR evidence fetch")

    p_bisect = subparsers.add_parser("bisect", help="Command-driven first bad commit search")
    p_bisect.add_argument("--repo", required=True, help="Path to local git repository")
    p_bisect.add_argument("--good", required=True, help="Known good commit hash")
    p_bisect.add_argument("--bad", required=True, help="Known bad commit hash")
    p_bisect.add_argument("--cmd", required=True, help="Shell command to run at each commit")
    p_bisect.add_argument("--out", default="", help="Output markdown path")
    p_bisect.add_argument("--max-steps", type=int, default=30, help="Maximum bisect steps")
    p_bisect.add_argument("--no-github", action="store_true", help="Disable optional GitHub PR evidence fetch")

    args = parser.parse_args()
    repo_path = args.repo

    try:
        if not os.path.isdir(repo_path):
            print(f"Error: repo path does not exist: {repo_path}")
            return
        if Repo is None:
            print("Error: missing dependency 'gitpython'. Please install requirements.")
            return
        try:
            Repo(repo_path)
        except InvalidGitRepositoryError:
            print(f"Error: not a git repository: {repo_path}")
            return

        table_name = table_name_for_repo(repo_path)
        configure_table(table_name)

        if args.command == "index":
            print(f"[index] repo={repo_path}")
            t0 = time.perf_counter()
            commits = extract_commits(repo_path, n=args.n)
            print(f"[index] extracted commits: {len(commits)}")
            if not commits:
                print("Error: failed to read commits from repo.")
                return
            if args.mode == "keyword":
                kpath, kcount = write_keyword_index(repo_path, commits)
                t1 = time.perf_counter()
                print(f"[index] keyword indexed commits: {kcount}")
                print(f"[index] keyword index path: {kpath}")
                print(f"[index] done in {t1 - t0:.2f}s")
                return
            try:
                records = index_commits(commits)
                t1 = time.perf_counter()
                print(f"[index] indexed commits: {len(records)}")
                print(f"[index] done in {t1 - t0:.2f}s")
                return
            except Exception as exc:
                print(f"Warning: Embedding model unavailable, falling back to keyword index. ({exc})")
                kpath, kcount = write_keyword_index(repo_path, commits)
                t1 = time.perf_counter()
                print(f"[index] keyword indexed commits: {kcount}")
                print(f"[index] keyword index path: {kpath}")
                print(f"[index] done in {t1 - t0:.2f}s")
                return

        if args.command == "trace":
            question = args.q
            topk = max(1, args.topk)
            contexts, _used_retrieval = resolve_contexts(
                repo_path,
                question,
                topk=topk,
                retrieval_mode=getattr(args, "retrieval", "vector"),
                command_label="trace",
            )
            if not contexts:
                print("Error: no related commits retrieved.")
                return

            target_info = get_symbol_or_lines_from_query(question)
            detected_file = target_info.get("file")
            detected_line = target_info.get("line")
            detected_symbol = target_info.get("symbol")
            resolved_line = None
            blame_info = ""
            introducing_commit = {}
            intro_files = []
            intro_msg_preview = ""
            intro_diff_preview = ""
            local_history_window = []
            github_pr_evidence = None

            if detected_file and not detected_line and detected_symbol:
                resolved_line = find_symbol_definition(repo_path, detected_file, detected_symbol)
                if resolved_line:
                    detected_line = resolved_line
                else:
                    fallback_file, fallback_line = find_symbol_in_repo(
                        repo_path, detected_symbol, preferred_file=detected_file
                    )
                    if fallback_file and fallback_line:
                        detected_file = fallback_file
                        resolved_line = fallback_line
                        detected_line = fallback_line

            if detected_file and detected_line:
                blame_dict = get_blame_for_line(repo_path, detected_file, detected_line)
                if blame_dict:
                    blame_info = (
                        f"file={blame_dict.get('file', '')}; "
                        f"line={blame_dict.get('line', '')}; "
                        f"commit={blame_dict.get('hash', '')}; "
                        f"author={blame_dict.get('author', '')}; "
                        f"date={blame_dict.get('date', '')}; "
                        f"line_content={blame_dict.get('line_content', '')}"
                    )
                    introducing_commit = get_introducing_commit(repo_path, blame_dict.get("hash", ""))
            elif detected_file:
                blame_info = get_file_blame(repo_path, detected_file)

            if introducing_commit:
                intro_msg_lines = introducing_commit.get("message", "").splitlines()[:10]
                intro_msg_preview = "\n".join(intro_msg_lines)
                intro_diff_clean = clean_diff(introducing_commit.get("diff", ""))
                intro_diff_lines = intro_diff_clean.splitlines()[:120]
                intro_diff_preview = "\n".join(intro_diff_lines)
                intro_files = introducing_commit.get("changed_files", []) or extract_changed_files(
                    introducing_commit.get("diff", "")
                )
                github_pr_evidence = build_github_pr_evidence(
                    repo_path,
                    introducing_commit.get("hash", ""),
                    disabled=getattr(args, "no_github", False),
                )

            if detected_file:
                local_history_window = get_local_history_window(repo_path, detected_file, n=5)

            used_hashes = []
            seen_hash = set()

            def add_hash(h):
                if h and h not in seen_hash:
                    seen_hash.add(h)
                    used_hashes.append(h)

            add_hash(introducing_commit.get("hash", ""))
            for row in local_history_window:
                add_hash(row.get("hash", ""))
            for ctx in contexts:
                add_hash(ctx.get("commit", {}).get("hash", ""))

            llm_contexts = contexts
            if introducing_commit:
                introducing_details_text = (
                    "=== Introducing Commit Details ===\n"
                    f"hash: {introducing_commit.get('hash', '')}\n"
                    f"author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}\n"
                    f"message:\n{intro_msg_preview}\n"
                    f"changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}\n"
                    f"diff_snippet: {intro_diff_preview if intro_diff_preview else '(unavailable)'}"
                )
                llm_contexts = [
                    {
                        "commit": introducing_commit,
                        "text": introducing_details_text,
                    }
                ] + llm_contexts
            if github_pr_evidence:
                llm_contexts = [
                    {
                        "commit": {
                            "hash": f"pr-{github_pr_evidence.get('pr_number', '')}",
                            "author": "github",
                            "date": "",
                            "message": "GitHub PR Evidence",
                            "diff": "",
                        },
                        "text": github_pr_evidence.get("text", ""),
                    }
                ] + llm_contexts
            if local_history_window:
                history_lines = []
                for row in local_history_window:
                    msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                    history_lines.append(f"{row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
                history_text = "=== Local History Window (file) ===\n" + "\n".join(history_lines)
                llm_contexts = llm_contexts + [
                    {
                        "commit": {
                            "hash": "local-history",
                            "author": "git log",
                            "date": "",
                            "message": "Recent file-local history window",
                            "diff": "",
                        },
                        "text": history_text,
                    }
                ]

            intro_text_for_judgement = (
                ((introducing_commit.get("message", "") or "") + "\n" + (introducing_commit.get("diff", "") or ""))
                .lower()
            )
            if github_pr_evidence and github_pr_evidence.get("available"):
                intro_text_for_judgement += "\n" + (github_pr_evidence.get("title", "") or "").lower()
                intro_text_for_judgement += "\n" + (github_pr_evidence.get("body", "") or "").lower()
            motive_keywords = ("because", "so that", "in order to", "to improve", "motivation", "reason", "why")
            has_explicit_motive = any(k in intro_text_for_judgement for k in motive_keywords)

            if has_explicit_motive:
                trace_question = (
                    f"{question}\n"
                    f"Evidence hashes you may use: {', '.join(used_hashes) if used_hashes else 'none'}\n"
                    "Answer format must be:\n"
                    "Evidence Hashes: <comma-separated hashes or none>\n"
                    "Conclusion: <one short paragraph, cite concrete commit evidence>\n"
                    "Do not use 可能/大概/推测."
                )
            else:
                trace_question = (
                    f"{question}\n"
                    f"Evidence hashes you may use: {', '.join(used_hashes) if used_hashes else 'none'}\n"
                    "Motive evidence is insufficient in introducing commit message/diff.\n"
                    "Return EXACTLY these two lines and nothing else:\n"
                    "Evidence Hashes: <comma-separated hashes or none>\n"
                    "Conclusion: I don't know (insufficient evidence in commit message/diff)."
                )
            if _llm_disabled():
                answer = "Evidence Hashes: none\nConclusion: I don't know (insufficient evidence in commit message/diff)."
            else:
                try:
                    answer = _generate_answer(trace_question, llm_contexts)
                except Exception as exc:
                    msg = str(exc)
                    if "No such file or directory" in msg or "not found" in msg.lower():
                        print(f"Error: model may be missing/not downloaded. Details: {exc}")
                    else:
                        print(f"Error: failed to run local model. Details: {exc}")
                    return

            if args.out:
                out_path = args.out
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("outputs", f"socratic_report_{ts}.md")
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            limitations = []
            if not detected_file:
                limitations.append("- Query did not include a concrete file path; target detection may be weak.")
            if detected_symbol and not detected_line and not resolved_line:
                limitations.append("- Symbol was detected but could not be resolved to a line number.")
            if detected_file and not introducing_commit:
                limitations.append("- Introducing commit not found; report falls back to file-level evidence.")
            if introducing_commit and not introducing_commit.get("diff", "").strip():
                limitations.append("- Introducing commit diff is empty or unavailable.")
            generic_count = 0
            for ctx in contexts:
                first_line = (ctx.get("commit", {}).get("message", "") or "").splitlines()[0].lower()
                if first_line.startswith("update") or first_line.startswith("updating") or first_line.startswith("merge"):
                    generic_count += 1
            if generic_count >= max(1, len(contexts) - 1):
                limitations.append("- Retrieved commit messages are generic, so why-level intent may be under-specified.")
            if "我不知道" in answer:
                limitations.append("- Evidence was insufficient for a confident why-explanation.")
            if not limitations:
                limitations.append("- No major limitations detected for this trace run.")

            lines = [
                "# Socratic Git Trace Report",
                "",
                "## Question",
                question,
                "",
                "## Detected Target",
                f"- file: {detected_file or '(not detected)'}",
                f"- symbol: {detected_symbol or '(not detected)'}",
                f"- line: {detected_line if detected_line else '(not detected)'}",
                f"- resolved_line: {resolved_line if resolved_line else '(none)'}",
                "",
                "## Introducing Commit",
            ]

            if introducing_commit:
                intro_msg = introducing_commit.get("message", "").splitlines()[0] if introducing_commit.get("message") else ""
                lines.extend(
                    [
                        f"- hash: {introducing_commit.get('hash', '')}",
                        f"- author: {introducing_commit.get('author', '')}",
                        f"- date: {introducing_commit.get('date', '')}",
                        f"- message: {intro_msg}",
                    ]
                )
            else:
                lines.append("- not found")

            lines.extend(
                [
                    "",
                    "## Introducing Commit Details",
                ]
            )

            if introducing_commit:
                lines.extend(
                    [
                        f"- hash: {introducing_commit.get('hash', '')}",
                        f"- author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}",
                        "- message (first 10 lines):",
                        "```text",
                        intro_msg_preview or "(empty)",
                        "```",
                        f"- changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}",
                        (
                            "- diff_snippet: (unavailable)"
                            if not intro_diff_preview
                            else "- diff_snippet (clean_diff + first 120 lines):"
                        ),
                    ]
                )
                if intro_diff_preview:
                    lines.extend(
                        [
                            "```diff",
                            intro_diff_preview,
                            "```",
                        ]
                    )
                else:
                    lines.extend(
                        [
                            "```diff",
                            "(unavailable)",
                            "```",
                        ]
                    )
            else:
                lines.append("- not available")

            lines.extend(
                [
                    "",
                    "## Local History Window",
                ]
            )
            if local_history_window:
                for row in local_history_window:
                    msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                    lines.append(f"- {row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
            else:
                lines.append("- not available")

            lines.extend(
                [
                    "",
                    "## Retrieved Commits (TopK)",
                ]
            )
            for i, ctx in enumerate(contexts[:topk], start=1):
                c = ctx.get("commit", {})
                first = (c.get("message", "") or "").splitlines()[0]
                files = extract_changed_files(c.get("diff", ""))
                lines.extend(
                    [
                        f"- [{i}] {c.get('hash', '')} {c.get('date', '')} {c.get('author', '')}",
                        f"  - message: {first}",
                        f"  - files: {', '.join(files) if files else '(none)'}",
                    ]
                )

            lines.extend(
                [
                    "",
                    "## Evidence Summary",
                ]
            )
            if used_hashes:
                for h in used_hashes:
                    lines.append(f"- {h}")
            else:
                lines.append("- (none)")
            if github_pr_evidence and github_pr_evidence.get("available") and github_pr_evidence.get("url"):
                lines.append(f"- pr_url: {github_pr_evidence.get('url')}")

            if github_pr_evidence:
                lines.extend(
                    [
                        "",
                        "## GitHub PR Evidence (Optional)",
                    ]
                )
                if github_pr_evidence.get("available"):
                    lines.extend(github_pr_evidence.get("text", "").splitlines()[1:])
                else:
                    lines.append(f"- pr: #{github_pr_evidence.get('pr_number', '')}")
                    lines.append("- PR evidence unavailable (rate limit or not found)")

            lines.extend(
                [
                    "",
                    "## Answer (Evidence-Driven)",
                    answer,
                    "",
                    "## Limitations",
                ]
            )
            lines.extend(limitations)
            lines.append("")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"report saved to {out_path}")
            return

        if args.command == "regress":
            file_path = args.file
            pattern = (args.pattern or "").strip()
            symbol = (args.symbol or "").strip()
            max_commits = max(1, args.max)
            if not pattern and not symbol:
                print("Error: provide --pattern or --symbol (or both).")
                return

            repo = Repo(repo_path)
            file_commits = list(repo.iter_commits(paths=file_path, max_count=max_commits))
            commit_hashes = [c.hexsha for c in file_commits]
            if not commit_hashes:
                print(f"Error: no commits found for file: {file_path}")
                return

            pattern_re = None
            if pattern:
                try:
                    pattern_re = re.compile(pattern)
                except Exception:
                    pattern_re = None

            def predicate_fn(commit_hash):
                text = get_file_content_at_commit(repo_path, commit_hash, file_path)
                if not text:
                    return False
                ok_pattern = True
                ok_symbol = True
                if pattern:
                    if pattern_re is not None:
                        ok_pattern = bool(pattern_re.search(text))
                    else:
                        ok_pattern = pattern in text
                if symbol:
                    ok_symbol = find_symbol_definition_in_text(text, symbol) is not None
                return ok_pattern and ok_symbol

            introducing_hash = find_introducing_commit_by_predicate(commit_hashes, predicate_fn)
            if not introducing_hash:
                print("Error: no introducing commit matched the predicate.")
                return

            introducing_commit = get_introducing_commit(repo_path, introducing_hash)
            intro_msg_lines = introducing_commit.get("message", "").splitlines()[:10]
            intro_msg_preview = "\n".join(intro_msg_lines)
            intro_diff = clean_diff(introducing_commit.get("diff", ""))
            intro_diff_preview = "\n".join(intro_diff.splitlines()[:120])
            intro_files = introducing_commit.get("changed_files", []) or extract_changed_files(
                introducing_commit.get("diff", "")
            )
            github_pr_evidence = build_github_pr_evidence(
                repo_path, introducing_hash, disabled=getattr(args, "no_github", False)
            )

            content_at_intro = get_file_content_at_commit(repo_path, introducing_hash, file_path)
            snippet = "(unavailable)"
            if content_at_intro:
                lines = content_at_intro.splitlines()
                match_line = None
                if pattern:
                    for i, line in enumerate(lines, start=1):
                        matched = bool(pattern_re.search(line)) if pattern_re is not None else (pattern in line)
                        if matched:
                            match_line = i
                            break
                if match_line is None and symbol:
                    match_line = find_symbol_definition_in_text(content_at_intro, symbol)
                if match_line is not None:
                    start = max(1, match_line - 5)
                    end = min(len(lines), match_line + 5)
                    chunk = []
                    for ln in range(start, end + 1):
                        chunk.append(f"{ln:>4}: {lines[ln - 1]}")
                    snippet = "\n".join(chunk)

            idx = commit_hashes.index(introducing_hash)
            left = max(0, idx - 2)
            right = min(len(file_commits), idx + 3)
            local_window = file_commits[left:right]

            used_hashes = []
            seen_hash = set()
            for h in [introducing_hash] + [c.hexsha for c in local_window]:
                if h and h not in seen_hash:
                    seen_hash.add(h)
                    used_hashes.append(h)

            motive_text = ((introducing_commit.get("message", "") or "") + "\n" + (introducing_commit.get("diff", "") or "")).lower()
            if github_pr_evidence and github_pr_evidence.get("available"):
                motive_text += "\n" + (github_pr_evidence.get("title", "") or "").lower()
                motive_text += "\n" + (github_pr_evidence.get("body", "") or "").lower()
            motive_keywords = ("because", "so that", "in order to", "to improve", "motivation", "reason", "why")
            has_explicit_motive = any(k in motive_text for k in motive_keywords)

            llm_contexts = [
                {
                    "commit": introducing_commit,
                    "text": (
                        "INTRODUCING COMMIT DETAILS\n"
                        f"hash={introducing_commit.get('hash', '')}\n"
                        f"message={introducing_commit.get('message', '')}\n"
                        f"changed_files={', '.join(intro_files) if intro_files else '(unavailable)'}\n"
                        f"diff_snippet={intro_diff_preview if intro_diff_preview else '(unavailable)'}\n"
                        f"matched_snippet=\n{snippet}"
                    ),
                }
            ]
            if local_window:
                hist_lines = []
                for c in local_window:
                    msg = c.message.strip().splitlines()[0] if c.message else ""
                    hist_lines.append(f"{c.hexsha} {c.committed_datetime.date().isoformat()} {msg}")
                llm_contexts.append(
                    {
                        "commit": {
                            "hash": "local-window",
                            "author": "git log",
                            "date": "",
                            "message": "Local history around introducing commit",
                            "diff": "",
                        },
                        "text": "\n".join(hist_lines),
                    }
                )
            if github_pr_evidence:
                llm_contexts.append(
                    {
                        "commit": {
                            "hash": f"pr-{github_pr_evidence.get('pr_number', '')}",
                            "author": "github",
                            "date": "",
                            "message": "GitHub PR Evidence",
                            "diff": "",
                        },
                        "text": github_pr_evidence.get("text", ""),
                    }
                )

            snippet_summary = ""
            if snippet and snippet != "(unavailable)":
                for raw in snippet.splitlines():
                    if ":" in raw:
                        snippet_summary = raw.split(":", 1)[1].strip()
                        if snippet_summary:
                            break
            if not snippet_summary:
                snippet_summary = "(snippet unavailable)"

            findings_text = (
                f"{pattern or symbol} was introduced in commit {introducing_commit.get('hash', '')} "
                f"on {introducing_commit.get('date', '')} by {introducing_commit.get('author', '')} "
                f"in {file_path}. Key snippet: {snippet_summary}"
            )

            motive_text_out = "Motive is not explicitly stated in commit message/diff."
            if has_explicit_motive:
                motive_question = (
                    f"Why was {pattern or symbol} introduced in {file_path}?\n"
                    f"Evidence hashes you may use: {', '.join(used_hashes)}\n"
                    "Answer in plain text with concrete evidence from commit message/diff. "
                    "If motive is not explicit, output exactly: Motive is not explicitly stated in commit message/diff."
                )
                if _llm_disabled():
                    motive_raw = "Motive is not explicitly stated in commit message/diff."
                else:
                    try:
                        motive_raw = _generate_answer(motive_question, llm_contexts)
                    except Exception as exc:
                        msg = str(exc)
                        if "No such file or directory" in msg or "not found" in msg.lower():
                            print(f"Error: model may be missing/not downloaded. Details: {exc}")
                        else:
                            print(f"Error: failed to run local model. Details: {exc}")
                        return

                def sanitize_llm_output(text):
                    cleaned = text or ""
                    if "<|im_" in cleaned:
                        cleaned = cleaned.split("<|im_", 1)[0]
                    lines_local = cleaned.splitlines()
                    kept = []
                    in_code = False
                    for ln in lines_local:
                        if ln.strip().startswith("```"):
                            in_code = not in_code
                            continue
                        if not in_code:
                            kept.append(ln)
                    cleaned = "\n".join(kept).strip()
                    cleaned = cleaned.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
                    return cleaned

                motive_text_out = sanitize_llm_output(motive_raw)
                if not motive_text_out:
                    motive_text_out = "Motive is not explicitly stated in commit message/diff."
                low = motive_text_out.lower()
                if "i don't know" in low or "insufficient evidence" in low:
                    motive_text_out = "Motive is not explicitly stated in commit message/diff."

            if args.out:
                out_path = args.out
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("outputs", f"regression_report_{ts}.md")
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            limitations = []
            if not intro_files:
                limitations.append("- changed_files unavailable from introducing commit.")
            if not intro_diff_preview:
                limitations.append("- diff_snippet unavailable for introducing commit.")
            if snippet == "(unavailable)":
                limitations.append("- Could not extract matched evidence snippet from file content at introducing commit.")
            if not has_explicit_motive:
                limitations.append("- Commit message/diff do not explicitly explain motive.")
            if not limitations:
                limitations.append("- No major limitations detected for this regression run.")

            target_desc = pattern if pattern else symbol
            lines = [
                "# Socratic Git Regression Report",
                "",
                "## Question",
                f"When was {target_desc} introduced in {file_path}?",
                "",
                "## Introducing Commit",
                f"- hash: {introducing_commit.get('hash', '')}",
                f"- author: {introducing_commit.get('author', '')}",
                f"- date: {introducing_commit.get('date', '')}",
                f"- message: {(introducing_commit.get('message', '') or '').splitlines()[0] if introducing_commit.get('message') else ''}",
                "",
                "## Evidence Snippets",
                f"- changed_files: {', '.join(intro_files) if intro_files else '(unavailable)'}",
                "- matched snippet (+/- 5 lines):",
                "```text",
                snippet,
                "```",
                "- diff_snippet:",
                "```diff",
                intro_diff_preview if intro_diff_preview else "(unavailable)",
                "```",
                "",
                "## Local History Window",
            ]

            for c in local_window:
                msg = c.message.strip().splitlines()[0] if c.message else ""
                lines.append(f"- {c.hexsha} {c.committed_datetime.date().isoformat()} {msg}")

            lines.extend(
                [
                    "",
                "## Answer (Evidence-Driven)",
                "Findings (Deterministic):",
                findings_text,
                "",
                "Motive (Evidence-Driven, Optional):",
                motive_text_out,
                "",
                "## Limitations",
            ]
            )
            if github_pr_evidence:
                lines.extend(
                    [
                        "",
                        "## GitHub PR Evidence (Optional)",
                    ]
                )
                if github_pr_evidence.get("available"):
                    lines.extend(github_pr_evidence.get("text", "").splitlines()[1:])
                else:
                    lines.append(f"- pr: #{github_pr_evidence.get('pr_number', '')}")
                    lines.append("- PR evidence unavailable (rate limit or not found)")
            lines.extend(limitations)
            lines.append("")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"report saved to {out_path}")
            return

        if args.command == "bisect":
            repo = Repo(repo_path)
            good_hash = args.good
            bad_hash = args.bad
            cmd = args.cmd
            max_steps = max(1, args.max_steps)

            if repo.is_dirty(untracked_files=True):
                print("Error: repository has uncommitted changes. Please commit/stash before bisect.")
                return

            try:
                range_hashes = repo.git.rev_list("--ancestry-path", f"{good_hash}..{bad_hash}").splitlines()
            except Exception as exc:
                print(f"Error: failed to build commit range from good/bad. Details: {exc}")
                return

            if not range_hashes:
                print("Error: no commits found between --good and --bad.")
                return

            commits_range = list(reversed(range_hashes))  # oldest -> newest, excludes good, includes bad

            # Run boundary checks and enforce bisect preconditions.
            good_run = run_cmd_at_commit(repo, good_hash, cmd)
            bad_run = run_cmd_at_commit(repo, bad_hash, cmd)
            if good_run.get("returncode", 1) != 0:
                print("Provided good commit does not pass; bisect precondition violated.")
                return
            if bad_run.get("returncode", 0) == 0:
                print("Provided bad commit does not fail; bisect precondition violated.")
                return

            def is_bad_fn(commit_hash):
                run = run_cmd_at_commit(repo, commit_hash, cmd)
                return {
                    "is_bad": run.get("returncode", 1) != 0,
                    "returncode": run.get("returncode", -1),
                    "stdout_tail": run.get("stdout_tail", ""),
                    "stderr_tail": run.get("stderr_tail", ""),
                }

            first_bad, steps, tested = bisect_search(commits_range, is_bad_fn, max_steps=max_steps)
            boundary_steps = [
                {"commit": good_hash, "status": "good", "returncode": good_run.get("returncode", -1), "note": "boundary=good"},
                {"commit": bad_hash, "status": "bad", "returncode": bad_run.get("returncode", -1), "note": "boundary=bad"},
            ]
            steps_all = boundary_steps + [
                {
                    "commit": s.get("commit", ""),
                    "status": s.get("status", ""),
                    "returncode": s.get("returncode", -1),
                    "note": "mid",
                }
                for s in steps
            ]
            first_bad_commit = {}
            changed_files = []
            diff_preview = ""
            stdout_tail = "(empty)"
            stderr_tail = "(empty)"
            motive = "Motive is not explicitly stated in commit message/diff."
            if first_bad:
                first_bad_commit = get_introducing_commit(repo_path, first_bad)
                changed_files = first_bad_commit.get("changed_files", []) or extract_changed_files(first_bad_commit.get("diff", ""))
                diff_preview = "\n".join((first_bad_commit.get("diff", "") or "").splitlines()[:120])
                fail_log = tested.get(first_bad, {})
                stdout_tail = fail_log.get("stdout_tail", "") or "(empty)"
                stderr_tail = fail_log.get("stderr_tail", "") or "(empty)"
                github_pr_evidence = build_github_pr_evidence(
                    repo_path, first_bad, disabled=getattr(args, "no_github", False)
                )

                motive_text = ((first_bad_commit.get("message", "") or "") + "\n" + (first_bad_commit.get("diff", "") or "")).lower()
                if github_pr_evidence and github_pr_evidence.get("available"):
                    motive_text += "\n" + (github_pr_evidence.get("title", "") or "").lower()
                    motive_text += "\n" + (github_pr_evidence.get("body", "") or "").lower()
                motive_keywords = ("because", "so that", "in order to", "to improve", "motivation", "reason", "why")
                has_explicit_motive = any(k in motive_text for k in motive_keywords)
                if has_explicit_motive:
                    if github_pr_evidence and github_pr_evidence.get("available"):
                        motive = "Possible motivation cues found in commit and/or linked PR evidence."
                    else:
                        motive = "Commit message/diff contains possible motivation cues; review message and diff context."

                findings = (
                    f"First bad commit is {first_bad} (date={first_bad_commit.get('date', '')}, "
                    f"author={first_bad_commit.get('author', '')}) under command: {cmd}"
                )
                if len(commits_range) <= 1:
                    findings += " Only 1 candidate commit in range; no mid-point tests were necessary."
            else:
                findings = f"No failing commit detected in range ({good_hash}..{bad_hash}) under command: {cmd}"
                github_pr_evidence = None

            if args.out:
                out_path = args.out
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("outputs", f"bisect_report_{ts}.md")
            out_dir = os.path.dirname(out_path) or "."
            os.makedirs(out_dir, exist_ok=True)

            lines = [
                "# Socratic Git Bisect Report",
                "",
                "## Inputs",
                f"- repo: {repo_path}",
                f"- good: {good_hash}",
                f"- bad: {bad_hash}",
                f"- cmd: `{cmd}`",
                f"- max_steps: {max_steps}",
                "- safety_note: `--cmd` executes arbitrary shell commands; use at your own risk.",
                "",
                "## Reproducibility",
                f"- good_returncode: {good_run.get('returncode', -1)}",
                f"- bad_returncode: {bad_run.get('returncode', -1)}",
                "",
                "## Steps",
                "| commit | pass_fail | returncode | note |",
                "|---|---|---|---|",
            ]
            for s in steps_all:
                status = "fail" if s["status"] == "bad" else "pass"
                lines.append(f"| {s['commit']} | {status} | {s['returncode']} | {s.get('note', '')} |")

            lines.extend(
                [
                    "",
                    "## First Bad Commit",
                    f"- hash: {first_bad if first_bad else '(none)'}",
                    f"- author: {first_bad_commit.get('author', '') if first_bad else ''}",
                    f"- date: {first_bad_commit.get('date', '') if first_bad else ''}",
                    (
                        f"- message: {(first_bad_commit.get('message', '') or '').splitlines()[0]}"
                        if first_bad and first_bad_commit.get("message")
                        else "- message: "
                    ),
                    f"- changed_files: {', '.join(changed_files) if changed_files else '(unavailable)'}",
                ]
            )
            if first_bad:
                lines.extend(
                    [
                        "- diff_snippet:",
                        "```diff",
                        diff_preview if diff_preview else "(unavailable)",
                        "```",
                    ]
                )
            else:
                lines.append("- diff_snippet: (unavailable)")

            lines.extend(
                [
                    "",
                    "## Failure Log Excerpt",
                    "### stdout_tail",
                    "```text",
                    stdout_tail,
                    "```",
                    "### stderr_tail",
                    "```text",
                    stderr_tail,
                    "```",
                    "",
                    "## Findings",
                    findings,
                    "",
                    "## Motive",
                    motive,
                ]
            )
            if github_pr_evidence:
                lines.extend(
                    [
                        "",
                        "## GitHub PR Evidence (Optional)",
                    ]
                )
                if github_pr_evidence.get("available"):
                    lines.extend(github_pr_evidence.get("text", "").splitlines()[1:])
                else:
                    lines.append(f"- pr: #{github_pr_evidence.get('pr_number', '')}")
                    lines.append("- PR evidence unavailable (rate limit or not found)")

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            print(f"report saved to {out_path}")
            return

        question = args.q
        topk = max(1, args.topk)
        contexts, _used_retrieval = resolve_contexts(
            repo_path,
            question,
            topk=topk,
            retrieval_mode=getattr(args, "retrieval", "vector"),
            command_label="ask",
        )
        if not contexts:
            print("Error: no related commits retrieved.")
            return

        target_info = get_symbol_or_lines_from_query(question)
        detected_file = target_info.get("file")
        detected_line = target_info.get("line")
        detected_symbol = target_info.get("symbol")
        resolved_line = None
        blame_info = ""
        introducing_commit = {}
        introducing_details_text = ""
        local_history_window = []

        # Phase 5 example:
        # python socratic_mvp.py ask --repo <repo> --q "In src/main.py, why was hybrid_search added?" --topk 3
        if detected_file and not detected_line and detected_symbol:
            resolved_line = find_symbol_definition(repo_path, detected_file, detected_symbol)
            if resolved_line:
                detected_line = resolved_line
            else:
                fallback_file, fallback_line = find_symbol_in_repo(
                    repo_path, detected_symbol, preferred_file=detected_file
                )
                if fallback_file and fallback_line:
                    detected_file = fallback_file
                    resolved_line = fallback_line
                    detected_line = fallback_line

        if detected_file and detected_line:
            blame_dict = get_blame_for_line(repo_path, detected_file, detected_line)
            if blame_dict:
                blame_info = (
                    f"file={blame_dict.get('file', '')}; "
                    f"line={blame_dict.get('line', '')}; "
                    f"commit={blame_dict.get('hash', '')}; "
                    f"author={blame_dict.get('author', '')}; "
                    f"date={blame_dict.get('date', '')}; "
                    f"line_content={blame_dict.get('line_content', '')}"
                )
                introducing_commit = get_introducing_commit(repo_path, blame_dict.get("hash", ""))
        elif detected_file:
            blame_info = get_file_blame(repo_path, detected_file)

        if introducing_commit:
            intro_msg_lines = introducing_commit.get("message", "").splitlines()[:10]
            intro_msg_preview = "\n".join(intro_msg_lines)
            intro_diff_clean = clean_diff(introducing_commit.get("diff", ""))
            intro_diff_lines = intro_diff_clean.splitlines()[:120]
            intro_diff_preview = "\n".join(intro_diff_lines)
            intro_files = extract_changed_files(introducing_commit.get("diff", ""))
            introducing_details_text = (
                "=== Introducing Commit Details ===\n"
                f"hash: {introducing_commit.get('hash', '')}\n"
                f"author/date: {introducing_commit.get('author', '')} / {introducing_commit.get('date', '')}\n"
                f"message:\n{intro_msg_preview}\n"
                f"files: {', '.join(intro_files) if intro_files else '(none)'}\n"
                f"diff_snippet:\n{intro_diff_preview}"
            )

        if detected_file:
            local_history_window = get_local_history_window(repo_path, detected_file, n=3)
        if local_history_window:
            print("=== Local History Window (file) ===")
            for row in local_history_window:
                msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                print(f"{row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())

        print_evidence(contexts, topk=topk)
        if detected_file:
            if detected_line:
                print(f"Detected target: file={detected_file}, line={detected_line}")
            else:
                print(f"Detected target: file={detected_file}")
        if detected_symbol:
            print(f"Detected symbol: {detected_symbol}")
        if resolved_line:
            print(f"Resolved line: {resolved_line}")
        if introducing_commit:
            intro_msg = introducing_commit.get("message", "").splitlines()[0] if introducing_commit.get("message") else ""
            print(f"Introducing commit: {introducing_commit.get('hash', '')} {intro_msg}".rstrip())
        if introducing_details_text:
            print(introducing_details_text)
        if blame_info:
            print(f"Blame Info: {blame_info}")

        print("=== Answer ===")
        try:
            llm_contexts = contexts
            if introducing_details_text:
                llm_contexts = [
                    {
                        "commit": introducing_commit,
                        "text": introducing_details_text,
                    }
                ] + llm_contexts
            if blame_info:
                llm_contexts = [
                    {
                        "commit": {
                            "hash": "blame",
                            "author": "git blame",
                            "date": "",
                            "message": "Blame summary from referenced file",
                            "diff": "",
                        },
                        "text": blame_info,
                    }
                ] + llm_contexts
            if local_history_window:
                history_lines = []
                for row in local_history_window:
                    msg = row.get("message", "").splitlines()[0] if row.get("message") else ""
                    history_lines.append(f"{row.get('hash', '')} {row.get('date', '')} {msg}".rstrip())
                history_text = "=== Local History Window (file) ===\n" + "\n".join(history_lines)
                llm_contexts = llm_contexts + [
                    {
                        "commit": {
                            "hash": "local-history",
                            "author": "git log",
                            "date": "",
                            "message": "Recent file-local history window",
                            "diff": "",
                        },
                        "text": history_text,
                    }
                ]
            if _llm_disabled():
                answer = "I don't know (insufficient evidence in commit message/diff)."
            else:
                answer = _generate_answer(question, llm_contexts)
        except Exception as exc:
            msg = str(exc)
            if "No such file or directory" in msg or "not found" in msg.lower():
                print(f"Error: model may be missing/not downloaded. Details: {exc}")
            else:
                print(f"Error: failed to run local model. Details: {exc}")
            return
        print(answer)

    except RuntimeError as exc:
        print(f"Error: {exc}")
    except Exception as exc:
        print(f"Error: {exc}")
        return


if __name__ == "__main__":
    main()
