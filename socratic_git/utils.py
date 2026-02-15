"""Utility helpers for CLI flow."""

import hashlib
import os

from socratic_git.miner import extract_changed_files


def table_name_for_repo(repo_path):
    """Derive a stable LanceDB table name from absolute repo path."""
    key = os.path.abspath(repo_path)
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"commits_{h}"


def print_evidence(contexts, topk=3):
    """Print evidence blocks in the same demo-friendly format."""
    print("=== Evidence Blocks ===")
    for i, ctx in enumerate(contexts[:topk], start=1):
        commit = ctx.get("commit", {})
        message_line = commit.get("message", "").splitlines()[0] if commit.get("message") else ""
        print(f"[{i}] id: {commit.get('hash', '')}")
        print(f"    date: {commit.get('date', '')}")
        print(f"    author: {commit.get('author', '')}")
        print(f"    message: {message_line}")
        files = extract_changed_files(commit.get("diff", ""))
        if files:
            print(f"    files: {', '.join(files)}")
