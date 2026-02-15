#!/usr/bin/env python3
import fnmatch
import subprocess
import sys

BLOCKED_PREFIXES = (
    "reports/",
    "data/",
    ".lancedb/",
    ".cache/huggingface/",
)
BLOCKED_GLOBS = (
    "*.safetensors",
    "*.bin",
    "*.gguf",
    "*.mlx",
    "*.pt",
    "*.pth",
)


def staged_files():
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        print("ERROR: failed to read staged files")
        print(proc.stderr.strip())
        sys.exit(2)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def is_blocked(path):
    p = path.replace("\\", "/")
    for prefix in BLOCKED_PREFIXES:
        if p.startswith(prefix):
            return f"blocked path prefix: {prefix}"
    for pat in BLOCKED_GLOBS:
        if fnmatch.fnmatch(p, pat):
            return f"blocked artifact pattern: {pat}"
    return None


def main():
    bad = []
    for path in staged_files():
        reason = is_blocked(path)
        if reason:
            bad.append((path, reason))

    if bad:
        print("Preflight FAILED: blocked files are staged:")
        for path, reason in bad:
            print(f"- {path} ({reason})")
        sys.exit(1)

    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
