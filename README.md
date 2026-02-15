# Socratic Git

Socratic Git is a local AI tool for **code archaeology**: it reads Git history, traces where logic came from, and explains code evolution with evidence.

Instead of giving generic guesses, it builds an evidence chain from commits, diffs, blame, and local history, then reasons with a local LLM.

---

## Why This Project Exists

Modern repositories accumulate years of hidden decisions.
When you ask:

- Why was this feature added?
- Who introduced this logic?
- When did this behavior start?

You usually have to manually inspect logs, diffs, and blame output.

Socratic Git automates that workflow into a local CLI pipeline:

`Git mining -> evidence retrieval -> introducing commit tracing -> local LLM explanation`

---

## Core Capabilities

- **File-level blame**
  Track the latest commit touching a file to get quick ownership and recency context.

- **Line-level blame**
  For line-specific questions (`line 120`, `L120`, `:120`), locate the exact commit that last changed that line.

- **Symbol detection**
  Detect function/class/symbol mentions from natural language queries and resolve likely source locations.

- **Introducing commit tracing**
  From blame results, fetch the introducing commit metadata and diff snippet as primary evidence.

- **Local history window**
  Add nearby file history (`git log -n 3 -- <file>`) to show short-term evolution context.

- **Local LLM reasoning**
  Use a local Qwen model via `mlx-lm` on Apple Silicon. No cloud API required.

---

## Demo (Expected Effect)

Ask a repository-level question:

```bash
python run.py ask --repo /path/to/repo --q "In src/main.py, why was hybrid_search added?" --topk 3
```

Expected behavior:

1. Detect target file/symbol (and line if resolvable).
2. Print retrieved Evidence blocks.
3. Print Introducing Commit Details (hash/message/files/diff snippet).
4. Print Local History Window for that file.
5. Generate a local answer grounded in evidence.

Typical output shape:

```text
=== Evidence Blocks ===
[1] id: ...
    date: ...
    author: ...
    message: ...

Detected target: file=src/main.py, line=...
Detected symbol: hybrid_search
Resolved line: ...
Introducing commit: <hash> <message>

=== Introducing Commit Details ===
hash: ...
author/date: ...
message: ...
files: ...
diff_snippet: ...

=== Local History Window (file) ===
<hash> <date> <message>
...

=== Answer ===
...
```

---

## Installation

### 1) Create Conda environment (Apple Silicon)

```bash
conda create -n socratic-git python=3.11 -y
conda activate socratic-git
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Build / refresh index

```bash
python run.py index --repo /path/to/repo --n 200
```

Arguments:

- `--repo`: local Git repository path
- `--n`: number of latest commits to index
- `--mode`: `vector` or `keyword` (default `vector`; offline fallback supported)

### Ask questions

```bash
python run.py ask --repo /path/to/repo --q "Why was login feature added?" --topk 3
```

Arguments:

- `--repo`: local Git repository path
- `--q`: natural language question
- `--topk`: number of retrieved commit contexts
- `--retrieval`: `vector` or `keyword` (default `vector`)

You can also run module entrypoint:

```bash
python -m socratic_git.cli index --repo /path/to/repo --n 200
python -m socratic_git.cli ask --repo /path/to/repo --q "..." --topk 3
```

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── run.py                     # unified entrypoint
├── data/                      # local indexes/caches (git-ignored)
│   ├── lancedb/
│   └── keyword_index/
├── reports/                   # generated reports (git-ignored)
├── scripts/
│   ├── legacy_mvp.py
│   ├── test_env.py
│   ├── preflight_check.py     # staged-files guard
│   └── install_hooks.sh       # install pre-commit hook
└── socratic_git/
    ├── __init__.py
    ├── cli.py                 # argparse CLI (index / ask / trace / regress / bisect)
    ├── miner.py               # git mining, blame, symbol and introducing commit helpers
    ├── rag.py                 # embeddings + LanceDB indexing/retrieval
    ├── llm.py                 # local mlx-lm generation
    └── utils.py               # helpers (table naming, evidence print)
```

---

## Why It Matters

Socratic Git is valuable when you need to understand **legacy code under time pressure**.

It helps teams:

- reduce onboarding time on large repos
- investigate regressions with provenance evidence
- explain “why” behind historical changes
- build confidence before refactors or migrations

In short, this is a practical AI assistant for **engineering memory**.

---

## Notes

- Fully local workflow: Git + embeddings + retrieval + LLM all run on your machine.
- If evidence is insufficient, the system is designed to answer:
  `我不知道（根据现有提交上下文无法判断）。`
- Better commit messages and cleaner diffs directly improve answer quality.
- Generated artifacts under `reports/` and `data/` are intentionally ignored from Git.
- Before push, run:
  `git status && git diff --cached --name-only && python scripts/preflight_check.py`
