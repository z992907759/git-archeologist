# Socratic Git

**English** | [中文](README.zh-CN.md)

Socratic Git is a local AI tool for **code archaeology and decision tracing**: it reads Git history, locates introducing commits, builds evidence chains, and produces explainable reports.

Principle: **evidence first, conclusion after**. If evidence is insufficient, the system explicitly outputs "unknown / insufficient evidence".

---

## Why This Exists

In large repositories, teams often need to answer:

- Why was this feature added?
- Who introduced this logic?
- When did this behavior start?

Traditional workflow requires manual `git log`, `git blame`, diff, PR/Issue digging. Socratic Git automates the workflow and summarizes with evidence.

---

## Core Capabilities

- **File-level / Line-level blame**: find the latest commit for a file or a specific line
- **Symbol resolution**: resolve functions/classes across files (optional LSP)
- **Introducing commit tracing**: message + diff snippet for the introducing commit
- **Local history window**: short context window around the target file
- **Evidence-driven answer**: conclusions must cite evidence; otherwise "unknown"
- **Decision Timeline**: filtered evidence timeline of meaningful commits
- **Cross-file co-change**: co-change stats + structural signals
- **Trace / Regress / Bisect reports**: Markdown + optional HTML reports
- **Chat mode**: reuse last trace for follow-up questions
- **Offline-ready**: falls back to keyword retrieval when embeddings are unavailable

---

## Quick Start

### 1) Create environment (Apple Silicon)

```bash
conda create -n socratic-git python=3.11 -y
conda activate socratic-git
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## CLI Usage

### Index

```bash
python run.py index --repo /path/to/repo --n 200
```

Options:
- `--mode`: `vector` / `keyword` (default vector, offline fallback supported)
- `--window-days`: limit history to last N days (large repo optimization)

### Trace (report)

```bash
python run.py trace --repo /path/to/repo --q "In src/main.py line 120, why was this added?" --out reports/trace.md
```

Common flags:
- `--html-out`: also export HTML
- `--symbol-resolver`: `heuristic` / `lsp`
- `--github-evidence on|off`
- `--window-days`
- `--verbose`

### Regress (introducing commit)

```bash
python run.py regress --repo /path/to/repo --file src/main.py --pattern "HYBRID" --out reports/regress.md
```

Common flags:
- `--symbol`: use symbol instead of pattern
- `--html-out`
- `--github-evidence on|off`
- `--window-days`

### Bisect (first bad)

```bash
python run.py bisect --repo /path/to/repo --good <hash> --bad <hash> --cmd "python -c 'print(1)'" --out reports/bisect.md
```

Common flags:
- `--bisect-mode`: `clone` / `worktree`
- `--verbose`

### Chat (follow-up loop)

```bash
python run.py chat --repo /path/to/repo
```

Commands:
- `/trace <question>`
- `/structured`
- `/motive`
- `/timeline`
- `/export <path.md>`
- `/exit`

---

## VSCode Extension (optional)

Location: `vscode-extension/`

What it does: right-click `Trace Decision`, runs CLI, renders report in a Webview.

Quick usage:
1. Open `vscode-extension/` in VSCode
2. `npm install`
3. `npm run compile`
4. Press `F5` to launch Extension Development Host

Settings:
- `socratic.pythonCommand`: e.g. `conda run -n socratic-git python`
- `socratic.cliArgsExtra`: extra CLI args, e.g. `--retrieval keyword --github-evidence off`

---

## GitHub Actions (CI)

Workflow in `.github/workflows/socratic.yml`:
- Runs on PR/push
- Generates trace + regress reports
- Uploads artifacts
- Defaults to `SOCRATIC_SKIP_LLM=1` + `--retrieval keyword`

---

## Project Structure

```text
.
├── README.md
├── README.en.md
├── README.zh-CN.md
├── requirements.txt
├── run.py
├── data/                      # local indexes/caches (git-ignored)
├── reports/                   # generated reports (git-ignored)
├── scripts/
├── socratic_git/              # core logic
└── vscode-extension/          # VSCode shell
```

---

## Known Limitations

- Why-level answers depend on evidence quality (generic commits => unknown)
- Very large repos should use `--window-days` or keyword retrieval
- LSP is optional; missing tools fall back to heuristics
- Offline mode falls back to keyword retrieval

---

## What NOT to commit

- `reports/`, `data/`, `.socratic_cache/`, model files, `*.db/*.sqlite/*.parquet`
- `node_modules/`, `dist/`, `out/`, virtualenv/cache dirs

---

## Roadmap / Future Work

- IDE/UX: richer VSCode workflows
- Why analysis: stronger evidence sources and confidence
- Large repo performance: incremental indexing + window tuning
- Symbol resolution: deeper LSP integration
- CI integration: auto reports for reviews
- Report UX: richer interactive HTML
