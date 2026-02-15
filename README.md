# Socratic Git

**English** | [中文](README.zh-CN.md)

Short description:
Evidence-driven Git history analysis and decision tracing.

---

## Core Capabilities

### 1. Trace Decision
- Locate introducing commit
- Structured change detection (parameter changes, thresholds, etc.)
- Evidence-layered motive analysis (A/B/C model)
- Motive scoring (0-10)
- GitHub Issue relevance guard
- Timeline filtering (noise commit removal)

### 2. Regression Analysis
- Identify first bad commit
- Safe bisect execution
- Clone / Worktree mode
- Head restoration verification
- Timing metrics (setup/bisect/cleanup)

### 3. Large-Repository Optimizations
- Incremental indexing (commit-level cache)
- SQLite cache for changed files and patch snippets
- AST cache (skip large files >500KB)
- ripgrep size limits and glob filtering
- Optional `--window-days`
- `--no-merges` default optimization

### 4. Cross-File Structural Analysis
- Co-change statistical coupling
- AST-based Python symbol extraction
- Optional LSP-based cross-file resolution
- Impact propagation (static import graph)
- Structural confidence scoring

### 5. Interactive Modes
- CLI
- Chat REPL (`/trace`, `/motive`, `/structured`, `/timeline`, `/export`)
- HTML export (`--html-out`)
- VSCode extension shell (calls CLI, no repo pollution)

### 6. CI Integration
- GitHub Actions workflow example
- Auto-generate trace/regress reports
- Upload artifacts
- Optional PR comment

---

## Installation

Minimal requirements:

- Python 3.10+
- ripgrep (`rg`)
- Git

Optional (for LSP mode):

- `pyright-langserver`
- `typescript-language-server`
- `gopls`

---

## Quick Start

### Trace

```bash
python run.py trace --repo . --q "Why was X changed?" --out report.md
```

Optional HTML:

```bash
python run.py trace --repo . --q "Why was X changed?" --out report.md --html-out report.html
```

### Regress

```bash
python run.py regress --repo . --file path --pattern "..." --out regress.md
```

### Bisect

```bash
python run.py bisect --repo . --good <sha> --bad <sha> --cmd "<predicate>"
```

Optional:

```bash
python run.py bisect --repo . --good <sha> --bad <sha> --cmd "<predicate>" --bisect-mode worktree --verbose
```

---

## Performance Notes

- First run builds incremental index
- Subsequent runs skip already indexed commits
- File/patch details are cached in sqlite
- Large repos benefit from `worktree` bisect mode and `--window-days`

Cache location:

```text
.socratic_cache/
```

To reset cache:

```bash
rm -rf .socratic_cache
```

---

## VSCode Extension

Minimal extension shell:

- Right-click -> Trace Decision
- Uses workspace root as repo
- Reports stored in VSCode `globalStorage`
- Does not write into user repository

Configuration:

```json
"socratic.pythonCommand": "conda run -n socratic-git python"
```

---

## Safety Guarantees

- No modification to user repo during trace/regress
- Bisect clone/worktree cleaned automatically
- HEAD restored after bisect
- HTML export is standalone (no CDN)
- LSP fallback is safe when tools are missing

---

## Completed Milestones

- Incremental commit indexing
- SQLite file/patch cache
- Structured change detection
- Motive scoring system
- GitHub relevance guard
- HTML interactive report export
- Chat REPL mode
- VSCode extension shell
- CI workflow
- Worktree bisect mode
- Optional LSP resolution (fallback safe)

---

## Roadmap / Future Work

- Full multi-language robust LSP integration
- Cross-repository analysis
- Distributed indexing for very large repos
- Advanced HTML dashboard UI
- Team-level ownership analytics
- Marketplace-ready VSCode extension packaging

---

## Limitations

- LSP requires external tools
- Motive quality depends on commit/issue quality
- Statistical co-change is not causality
- Very large monorepos may still require window filtering
