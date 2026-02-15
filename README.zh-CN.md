# Socratic Git

[English](README.en.md) | **中文**

Socratic Git 是一个本地 AI 工具，用于 **代码考古与决策追溯**：它读取 Git 历史、定位引入点、构建证据链，并生成可解释的报告。

核心理念：**先证据，后结论**。当证据不足时，系统会明确输出“不知道/证据不足”。

---

## 为什么需要它

在大型仓库中，团队经常需要回答：

- 为什么加了这个功能？
- 谁引入了这段逻辑？
- 这个行为是从什么时候开始的？

传统做法需要人工查 `git log`、`git blame`、diff、PR/Issue。Socratic Git 将这些步骤自动化，并用本地 LLM 做证据驱动的总结。

---

## 核心能力

- **File-level / Line-level blame**：定位文件或行的最近引入提交
- **Symbol 解析**：从问题中解析函数/类名，并跨文件定位定义（可选 LSP）
- **Introducing commit tracing**：获取引入提交的 message + diff 片段
- **Local history window**：展示目标文件附近的提交时间窗
- **Evidence-driven Answer**：结论必须引用证据，不足则输出未知
- **Decision Timeline**：过滤噪音提交的决策演化时间线
- **Cross-file co-change**：共变文件统计 + 结构依赖信号
- **Trace / Regress / Bisect 报告**：自动生成 Markdown + 可选 HTML 报告
- **Chat 模式**：一次 trace 后可多轮追问
- **离线可用**：向量不可用时自动降级为 keyword 检索

---

## 快速上手

### 1) 创建环境（Apple Silicon）

```bash
conda create -n socratic-git python=3.11 -y
conda activate socratic-git
```

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

---

## CLI 使用

### Index（索引）

```bash
python run.py index --repo /path/to/repo --n 200
```

参数：
- `--mode`：`vector` / `keyword`（默认 vector，离线会自动降级）
- `--window-days`：只扫描最近 N 天提交（大仓库优化）

### Trace（证据报告）

```bash
python run.py trace --repo /path/to/repo --q "In src/main.py line 120, why was this added?" --out reports/trace.md
```

常用参数：
- `--html-out`：额外导出 HTML
- `--symbol-resolver`：`heuristic` / `lsp`
- `--github-evidence on|off`
- `--window-days`
- `--verbose`

### Regress（回归引入点）

```bash
python run.py regress --repo /path/to/repo --file src/main.py --pattern "HYBRID" --out reports/regress.md
```

常用参数：
- `--symbol`：改为基于 symbol 寻找引入点
- `--html-out`
- `--github-evidence on|off`
- `--window-days`

### Bisect（命令驱动定位 first bad）

```bash
python run.py bisect --repo /path/to/repo --good <hash> --bad <hash> --cmd "python -c 'print(1)'" --out reports/bisect.md
```

常用参数：
- `--bisect-mode`：`clone` / `worktree`
- `--verbose`

### Chat（多轮追问）

```bash
python run.py chat --repo /path/to/repo
```

内置指令：
- `/trace <question>`
- `/structured`
- `/motive`
- `/timeline`
- `/export <path.md>`
- `/exit`

---

## VSCode 扩展（可选）

目录：`vscode-extension/`

能力：右键 `Trace Decision`，自动调用 CLI 并在 Webview 展示报告。

快速使用：
1. 在 VSCode 打开 `vscode-extension/`
2. `npm install`
3. `npm run compile`
4. 按 `F5` 启动 Extension Development Host

插件设置：
- `socratic.pythonCommand`：支持 conda 路径，如 `conda run -n socratic-git python`
- `socratic.cliArgsExtra`：额外 CLI 参数（如 `--retrieval keyword --github-evidence off`）

---

## GitHub Actions (CI)

已提供 `.github/workflows/socratic.yml`：
- PR / push 自动生成 trace + regress 报告
- 产出为 GitHub Actions artifacts
- 默认使用 `SOCRATIC_SKIP_LLM=1` + `--retrieval keyword`

---

## 项目结构

```text
.
├── README.md
├── README.en.md
├── README.zh-CN.md
├── requirements.txt
├── run.py
├── data/                      # 本地索引与缓存（git-ignored）
├── reports/                   # 生成报告（git-ignored）
├── scripts/
├── socratic_git/              # 核心逻辑
└── vscode-extension/          # VSCode 扩展壳
```

---

## 已知限制

- why 解释强依赖证据质量（commit/PR/issue 过于泛时会输出未知）
- 超大仓库需配合 `--window-days` 或 keyword 模式
- LSP 解析为可选能力，工具缺失会回退 heuristic
- 离线环境会降级到 keyword 检索

---

## What NOT to commit

- `reports/`、`data/`、`.socratic_cache/`、模型文件、`.db/*.sqlite/*.parquet`
- `node_modules/`、`dist/`、`out/`、虚拟环境目录

---

## Roadmap / Future Work

- IDE/UX：更完整的 VSCode 插件交互体验
- why：更强的证据来源与置信度表达
- 大仓库性能：增量索引与窗口优化
- 符号解析：更深层的 LSP 解析
- CI 集成：自动生成报告供 review
- 报告体验：更交互的 HTML 展示
