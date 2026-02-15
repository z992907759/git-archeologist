# Socratic Git

[English](README.md) | **中文**

简述：
基于证据的 Git 历史分析与决策追溯工具。

---

## 核心能力

### 1. 决策追踪（Trace Decision）
- 定位引入提交（introducing commit）
- 结构化变更检测（参数、阈值等）
- 分层动机证据分析（A/B/C 模型）
- 动机评分（0-10）
- GitHub Issue 相关性守卫（relevance guard）
- 时间线噪音过滤（移除低信息提交）

### 2. 回归分析（Regression Analysis）
- 定位 first bad commit
- 安全 bisect 执行
- Clone / Worktree 双模式
- HEAD 恢复校验
- 计时指标（setup/bisect/cleanup）

### 3. 大仓库优化（Large-Repository Optimizations）
- 增量索引（commit 级缓存）
- changed files 与 patch 片段的 SQLite 缓存
- AST 缓存（大文件 >500KB 跳过）
- ripgrep 文件大小限制与 glob 过滤
- 可选 `--window-days`
- 默认 `--no-merges` 优化

### 4. 跨文件结构分析（Cross-File Structural Analysis）
- 共变统计耦合（co-change）
- 基于 AST 的 Python 符号提取
- 可选 LSP 跨文件符号解析
- 影响传播图（静态 import 图）
- 结构置信度评分

### 5. 交互模式（Interactive Modes）
- CLI
- Chat REPL（`/trace`、`/motive`、`/structured`、`/timeline`、`/export`）
- HTML 导出（`--html-out`）
- VSCode 插件壳（调用 CLI，不污染仓库）

### 6. CI 集成（CI Integration）
- GitHub Actions 工作流示例
- 自动生成 trace/regress 报告
- 上传 artifacts
- 可选 PR 评论

---

## 安装

最低要求：

- Python 3.10+
- ripgrep（`rg`）
- Git

可选（LSP 模式）：

- `pyright-langserver`
- `typescript-language-server`
- `gopls`

---

## 快速开始

### Trace

```bash
python run.py trace --repo . --q "Why was X changed?" --out report.md
```

可选 HTML：

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

可选：

```bash
python run.py bisect --repo . --good <sha> --bad <sha> --cmd "<predicate>" --bisect-mode worktree --verbose
```

---

## 性能说明

- 首次运行会构建增量索引
- 后续运行会跳过已索引提交
- 文件与 patch 细节会缓存到 sqlite
- 大仓库建议使用 `worktree` 模式和 `--window-days`

缓存目录：

```text
.socratic_cache/
```

清理缓存：

```bash
rm -rf .socratic_cache
```

---

## VSCode 插件

当前是最小插件壳：

- 右键 -> Trace Decision
- 使用 workspace root 作为 repo
- 报告存到 VSCode `globalStorage`
- 不写入用户仓库

配置示例：

```json
"socratic.pythonCommand": "conda run -n socratic-git python"
```

---

## 安全保证

- trace/regress 不修改用户仓库
- bisect 的 clone/worktree 自动清理
- bisect 结束后会校验并恢复 HEAD
- HTML 导出为单文件（不依赖 CDN）
- LSP 工具缺失时安全回退

---

## 已完成里程碑

- 增量 commit 索引
- SQLite files/patch 缓存
- 结构化变更检测
- 动机评分系统
- GitHub 相关性守卫
- HTML 交互报告导出
- Chat REPL 模式
- VSCode 插件壳
- CI 工作流
- Worktree bisect 模式
- 可选 LSP 解析（安全回退）

---

## 未来规划

- 多语言稳健 LSP 集成
- 跨仓库分析
- 超大仓库分布式索引
- 高级 HTML dashboard UI
- 团队级 ownership 分析
- VSCode 插件市场化打包

---

## 当前限制

- LSP 依赖外部工具安装
- 动机解释质量依赖 commit/issue 质量
- 共变统计不等于因果关系
- 超大 monorepo 仍可能需要窗口过滤
