# Socratic Git VSCode Shell

Minimal VSCode extension shell that calls existing CLI:

- Right click in editor: `Trace Decision`
- Uses current file + cursor line
- Runs:
  - `python run.py trace --repo <workspaceFolder> --q "In <file> line <n>, why was this added?" --out <report>`
- Opens generated markdown report in a Webview panel

## Setup

1. Open `vscode-extension/` in VSCode (Extension Development Host workflow).
2. Install deps:
   - `npm install`
3. Compile:
   - `npm run compile`
4. Press `F5` to launch Extension Development Host.

## Usage

1. Open a workspace that contains `run.py` at root.
2. Open any source file, place cursor on a line.
3. Right click editor -> `Trace Decision`.
4. Wait for command execution; report is shown in Webview.

## Notes

- This extension only provides UI + local CLI invocation.
- Existing Python project structure and CLI behavior are unchanged.
- Report output path defaults to `reports/trace/vscode_trace_<timestamp>.md`.
