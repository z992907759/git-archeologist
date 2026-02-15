import * as vscode from "vscode";
import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

function escapeHtml(input: string): string {
  return input
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function safeName(input: string): string {
  return input.replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function parseArgv(input: string): string[] {
  const src = (input || "").trim();
  if (!src) return [];
  const out: string[] = [];
  const re = /"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^\s]+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    const token = m[1] ?? m[2] ?? m[3] ?? "";
    out.push(token.replace(/\\(["'])/g, "$1"));
  }
  return out;
}

function quoteForDisplay(value: string): string {
  if (!value) return `""`;
  if (!/[^\w@%+=:,./-]/.test(value)) return value;
  return `"${value.replace(/"/g, '\\"')}"`;
}

function getReportsDir(context: vscode.ExtensionContext): string {
  return path.join(context.globalStorageUri.fsPath, "reports");
}

function resolveRunPy(workspacePath: string, context: vscode.ExtensionContext): string {
  const inWorkspace = path.join(workspacePath, "run.py");
  if (fs.existsSync(inWorkspace)) {
    return inWorkspace;
  }
  const nearExtension = path.resolve(context.extensionPath, "..", "run.py");
  if (fs.existsSync(nearExtension)) {
    return nearExtension;
  }
  return "run.py";
}

function renderWebview(panel: vscode.WebviewPanel, title: string, content: string): void {
  panel.webview.html = `
    <!doctype html>
    <html>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 16px;">
      <h2>${escapeHtml(title)}</h2>
      <pre style="white-space: pre-wrap; line-height: 1.5; border: 1px solid #ddd; padding: 12px;">${escapeHtml(content)}</pre>
    </body>
    </html>
  `;
}

export function activate(context: vscode.ExtensionContext) {
  const traceCmd = vscode.commands.registerCommand(
    "socraticGit.traceDecision",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("No active editor found.");
        return;
      }

      const workspaceFolder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
      if (!workspaceFolder) {
        vscode.window.showErrorMessage("Please open a folder/workspace first.");
        return;
      }

      const workspacePath = workspaceFolder.uri.fsPath;
      const filePath = editor.document.uri.fsPath;
      const relFile = path.relative(workspacePath, filePath).replace(/\\/g, "/");
      const lineNo = editor.selection.active.line + 1;
      const question = `In ${relFile} line ${lineNo}, why was this added?`;
      const cfg = vscode.workspace.getConfiguration("socratic");
      const pythonCommand = (cfg.get<string>("pythonCommand", "python3") || "python3").trim();
      const cliArgsExtra = (cfg.get<string>("cliArgsExtra", "") || "").trim();
      const runPy = resolveRunPy(workspacePath, context);
      const reportsDir = getReportsDir(context);
      fs.mkdirSync(reportsDir, { recursive: true });
      const outFile = path.join(
        reportsDir,
        `${Date.now()}_${safeName(path.basename(relFile))}_${lineNo}.md`
      );

      const pyArgv = parseArgv(pythonCommand);
      if (pyArgv.length === 0) {
        vscode.window.showErrorMessage('Invalid "socratic.pythonCommand".');
        return;
      }
      const executable = pyArgv[0];
      const baseArgs = pyArgv.slice(1);
      const extraArgs = parseArgv(cliArgsExtra);
      const cmdArgs = [
        ...baseArgs,
        runPy,
        "trace",
        "--repo",
        workspacePath,
        "--q",
        question,
        ...extraArgs,
        "--out",
        outFile,
      ];
      const cmdForDisplay = [executable, ...cmdArgs].map(quoteForDisplay).join(" ");

      vscode.window.withProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: "Socratic Git: running trace...",
          cancellable: false,
        },
        () => new Promise<void>((resolve) => {
            let stdout = "";
            let stderr = "";
            const child = spawn(executable, cmdArgs, { cwd: workspacePath });
            child.stdout.on("data", (buf) => {
              stdout += String(buf);
            });
            child.stderr.on("data", (buf) => {
              stderr += String(buf);
            });
            child.on("error", (error) => {
              const panel = vscode.window.createWebviewPanel(
                "socraticGitTrace",
                "Socratic Git Trace Error",
                vscode.ViewColumn.Beside,
                { enableScripts: false }
              );
              const errText =
                `Trace command failed.\n\n` +
                `Command:\n${cmdForDisplay}\n\n` +
                `stdout:\n${stdout || "(empty)"}\n\n` +
                `stderr:\n${stderr || error.message || "(empty)"}\n\n` +
                `Tip:\nConfigure "socratic.pythonCommand", e.g.:\n` +
                `conda run -n socratic-git python`;
              renderWebview(panel, "Socratic Git Trace Error", errText);
              vscode.window.showErrorMessage("Socratic trace failed. See Webview for details.");
              resolve();
            });
            child.on("close", (code) => {
              const panel = vscode.window.createWebviewPanel(
                "socraticGitTrace",
                "Socratic Git Trace Report",
                vscode.ViewColumn.Beside,
                { enableScripts: false }
              );

              if (code !== 0) {
                const errText =
                  `Trace command failed.\n\n` +
                  `Command:\n${cmdForDisplay}\n\n` +
                  `stdout:\n${stdout || "(empty)"}\n\n` +
                  `stderr:\n${stderr || "(empty)"}\n\n` +
                  `Tip:\nConfigure "socratic.pythonCommand", e.g.:\n` +
                  `conda run -n socratic-git python`;
                renderWebview(panel, "Socratic Git Trace Error", errText);
                vscode.window.showErrorMessage("Socratic trace failed. See Webview for details.");
                resolve();
                return;
              }

              if (!fs.existsSync(outFile)) {
                const errText =
                  `Report file not found.\n\n` +
                  `Expected:\n${outFile}\n\n` +
                  `Command:\n${cmdForDisplay}\n\n` +
                  `stdout:\n${stdout || "(empty)"}\n\n` +
                  `stderr:\n${stderr || "(empty)"}`;
                renderWebview(panel, "Socratic Git Trace Error", errText);
                resolve();
                return;
              }

              const content = fs.readFileSync(outFile, "utf-8");
              renderWebview(panel, "Socratic Git Trace Report", `Report: ${outFile}\n\n${content}`);
              resolve();
            });
          }),
      );
    }
  );

  const openReportsCmd = vscode.commands.registerCommand(
    "socraticGit.openReportsFolder",
    async () => {
      const reportsDir = getReportsDir(context);
      fs.mkdirSync(reportsDir, { recursive: true });
      await vscode.commands.executeCommand("revealFileInOS", vscode.Uri.file(reportsDir));
    }
  );

  const clearReportsCmd = vscode.commands.registerCommand(
    "socraticGit.clearReports",
    async () => {
      const reportsDir = getReportsDir(context);
      try {
        fs.rmSync(reportsDir, { recursive: true, force: true });
        fs.mkdirSync(reportsDir, { recursive: true });
        vscode.window.showInformationMessage("Socratic reports cleared.");
      } catch (err: any) {
        vscode.window.showErrorMessage(`Failed to clear reports: ${err?.message || String(err)}`);
      }
    }
  );

  context.subscriptions.push(traceCmd, openReportsCmd, clearReportsCmd);
}

export function deactivate() {}
