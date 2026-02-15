#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Socratic Git Safe Status Check =="
echo "repo: $ROOT_DIR"
echo

check_path() {
  local p="$1"
  if [ -e "$p" ]; then
    echo "[WARN] exists: $p  (runtime/generated; should not be committed)"
  else
    echo "[OK]   missing: $p"
  fi
}

check_glob() {
  local pat="$1"
  shopt -s nullglob
  local matches=( $pat )
  shopt -u nullglob
  if [ "${#matches[@]}" -gt 0 ]; then
    echo "[WARN] matches: $pat"
    for m in "${matches[@]}"; do
      echo "       - $m"
    done
  else
    echo "[OK]   no match: $pat"
  fi
}

check_path "reports"
check_path ".socratic_cache"
check_path "data/lancedb"
check_path "data/keyword_index"
check_path ".lancedb"
check_path "models"
check_path "checkpoints"
check_path "node_modules"
check_path "out"
check_path "dist"

echo
check_glob "*.db"
check_glob "*.sqlite"
check_glob "*.parquet"
check_glob "*.gguf"
check_glob "*.safetensors"

echo
echo "Tip: run before push"
echo "  git status"
echo "  git diff --cached --name-only"
echo "  bash tools/safe_status.sh"
