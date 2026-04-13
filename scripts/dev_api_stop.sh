#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/logs/api.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "未找到 $PID_FILE，尝试释放 8000 端口…"
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
  fi
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID" 2>/dev/null || true
  sleep 1
  kill -9 "$PID" 2>/dev/null || true
  echo "已停止 PID $PID"
else
  echo "PID $PID 不存在，清理 pid 文件"
fi
rm -f "$PID_FILE"
