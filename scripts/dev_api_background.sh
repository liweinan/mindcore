#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
PID_FILE="$ROOT/logs/api.pid"
LOG_FILE="$ROOT/logs/api.log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "API 已在运行，PID $(cat "$PID_FILE")"
  exit 0
fi

if command -v lsof >/dev/null 2>&1 && lsof -ti:8000 >/dev/null 2>&1; then
  echo "端口 8000 已被占用，请先结束占用进程或换端口。"
  exit 1
fi

if [ ! -d .venv ]; then
  echo "未找到 .venv，请先: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate
nohup uvicorn api.main:app --host 127.0.0.1 --port 8000 >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "MindCore API 已后台启动，PID $(cat "$PID_FILE")，日志 $LOG_FILE"
echo "停止: ./scripts/dev_api_stop.sh"
