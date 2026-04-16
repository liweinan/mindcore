#!/usr/bin/env bash
# 起 API 并 POST /v1/chat，验证经 Ollama 拿到真实回复（需本机 11434 已就绪且已 pull 默认模型）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

./scripts/dev_api_stop.sh || true

export QDRANT_RAG_COLLECTION="${QDRANT_RAG_COLLECTION:-mental_health_knowledge}"

./scripts/dev_api_background.sh
for _ in $(seq 1 20); do
  if curl -sf http://127.0.0.1:8000/ready >/dev/null; then
    break
  fi
  sleep 1
done

echo "=== POST /v1/chat ==="
curl -sS --max-time 180 -X POST "http://127.0.0.1:8000/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"smoke_user","message":"用一句话用中文说晚安"}'
echo ""

./scripts/dev_api_stop.sh
