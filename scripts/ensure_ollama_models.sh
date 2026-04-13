#!/usr/bin/env bash
# 自动拉取对话与嵌入小模型（Ollama）。优先本机 ollama 命令；否则 docker compose --profile ollama。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHAT_MODEL="${OLLAMA_CHAT_MODEL:-qwen2.5:3b}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"
BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

wait_ollama_http() {
  echo "等待 Ollama HTTP (${BASE_URL}) …"
  for _ in $(seq 1 90); do
    if curl -sf "${BASE_URL}/api/tags" >/dev/null 2>&1; then
      echo "Ollama API 已就绪"
      return 0
    fi
    sleep 2
  done
  echo "超时：${BASE_URL} 无响应"
  return 1
}

pull_in_compose() {
  echo ">>> 启动 Compose 中的 ollama（profile ollama）"
  docker compose --profile ollama up -d ollama
  wait_ollama_http
  echo ">>> 容器内拉取 ${CHAT_MODEL}（体积较大，首次需数分钟）"
  docker compose exec -T ollama ollama pull "${CHAT_MODEL}"
  echo ">>> 容器内拉取 ${EMBED_MODEL}"
  docker compose exec -T ollama ollama pull "${EMBED_MODEL}"
}

if command -v ollama >/dev/null 2>&1; then
  echo ">>> 使用本机 ollama 拉取 ${CHAT_MODEL}、${EMBED_MODEL}"
  ollama pull "${CHAT_MODEL}"
  ollama pull "${EMBED_MODEL}"
else
  pull_in_compose
fi

echo "完成。对话模型: ${CHAT_MODEL}，嵌入: ${EMBED_MODEL}"
