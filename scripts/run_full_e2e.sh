#!/usr/bin/env bash
# 全链路验证：API 健康 → 真实 /v1/chat → HTTP E2E → Playwright（含 UI 截图）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE_URL="${MINDCORE_E2E_BASE_URL:-http://127.0.0.1:8000}"

echo "=== 1/4 检查 API ==="
curl -sf --noproxy '*' "${BASE_URL}/health" >/dev/null
curl -sf --noproxy '*' "${BASE_URL}/ready" >/dev/null
echo "API 就绪: ${BASE_URL}"

echo "=== 2/4 真实对话 smoke ==="
curl -sf --noproxy '*' --max-time 180 -X POST "${BASE_URL}/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"full_e2e","message":"用一句话用中文说晚安"}' | python3 -c "
import json, sys
data = json.load(sys.stdin)
assert data.get('reply'), 'reply 为空'
assert 1 <= int(data.get('risk_level', 0)) <= 5
print('reply:', data['reply'][:80] + ('…' if len(data['reply']) > 80 else ''))
print('risk_level:', data['risk_level'], 'inference_time_ms:', data['inference_time_ms'])
"

PYTEST="${ROOT}/.venv/bin/pytest"
if [ ! -x "$PYTEST" ]; then
  PYTEST="uv run pytest"
fi

echo "=== 3/4 HTTP E2E ==="
$PYTEST tests/test_e2e_api.py -v

echo "=== 4/4 Playwright（含截图）==="
$PYTEST tests/test_frontend_playwright.py -v --browser chromium

echo ""
echo "全链路验证完成。截图目录: docs/screenshots/"
