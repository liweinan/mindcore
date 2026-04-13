"""对运行中的 API 做 HTTP 级 E2E（需基础设施与 uvicorn 已启动）。"""

from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.getenv("MINDCORE_E2E_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as http_client:
        yield http_client


@pytest.fixture(scope="module", autouse=True)
def _require_running_api(client: httpx.Client) -> None:
    try:
        response = client.get("/health", timeout=3.0)
    except httpx.ConnectError as exc:
        pytest.skip(f"未连上 API（{BASE_URL}），请先启动服务: {exc}")
    if response.status_code != 200:
        pytest.skip(f"GET /health 非 200: {response.status_code}")


@pytest.fixture(scope="module")
def chat_ok(client: httpx.Client) -> None:
    probe = client.post(
        "/v1/chat",
        json={"user_id": "_e2e_chat_probe", "message": "ping"},
    )
    if probe.status_code != 200:
        pytest.skip(
            f"POST /v1/chat 返回 {probe.status_code}：请启动 Ollama（默认 http://127.0.0.1:11434）"
            f"或在 .env 中设 USE_TEMPLATE_FALLBACK=true 后重启 API。响应: {probe.text[:400]}"
        )


def test_health(client: httpx.Client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") == "ok"
    assert "timestamp" in body


def test_ready(client: httpx.Client) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json().get("status") == "ready"


def test_metrics(client: httpx.Client) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mindcore" in response.text or "python" in response.text.lower()


def test_chat_new_session(client: httpx.Client, chat_ok: None) -> None:
    payload = {
        "user_id": "e2e_user",
        "message": "最近总是睡不着，感觉做什么都没意思，不想见人",
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "session_id" in data
    assert data.get("reply")
    assert 1 <= int(data.get("risk_level", 0)) <= 5
    assert 0.0 <= float(data.get("confidence", -1)) <= 1.0
    assert int(data.get("inference_time_ms", -1)) >= 0


def test_chat_continue_session(client: httpx.Client, chat_ok: None) -> None:
    first = client.post(
        "/v1/chat",
        json={"user_id": "e2e_user2", "message": "你好"},
    )
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    second = client.post(
        "/v1/chat",
        json={"user_id": "e2e_user2", "message": "今天天气不错", "session_id": session_id},
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
