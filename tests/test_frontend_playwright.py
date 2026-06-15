"""学习页 Playwright E2E 与 UI 截图。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, Route, expect

from tests.conftest_playwright_web import start_test_web_server, stop_test_web_server

BASE_URL = os.getenv("MINDCORE_E2E_BASE_URL", "http://127.0.0.1:8000")
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


@pytest.fixture(scope="module")
def web_base_url() -> str:
    base_url, process = start_test_web_server()
    yield base_url
    stop_test_web_server(process)


@pytest.fixture(scope="module")
def api_available() -> bool:
    try:
        with httpx.Client(base_url=BASE_URL, timeout=3.0, trust_env=False) as client:
            response = client.get("/health")
    except httpx.ConnectError:
        return False
    return response.status_code == 200


@pytest.fixture(scope="module")
def chat_available(api_available: bool) -> bool:
    if not api_available:
        return False
    with httpx.Client(base_url=BASE_URL, timeout=30.0, trust_env=False) as client:
        probe = client.post("/v1/chat", json={"user_id": "_pw_probe", "message": "ping"})
    return probe.status_code == 200


@pytest.fixture
def learn_page(page: Page, web_base_url: str) -> Page:
    page.goto(web_base_url)
    page.wait_for_load_state("networkidle")
    return page


def test_learn_page_loads(learn_page: Page) -> None:
    expect(learn_page).to_have_title("MindCore 学习对话")
    expect(learn_page.get_by_role("heading", name="MindCore 学习对话")).to_be_visible()
    expect(learn_page.get_by_test_id("user-id")).to_have_value("demo")
    expect(learn_page.get_by_test_id("message")).to_be_visible()
    expect(learn_page.get_by_test_id("send-btn")).to_have_text("发送")


def test_send_message_with_mock_api(learn_page: Page) -> None:
    def handle_chat(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=(
                '{"session_id":"11111111-1111-1111-1111-111111111111",'
                '"reply":"你好，我在这里陪你。",'
                '"risk_level":1,'
                '"confidence":0.9,'
                '"inference_time_ms":42}'
            ),
        )

    learn_page.route("**/v1/chat", handle_chat)
    learn_page.get_by_test_id("message").fill("你好")
    learn_page.get_by_test_id("send-btn").click()

    expect(learn_page.get_by_test_id("send-btn")).to_have_text("发送", timeout=10_000)
    expect(learn_page.get_by_test_id("result")).to_be_visible()
    expect(learn_page.get_by_test_id("reply")).to_have_text("你好，我在这里陪你。")
    expect(learn_page.get_by_test_id("session-id")).to_have_text(
        "11111111-1111-1111-1111-111111111111"
    )
    expect(learn_page.get_by_test_id("error")).to_have_count(0)


def test_screenshot_initial_page(learn_page: Page) -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    learn_page.screenshot(path=str(SCREENSHOT_DIR / "learn-initial.png"), full_page=True)
    assert (SCREENSHOT_DIR / "learn-initial.png").is_file()


def test_screenshot_after_reply(learn_page: Page) -> None:
    def handle_chat(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=(
                '{"session_id":"11111111-1111-1111-1111-111111111111",'
                '"reply":"你好，我在这里陪你。",'
                '"risk_level":1,'
                '"confidence":0.9,'
                '"inference_time_ms":42}'
            ),
        )

    learn_page.route("**/v1/chat", handle_chat)
    learn_page.get_by_test_id("message").fill("你好")
    learn_page.get_by_test_id("send-btn").click()
    expect(learn_page.get_by_test_id("result")).to_be_visible(timeout=10_000)

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    learn_page.screenshot(path=str(SCREENSHOT_DIR / "learn-with-reply.png"), full_page=True)
    assert (SCREENSHOT_DIR / "learn-with-reply.png").is_file()


@pytest.fixture
def live_learn_page(page: Page, api_available: bool) -> Page:
    if not api_available:
        pytest.skip(f"未连上 API（{BASE_URL}），跳过联调测试")
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    return page


def test_send_message_live_api(live_learn_page: Page, chat_available: bool) -> None:
    if not chat_available:
        pytest.skip("POST /v1/chat 不可用，需 Ollama、Qdrant 及 RAG 集合")

    live_learn_page.get_by_test_id("message").fill("你好，今天心情还可以")
    live_learn_page.get_by_test_id("send-btn").click()

    expect(live_learn_page.get_by_test_id("send-btn")).to_have_text("发送", timeout=120_000)
    expect(live_learn_page.get_by_test_id("result")).to_be_visible()
    expect(live_learn_page.get_by_test_id("reply")).not_to_be_empty()
    expect(live_learn_page.get_by_test_id("session-id")).not_to_have_text("-")
    expect(live_learn_page.get_by_test_id("error")).to_have_count(0)


def test_live_page_served_by_api(live_learn_page: Page) -> None:
    expect(live_learn_page).to_have_title("MindCore 学习对话")
    expect(live_learn_page.get_by_test_id("send-btn")).to_be_visible()


def test_screenshot_live_reply(live_learn_page: Page, chat_available: bool) -> None:
    if not chat_available:
        pytest.skip("POST /v1/chat 不可用，需 Ollama、Qdrant 及 RAG 集合")

    live_learn_page.get_by_test_id("message").fill("最近总是睡不着，感觉做什么都没意思")
    live_learn_page.get_by_test_id("send-btn").click()
    expect(live_learn_page.get_by_test_id("result")).to_be_visible(timeout=120_000)
    expect(live_learn_page.get_by_test_id("reply")).not_to_be_empty()

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    live_learn_page.screenshot(path=str(SCREENSHOT_DIR / "learn-live-reply.png"), full_page=True)
    assert (SCREENSHOT_DIR / "learn-live-reply.png").is_file()
