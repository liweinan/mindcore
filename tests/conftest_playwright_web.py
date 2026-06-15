"""Playwright 测试共用 fixture。"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import FileResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
WEB_DIST = WEB_DIR / "dist"


def build_web_if_needed() -> None:
    index_file = WEB_DIST / "index.html"
    if index_file.is_file():
        return
    if not (WEB_DIR / "package.json").is_file():
        raise RuntimeError("未找到 web/package.json，无法构建 React 前端")
    subprocess.run(["npm", "install"], cwd=WEB_DIR, check=True)
    subprocess.run(["npm", "run", "build"], cwd=WEB_DIR, check=True)
    if not index_file.is_file():
        raise RuntimeError("前端构建失败，未生成 web/dist/index.html")


def _make_web_app() -> Starlette:
    build_web_if_needed()
    routes: list[Route | Mount] = [
        Route("/", lambda request: FileResponse(WEB_DIST / "index.html")),
    ]
    assets_dir = WEB_DIST / "assets"
    if assets_dir.is_dir():
        routes.append(Mount("/assets", StaticFiles(directory=assets_dir), name="assets"))
    return Starlette(routes=routes)


app = _make_web_app()


@pytest.fixture(scope="session", autouse=True)
def _ensure_react_build() -> None:
    build_web_if_needed()


def _pick_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(base_url: str, timeout_sec: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/", timeout=1.0, trust_env=False)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError(f"测试 Web 服务未在 {timeout_sec}s 内就绪: {base_url}")


def start_test_web_server() -> tuple[str, subprocess.Popen[bytes]]:
    build_web_if_needed()
    global app
    app = _make_web_app()
    port = _pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.conftest_playwright_web:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(PROJECT_ROOT),
    )
    try:
        _wait_until_ready(base_url)
    except Exception:
        process.terminate()
        process.wait(timeout=5)
        raise
    return base_url, process


def stop_test_web_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
