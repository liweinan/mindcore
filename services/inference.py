"""推理：默认规则/关键词基线；可选通过 INFERENCE_URL 调用外部生成服务。"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

RISK_KEYWORDS = ("睡不着", "没意思", "不想见人", "难过", "想死", "绝望", "不想活了")
MOCK_REPLIES: dict[int, str] = {
    1: "听起来状态还不错，保持好的生活习惯很重要哦。",
    2: "有些小困扰是正常的，要不要试试深呼吸？",
    3: "你愿意聊这些已经很棒了，我们可以一起想想解决办法。",
    4: "这听起来确实很不容易，我在这里陪着你。",
    5: "你的感受很重要，请记得你不是一个人在面对。",
}


def _baseline_risk_and_confidence(message: str) -> tuple[int, float]:
    risk_level = sum(1 for kw in RISK_KEYWORDS if kw in message)
    risk_level = min(5, max(1, risk_level))
    if risk_level in (1, 5):
        confidence = 0.9
    else:
        confidence = 0.6
    return risk_level, confidence


async def infer(message: str, session_id: str) -> dict[str, Any]:
    start_time = time.perf_counter()
    inference_url = (os.getenv("INFERENCE_URL") or "").strip()
    use_mock = os.getenv("USE_MOCK_INFERENCE", "true").lower() in ("1", "true", "yes")

    risk_level, confidence = _baseline_risk_and_confidence(message)
    reply: str
    model_version: str
    inference_time_ms: int

    if inference_url and not use_mock:
        prompt = (
            f"会话 {session_id}。\n用户说：{message}\n"
            "你是持证心理咨询师风格的助手，回复简短、共情、安全，不要诊断。"
        )
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{inference_url.rstrip('/')}/generate",
                json={"prompt": prompt, "max_tokens": 256, "temperature": 0.7},
            )
            response.raise_for_status()
            payload = response.json()
        reply = (payload.get("text") or "").strip() or MOCK_REPLIES.get(
            risk_level, "我在这里，愿意听你说更多。"
        )
        model_version = "remote-mlx"
        inference_time_ms = int(payload.get("inference_time_ms") or 0)
        if inference_time_ms <= 0:
            inference_time_ms = int((time.perf_counter() - start_time) * 1000)
    else:
        reply = MOCK_REPLIES.get(risk_level, "我在这里，愿意听你说更多。")
        model_version = "v1.0"
        inference_time_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "reply": reply,
        "risk_level": risk_level,
        "confidence": confidence,
        "model_version": model_version,
        "inference_time_ms": inference_time_ms,
    }
