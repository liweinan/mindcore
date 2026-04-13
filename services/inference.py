"""推理：关键词基线（风险）；回复优先级 — 远程 INFERENCE_URL > 本地 Ollama > 模板句。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from services.rag import retrieve_rag_context

logger = logging.getLogger(__name__)

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
    ollama_base = (os.getenv("OLLAMA_BASE_URL") or "").strip()

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
    elif ollama_base:
        try:
            chat_model = (os.getenv("OLLAMA_CHAT_MODEL") or "qwen2.5:3b").strip()
            embed_model = (os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text").strip()
            collection = (os.getenv("QDRANT_RAG_COLLECTION") or "").strip()
            qdrant_host = (os.getenv("QDRANT_HOST") or "localhost").strip()
            qdrant_port = int(os.getenv("QDRANT_PORT") or "6333")
            top_k = int(os.getenv("QDRANT_RAG_TOP_K") or "3")

            rag_block = ""
            if collection:
                try:
                    rag_block = await retrieve_rag_context(
                        query=message,
                        ollama_base_url=ollama_base,
                        embed_model=embed_model,
                        qdrant_host=qdrant_host,
                        qdrant_port=qdrant_port,
                        collection=collection,
                        top_k=top_k,
                    )
                except Exception as exc:
                    logger.warning("RAG 检索失败，继续无知识片段生成: %s", exc)

            system_parts = [
                "你是心理健康方向的倾听与自助支持助手（非执业医师），回复简洁、共情、避免下诊断或开药。",
                f"当前会话 id：{session_id}。",
            ]
            if rag_block:
                system_parts.append(rag_block)
            system_prompt = "\n".join(system_parts)

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{ollama_base.rstrip('/')}/api/chat",
                    json={
                        "model": chat_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": message},
                        ],
                        "stream": False,
                        "options": {"temperature": 0.6},
                    },
                )
                response.raise_for_status()
                payload = response.json()
            msg = payload.get("message") or {}
            reply = (msg.get("content") or "").strip() or MOCK_REPLIES.get(
                risk_level, "我在这里，愿意听你说更多。"
            )
            model_version = f"ollama:{chat_model}"
        except Exception as exc:
            logger.warning("Ollama 调用失败，回退模板回复: %s", exc)
            reply = MOCK_REPLIES.get(risk_level, "我在这里，愿意听你说更多。")
            model_version = "v1.0-fallback"
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
