"""对话推理与 RAG 编排。

默认经 **Ollama** 生成回复（`OLLAMA_BASE_URL` 默认为 `http://127.0.0.1:11434`），不再提供「未配置则仅用关键词模板」的路径。
仅当 `USE_TEMPLATE_FALLBACK=true` 时，Ollama 失败才回退模板。风险等级仍可用关键词启发式。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from api.config import get_settings
from services.inference_errors import InferenceUnavailableError
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


def _template_reply(risk_level: int) -> str:
    return MOCK_REPLIES.get(risk_level, "我在这里，愿意听你说更多。")


async def infer(message: str, session_id: str) -> dict[str, Any]:
    start_time = time.perf_counter()
    cfg = get_settings()
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
        reply = (payload.get("text") or "").strip() or _template_reply(risk_level)
        model_version = "remote-mlx"
        inference_time_ms = int(payload.get("inference_time_ms") or 0)
        if inference_time_ms <= 0:
            inference_time_ms = int((time.perf_counter() - start_time) * 1000)
        return {
            "reply": reply,
            "risk_level": risk_level,
            "confidence": confidence,
            "model_version": model_version,
            "inference_time_ms": inference_time_ms,
        }

    ollama_base = cfg.ollama_base_url.rstrip("/")
    chat_model = cfg.ollama_chat_model.strip()
    embed_model = cfg.ollama_embed_model.strip()
    collection = (cfg.qdrant_rag_collection or "").strip()
    top_k = cfg.qdrant_rag_top_k
    allow_fallback = cfg.use_template_fallback

    try:
        rag_block = ""
        if collection:
            try:
                rag_block = await retrieve_rag_context(
                    query=message,
                    ollama_base_url=ollama_base,
                    embed_model=embed_model,
                    qdrant_host=cfg.qdrant_host,
                    qdrant_port=cfg.qdrant_port,
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
                f"{ollama_base}/api/chat",
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
        raw_reply = (msg.get("content") or "").strip()
        if not raw_reply:
            if allow_fallback:
                reply = _template_reply(risk_level)
                model_version = "v1.0-fallback-empty"
            else:
                raise InferenceUnavailableError("Ollama 返回空内容")
        else:
            reply = raw_reply
            model_version = f"ollama:{chat_model}"
        inference_time_ms = int((time.perf_counter() - start_time) * 1000)
    except InferenceUnavailableError:
        raise
    except Exception as exc:
        if allow_fallback:
            logger.warning("Ollama 调用失败，回退模板回复: %s", exc)
            reply = _template_reply(risk_level)
            model_version = "v1.0-fallback"
            inference_time_ms = int((time.perf_counter() - start_time) * 1000)
        else:
            raise InferenceUnavailableError(f"Ollama 不可用: {exc}") from exc

    return {
        "reply": reply,
        "risk_level": risk_level,
        "confidence": confidence,
        "model_version": model_version,
        "inference_time_ms": inference_time_ms,
    }
