"""对话推理编排：固定流水线——关键词风险基线 → Ollama 嵌入 + Qdrant 检索 → Ollama 对话模型。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from api.config import get_settings
from services.inference_errors import InferenceUnavailableError
from services.rag import retrieve_rag_context

logger = logging.getLogger(__name__)


def _build_http_timeout(timeout_seconds: float) -> httpx.Timeout:
    if timeout_seconds <= 0:
        return httpx.Timeout(timeout=None)
    return httpx.Timeout(timeout_seconds)


def _format_ollama_exception(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        url = str(exc.request.url)
        snippet = (exc.response.text or "")[:300].replace("\n", " ").strip()
        base = f"HTTP {exc.response.status_code} @ {url}"
        if snippet:
            return f"{base} — {snippet}"
        return base
    text = str(exc).strip()
    if text:
        return text
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        c = str(cause).strip()
        if c:
            return f"{type(exc).__name__}（底层: {c}）"
    return type(exc).__name__


RISK_KEYWORDS = ("睡不着", "没意思", "不想见人", "难过", "想死", "绝望", "不想活了")


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
    cfg = get_settings()
    collection = (cfg.qdrant_rag_collection or "").strip()
    if not collection:
        raise InferenceUnavailableError(
            "未配置 QDRANT_RAG_COLLECTION，无法执行 RAG（嵌入 + Qdrant）流程"
        )

    risk_level, confidence = _baseline_risk_and_confidence(message)

    ollama_base = cfg.ollama_base_url.rstrip("/")
    chat_model = cfg.ollama_chat_model.strip()
    embed_model = cfg.ollama_embed_model.strip()
    top_k = cfg.qdrant_rag_top_k

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
        raise InferenceUnavailableError(
            f"RAG 检索失败（嵌入或 Qdrant）: {_format_ollama_exception(exc)}"
        ) from exc

    system_parts = [
        "你是心理健康方向的倾听与自助支持助手（非执业医师），回复简洁、共情、避免下诊断或开药。",
        f"当前会话 id：{session_id}。",
    ]
    if rag_block:
        system_parts.append(rag_block)
    else:
        system_parts.append(
            "知识库检索未返回片段；请仅依据通用倾听原则回复，避免编造事实。"
        )

    system_prompt = "\n".join(system_parts)

    chat_body: dict[str, Any] = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "stream": False,
        "options": {"temperature": 0.6},
    }
    if cfg.inference_debug_log:
        logger.info(
            "Ollama chat 请求 JSON=%s",
            json.dumps(chat_body, ensure_ascii=False),
        )

    try:
        async with httpx.AsyncClient(
            timeout=_build_http_timeout(cfg.ollama_chat_timeout_sec),
            trust_env=False,
        ) as client:
            response = await client.post(
                f"{ollama_base}/api/chat",
                json=chat_body,
            )
            response.raise_for_status()
            payload = response.json()
        msg = payload.get("message") or {}
        raw_reply = (msg.get("content") or "").strip()
        if not raw_reply:
            raise InferenceUnavailableError("Ollama 返回空内容")
        reply = raw_reply
        model_version = f"ollama:{chat_model}"
        inference_time_ms = int((time.perf_counter() - start_time) * 1000)
    except InferenceUnavailableError:
        raise
    except Exception as exc:
        raise InferenceUnavailableError(
            f"Ollama 对话不可用: {_format_ollama_exception(exc)}"
        ) from exc

    return {
        "reply": reply,
        "risk_level": risk_level,
        "confidence": confidence,
        "model_version": model_version,
        "inference_time_ms": inference_time_ms,
    }
