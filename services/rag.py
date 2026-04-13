"""Qdrant RAG：查询向量默认用 Ollama /api/embeddings（如 nomic-embed-text，768 维）。"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from qdrant_client import QdrantClient

from api.config import get_settings

logger = logging.getLogger(__name__)

OLLAMA_EMBED_CLIENT_TIMEOUT_SEC = 120.0


async def ollama_embed(client: httpx.AsyncClient, base_url: str, model: str, text: str) -> list[float]:
    url = f"{base_url.rstrip('/')}/api/embeddings"
    response = await client.post(url, json={"model": model, "prompt": text})
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    embedding = payload.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Ollama 返回中缺少 embedding 数组")
    return [float(x) for x in embedding]


async def retrieve_rag_context(
    *,
    query: str,
    ollama_base_url: str,
    embed_model: str,
    qdrant_host: str,
    qdrant_port: int,
    collection: str,
    top_k: int,
) -> str:
    if not collection.strip():
        return ""
    async with httpx.AsyncClient(
        timeout=OLLAMA_EMBED_CLIENT_TIMEOUT_SEC,
        trust_env=False,
    ) as http_client:
        vector = await ollama_embed(http_client, ollama_base_url, embed_model, query)

    debug_log = get_settings().inference_debug_log
    if debug_log:
        logger.info(
            "RAG 嵌入模型=%s 向量维度=%s 向量(JSON)=%s",
            embed_model,
            len(vector),
            json.dumps(vector, ensure_ascii=False),
        )

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )
    if debug_log:
        rows: list[dict[str, Any]] = []
        for hit in hits:
            rows.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
            )
        logger.info(
            "RAG Qdrant collection=%s top_k=%s 命中=%s",
            collection,
            top_k,
            json.dumps(rows, ensure_ascii=False, default=str),
        )
    chunks: list[str] = []
    for hit in hits:
        payload = hit.payload or {}
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content.strip())
    if not chunks:
        return ""
    numbered = "\n".join(f"- {c}" for c in chunks)
    return f"以下是与用户问题相关的知识片段（仅供辅助，非诊断依据）：\n{numbered}\n"
