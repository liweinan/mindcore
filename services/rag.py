"""Qdrant RAG：查询向量默认用 Ollama /api/embeddings（如 nomic-embed-text，768 维）。"""

from __future__ import annotations

from typing import Any

import httpx
from qdrant_client import QdrantClient


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
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        vector = await ollama_embed(http_client, ollama_base_url, embed_model, query)

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    hits = client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
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
