#!/usr/bin/env python3
"""在 Qdrant 中创建 mental_health_knowledge 集合并写入示例文档。

必须通过 Ollama 的 /api/embeddings 生成真实向量（与在线 RAG 一致）：
- 必填环境变量：OLLAMA_BASE_URL（例如 http://127.0.0.1:11434）
- 可选：OLLAMA_EMBED_MODEL（默认 nomic-embed-text，768 维）
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "mental_health_knowledge"
logger = logging.getLogger("services.rag")


def ollama_embed_text(text: str, base_url: str, model: str) -> list[float]:
    url = f"{base_url.rstrip('/')}/api/embeddings"
    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json={"model": model, "prompt": text})
        response.raise_for_status()
        embedding = response.json().get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Ollama 返回中缺少 embedding")
    return [float(x) for x in embedding]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ollama_base = (os.getenv("OLLAMA_BASE_URL") or "").strip()
    if not ollama_base:
        print(
            "错误：未设置 OLLAMA_BASE_URL。本脚本仅支持通过 Ollama 嵌入接口写入真实向量。\n"
            "示例：export OLLAMA_BASE_URL=http://127.0.0.1:11434",
            file=sys.stderr,
        )
        sys.exit(1)

    embed_model = (os.getenv("OLLAMA_EMBED_MODEL") or "nomic-embed-text").strip()
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port)

    documents: list[dict] = [
        {
            "content": "抑郁症的核心症状包括：持续的情绪低落、兴趣丧失、精力减退，持续时间超过2周。",
            "tags": ["抑郁", "诊断标准"],
            "chunk_type": "knowledge",
            "source": "dsm5",
        },
        {
            "content": (
                "CBT认知重构：帮助来访者识别并挑战自动负性思维，例如'我一无是处'可以重构为'我有缺点，但也有优点'。"
            ),
            "tags": ["CBT", "认知重构"],
            "chunk_type": "intervention",
            "source": "cbt_manual",
        },
        {
            "content": "当用户表达自杀意念时，立即询问计划、评估风险，必要时转介紧急干预，不要承诺保密。",
            "tags": ["安全", "危机干预"],
            "chunk_type": "knowledge",
            "source": "expert",
        },
    ]

    vectors = [ollama_embed_text(doc["content"], ollama_base, embed_model) for doc in documents]
    for vector in vectors:
        logger.info(
            "RAG 嵌入模型=%s 向量维度=%s 向量(JSON)=%s",
            embed_model,
            len(vector),
            json.dumps(vector, ensure_ascii=False),
        )
    dim = len(vectors[0])

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points: list[PointStruct] = []
    for index, doc in enumerate(documents):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindcore:{index}")),
                vector=vectors[index],
                payload={
                    "chunk_type": doc["chunk_type"],
                    "source": doc["source"],
                    "tags": doc["tags"],
                    "content": doc["content"],
                    "embedding_model": embed_model,
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"已写入 {len(points)} 条向量到 {COLLECTION_NAME}（Ollama 嵌入 {embed_model}，维度 {dim}）")


if __name__ == "__main__":
    main()
