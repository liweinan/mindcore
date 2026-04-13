#!/usr/bin/env python3
"""在 Qdrant 中创建 mental_health_knowledge 集合并写入示例文档（确定性伪向量，便于无 GPU 环境）。"""

from __future__ import annotations

import os
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "mental_health_knowledge"
VECTOR_DIM = 768


def deterministic_unit_vector(seed: int, dim: int = VECTOR_DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vector)) + 1e-9
    vector /= norm
    return vector.tolist()


def main() -> None:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

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

    points: list[PointStruct] = []
    for index, doc in enumerate(documents):
        points.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"mindcore:{index}")),
                vector=deterministic_unit_vector(index + 1),
                payload={
                    "chunk_type": doc["chunk_type"],
                    "source": doc["source"],
                    "tags": doc["tags"],
                    "content": doc["content"],
                    "embedding_model": "deterministic-seed",
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"已写入 {len(points)} 条向量到 {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
