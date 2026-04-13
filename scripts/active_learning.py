#!/usr/bin/env python3
"""按置信度与风险筛选高价值样本并写入 annotation_tasks（幂等：冲突则跳过）。"""

from __future__ import annotations

import asyncio
import os

import asyncpg


async def select_high_value_samples() -> None:
    database_url = os.getenv("DATABASE_URL", "postgresql://admin:secret@localhost:5432/mental_health")
    conn = await asyncpg.connect(database_url)
    try:
        low_confidence = await conn.fetch(
            """
            SELECT m.id, m.content, m.confidence, m.risk_level
            FROM messages m
            LEFT JOIN annotation_tasks a ON m.id = a.message_id
            WHERE m.role = 'user'
              AND a.id IS NULL
              AND m.confidence IS NOT NULL
              AND m.confidence BETWEEN 0.4 AND 0.6
            LIMIT 100
            """
        )

        high_risk_low_confidence = await conn.fetch(
            """
            SELECT m.id, m.content, m.confidence, m.risk_level
            FROM messages m
            LEFT JOIN annotation_tasks a ON m.id = a.message_id
            WHERE m.role = 'user'
              AND a.id IS NULL
              AND m.risk_level >= 4
              AND m.confidence IS NOT NULL
              AND m.confidence < 0.8
            LIMIT 50
            """
        )

        merged: list = []
        seen: set = set()
        for row in list(low_confidence) + list(high_risk_low_confidence):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            merged.append(row)

        for sample in merged:
            risk = int(sample["risk_level"] or 0)
            priority = 1 if risk >= 4 else 5
            await conn.execute(
                """
                INSERT INTO annotation_tasks (message_id, priority, status)
                VALUES ($1, $2, 'pending')
                ON CONFLICT (message_id) DO NOTHING
                """,
                sample["id"],
                priority,
            )

        print(f"本轮处理 {len(merged)} 条候选，已尝试创建标注任务（重复 message_id 会自动跳过）。")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(select_high_value_samples())
