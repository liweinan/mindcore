from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import asyncpg
from fastapi import BackgroundTasks, FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from api.config import settings
from services.inference import infer

logger = logging.getLogger(__name__)

REQUEST_LATENCY = Histogram(
    "mindcore_chat_latency_seconds",
    "Latency of /v1/chat handler",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
CHAT_REQUESTS = Counter("mindcore_chat_requests_total", "Total /v1/chat requests", ["status"])


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=64)
    audio_url: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    risk_level: int
    confidence: float
    inference_time_ms: int


db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10)
    yield
    if db_pool is not None:
        await db_pool.close()
        db_pool = None


app = FastAPI(title="MindCore", version="1.0.0", lifespan=lifespan)


async def create_annotation_task(message_id: str, predicted_risk: int) -> None:
    assert db_pool is not None
    priority = max(1, min(10, 10 - predicted_risk))
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO annotation_tasks (message_id, priority, status)
            VALUES ($1, $2, 'pending')
            ON CONFLICT (message_id) DO NOTHING
            """,
            uuid.UUID(message_id),
            priority,
        )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    assert db_pool is not None
    with REQUEST_LATENCY.time():
        session_id_str = request.session_id or str(uuid.uuid4())
        try:
            session_uuid = uuid.UUID(session_id_str)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="session_id 不是合法 UUID") from exc

        async with db_pool.acquire() as conn:
            if request.session_id is None:
                await conn.execute(
                    "INSERT INTO sessions (id, user_id, metadata) VALUES ($1, $2, '{}'::jsonb)",
                    session_uuid,
                    request.user_id,
                )
            else:
                row = await conn.fetchrow("SELECT id FROM sessions WHERE id = $1", session_uuid)
                if row is None:
                    raise HTTPException(status_code=404, detail="session 不存在")

        result = await infer(request.message, session_id_str)
        user_msg_id = uuid.uuid4()
        assistant_msg_id = uuid.uuid4()

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO messages (
                        id, session_id, role, content, risk_level, has_audio, audio_url,
                        confidence, model_version, inference_time_ms
                    )
                    VALUES ($1, $2, 'user', $3, $4, $5, $6, $7, NULL, NULL)
                    """,
                    user_msg_id,
                    session_uuid,
                    request.message,
                    result["risk_level"],
                    request.audio_url is not None,
                    request.audio_url,
                    result["confidence"],
                )
                await conn.execute(
                    """
                    INSERT INTO messages (
                        id, session_id, role, content, risk_level, confidence,
                        model_version, inference_time_ms
                    )
                    VALUES ($1, $2, 'assistant', $3, NULL, NULL, $4, $5)
                    """,
                    assistant_msg_id,
                    session_uuid,
                    result["reply"],
                    result["model_version"],
                    result["inference_time_ms"],
                )

        if result["confidence"] < 0.7:
            background_tasks.add_task(create_annotation_task, str(user_msg_id), int(result["risk_level"]))

        CHAT_REQUESTS.labels(status="ok").inc()
        return ChatResponse(
            session_id=session_id_str,
            reply=result["reply"],
            risk_level=int(result["risk_level"]),
            confidence=float(result["confidence"]),
            inference_time_ms=int(result["inference_time_ms"]),
        )


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/ready")
async def ready():
    if db_pool is None:
        raise HTTPException(status_code=503, detail="数据库未就绪")
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    except Exception as exc:
        logger.exception("readiness check failed")
        raise HTTPException(status_code=503, detail="数据库不可用") from exc
    return {"status": "ready"}
