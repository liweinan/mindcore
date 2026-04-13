"""异步多模态等重型任务（占位实现，可接 Whisper / 视觉管线）。"""

from __future__ import annotations

import logging
import time
from typing import Any

from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3)
def process_multimodal(
    self,
    message_id: str,
    audio_url: str,
    video_url: str | None = None,
) -> dict[str, Any]:
    try:
        logger.info("process_multimodal message_id=%s audio=%s video=%s", message_id, audio_url, video_url)
        time.sleep(0.5)
        return {"status": "success", "message_id": message_id}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60) from exc
