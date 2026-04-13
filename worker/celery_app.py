import os

from celery import Celery

from api.config import settings

os.environ.setdefault("CELERY_BROKER_URL", settings.celery_broker_url)
os.environ.setdefault("CELERY_RESULT_BACKEND", settings.celery_result_backend)

celery_app = Celery(
    "mindcore",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

from worker import tasks  # noqa: E402, F401
