from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://admin:secret@localhost:5432/mental_health"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen2.5:0.5b-instruct-q2_K"
    ollama_embed_model: str = "nomic-embed-text"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_rag_collection: str = ""
    qdrant_rag_top_k: int = 3
    use_template_fallback: bool = False
    # 为 true 时打印嵌入向量、Qdrant 命中与 Ollama chat 请求体（含用户消息，仅用于排障）
    inference_debug_log: bool = True

    @field_validator("ollama_base_url", mode="before")
    @classmethod
    def normalize_ollama_base(cls, value: object) -> str:
        if value is None:
            return "http://127.0.0.1:11434"
        text = str(value).strip()
        return text if text else "http://127.0.0.1:11434"


def get_settings() -> Settings:
    """每次调用重新读环境，便于测试与热更新 .env。"""
    return Settings()


settings = Settings()
