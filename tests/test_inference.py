import pytest

from services.inference import RISK_KEYWORDS, infer


@pytest.mark.asyncio
async def test_infer_ollama_fails_with_template_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_TEMPLATE_FALLBACK", "true")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59998")
    monkeypatch.setenv("QDRANT_RAG_COLLECTION", "")
    result = await infer("最近总是睡不着，感觉做什么都没意思", "sess-1")
    assert result["risk_level"] >= 2
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reply"]
    assert result["model_version"] == "v1.0-fallback"


@pytest.mark.asyncio
async def test_infer_ollama_fails_without_fallback_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from services.inference_errors import InferenceUnavailableError

    monkeypatch.setenv("USE_TEMPLATE_FALLBACK", "false")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:59998")
    monkeypatch.setenv("QDRANT_RAG_COLLECTION", "")
    with pytest.raises(InferenceUnavailableError):
        await infer("你好", "sess-2")


def test_risk_keywords_cover_expected_phrases() -> None:
    assert "想死" in RISK_KEYWORDS
