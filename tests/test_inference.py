from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from services.inference import RISK_KEYWORDS, infer
from services.inference_errors import InferenceUnavailableError


@pytest.mark.asyncio
async def test_infer_pipeline_rag_then_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_RAG_COLLECTION", "test-collection")
    with patch("services.inference.retrieve_rag_context", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = "以下是与用户问题相关的知识片段（仅供辅助，非诊断依据）：\n- 片段A\n"
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "模型回复  "}}
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__.return_value = mock_http
        mock_http.__aexit__.return_value = None
        with patch("services.inference.httpx.AsyncClient", return_value=mock_http):
            result = await infer("你好", "sess-1")
    assert result["reply"] == "模型回复"
    assert result["risk_level"] == 1
    assert result["confidence"] == 0.9
    assert result["model_version"].startswith("ollama:")
    assert result["inference_time_ms"] >= 0


@pytest.mark.asyncio
async def test_infer_missing_collection_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_RAG_COLLECTION", "")
    with pytest.raises(InferenceUnavailableError) as raised:
        await infer("你好", "sess-2")
    assert "QDRANT_RAG_COLLECTION" in raised.value.message


@pytest.mark.asyncio
async def test_infer_rag_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_RAG_COLLECTION", "c")
    with patch(
        "services.inference.retrieve_rag_context",
        new_callable=AsyncMock,
        side_effect=RuntimeError("embed failed"),
    ):
        with pytest.raises(InferenceUnavailableError) as raised:
            await infer("你好", "sess-3")
    assert "RAG" in raised.value.message


@pytest.mark.asyncio
async def test_infer_ollama_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_RAG_COLLECTION", "c")
    with patch("services.inference.retrieve_rag_context", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = "- x\n"
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_http.__aenter__.return_value = mock_http
        mock_http.__aexit__.return_value = None
        with patch("services.inference.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(InferenceUnavailableError) as raised:
                await infer("你好", "sess-4")
    assert raised.value.message.startswith("Ollama 对话不可用:")


@pytest.mark.asyncio
async def test_infer_empty_ollama_reply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_RAG_COLLECTION", "c")
    with patch("services.inference.retrieve_rag_context", new_callable=AsyncMock) as mock_rag:
        mock_rag.return_value = "- x\n"
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": ""}}
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__.return_value = mock_http
        mock_http.__aexit__.return_value = None
        with patch("services.inference.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(InferenceUnavailableError) as raised:
                await infer("你好", "sess-5")
    assert "空内容" in raised.value.message


def test_risk_keywords_cover_expected_phrases() -> None:
    assert "想死" in RISK_KEYWORDS
