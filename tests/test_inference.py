import pytest

from services.inference import RISK_KEYWORDS, infer


@pytest.mark.asyncio
async def test_infer_mock_baseline():
    result = await infer("最近总是睡不着，感觉做什么都没意思", "sess-1")
    assert result["risk_level"] >= 2
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["reply"]
    assert result["model_version"] == "v1.0"


@pytest.mark.asyncio
async def test_risk_keywords_cover_expected_phrases():
    assert "想死" in RISK_KEYWORDS
