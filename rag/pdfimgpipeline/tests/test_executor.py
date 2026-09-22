import asyncio

from pdfimgpipeline.executor import ExtractorExecutor


async def test_vlm_timeout_falls_back_to_ocr():
    async def ocr(path):
        return {"text": "ocr-ok"}

    async def vlm(path):
        await asyncio.sleep(0.2)
        return {"text": "slow"}

    ex = ExtractorExecutor(ocr, vlm, vlm_timeout=0.05, ocr_timeout=1.0, max_retries=0)
    out = await ex.execute("x.png", "vlm")
    assert out["engine"] == "ocr"
    assert out["fallback_from"] == "vlm"
    assert out["data"]["text"] == "ocr-ok"


async def test_retry_then_success():
    calls = {"n": 0}

    async def ocr(path):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return {"text": "ok"}

    async def vlm(path):
        return {"text": ""}

    ex = ExtractorExecutor(ocr, vlm, max_retries=2, retry_backoff=0.001)
    out = await ex.execute("x.png", "ocr")
    assert out["data"]["text"] == "ok"
    assert calls["n"] == 3


async def test_non_retryable_error_is_not_retried():
    calls = {"n": 0}

    async def ocr(path):
        calls["n"] += 1
        raise ValueError("bad input")

    async def vlm(path):
        return {"text": ""}

    ex = ExtractorExecutor(ocr, vlm, max_retries=3, retry_backoff=0.001)
    out = await ex.execute("x.png", "ocr")
    assert out["data"] is None
    assert "bad input" in out["error"]
    assert calls["n"] == 1


async def test_hybrid_merges_both():
    async def ocr(path):
        return {"text": "ocr-text"}

    async def vlm(path):
        return {"text": "vlm-desc"}

    ex = ExtractorExecutor(ocr, vlm)
    out = await ex.execute("x.png", "hybrid")
    assert out["engine"] == "hybrid"
    assert out["data"]["ocr_text"] == "ocr-text"
    assert out["data"]["vlm_description"] == "vlm-desc"


async def test_hybrid_survives_one_side_failing():
    async def ocr(path):
        raise RuntimeError("ocr down")

    async def vlm(path):
        return {"text": "vlm-desc"}

    ex = ExtractorExecutor(ocr, vlm, max_retries=0)
    out = await ex.execute("x.png", "hybrid")
    assert out["engine"] == "vlm(fallback)"
    assert out["data"]["text"] == "vlm-desc"
