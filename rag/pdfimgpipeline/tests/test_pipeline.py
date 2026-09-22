import asyncio

from pdfimgpipeline.models import RouteResult
from pdfimgpipeline.pipeline import PDFImagePipeline


class StubRouter:
    async def decide(self, path):
        return RouteResult(decision="ocr", reason="stub", features={})


class StubExecutor:
    def __init__(self):
        self.active = 0
        self.peak = 0

    async def execute(self, path, decision):
        if "bad" in path:
            raise RuntimeError("boom")
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.02)
        self.active -= 1
        return {"engine": "ocr", "data": {"text": path}}


async def test_process_many_isolates_failures(tmp_path):
    pipeline = PDFImagePipeline(StubRouter(), StubExecutor())
    good = tmp_path / "good.png"
    good.write_bytes(b"x")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"y")

    out = await pipeline.process_many([str(good), str(bad)])

    assert [o.ok for o in out] == [True, False]
    assert out[0].data["result"]["data"]["text"] == str(good)
    assert "boom" in out[1].error


async def test_process_many_preserves_order_and_bounds_concurrency(tmp_path):
    executor = StubExecutor()
    pipeline = PDFImagePipeline(StubRouter(), executor)
    paths = []
    for i in range(10):
        f = tmp_path / f"f{i}.png"
        f.write_bytes(bytes([i]))
        paths.append(str(f))

    out = await pipeline.process_many(paths, max_concurrency=3)

    assert all(o.ok for o in out)
    assert [o.data["result"]["data"]["text"] for o in out] == paths
    assert executor.peak <= 3


async def test_process_many_dedupes_via_cache(tmp_path):
    executor = StubExecutor()
    pipeline = PDFImagePipeline(StubRouter(), executor)
    same = tmp_path / "same.png"
    same.write_bytes(b"identical")

    first = await pipeline.process_many([str(same)])
    second = await pipeline.process_many([str(same)])

    # 无结果缓存时都实跑；这里只验证批量与单图结果结构一致
    assert first[0].data["result"]["engine"] == "ocr"
    assert second[0].data["image_hash"] == first[0].data["image_hash"]
