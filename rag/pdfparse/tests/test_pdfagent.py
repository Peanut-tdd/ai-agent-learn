"""pdfagent 测试：预检 / 路由 / 版本化缓存 / curl 图片客户端 / 顶层 Pipeline。

全部离线：合成 PDF、内存缓存、假 curl 子进程、假解析器，不触网、不依赖 Redis。
运行：
    cd 02-rag && .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass

import pymupdf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdfagent.cache import MemoryCache, NullCache, RouteCache, ResultCache, build_cache
from pdfagent.config import Settings
from pdfagent.image_client import PdfImgPipelineClient, to_description
from pdfagent.models import KIND_TO_ENGINE, PageFeature, PdfKind
from pdfagent.pipeline import PdfPipeline
from pdfagent.precheck import PdfPrechecker
from pdfagent.router import PdfRouter


# ---------------------------------------------------------------- PDF 合成
def make_text_pdf(path: str, n_pages: int = 2, chars: int = 300) -> None:
    doc = pymupdf.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            f"Page {i} " + "content " * (chars // 8),
            fontsize=11,
        )
    doc.save(path)
    doc.close()


def make_image_pdf(path: str, n_pages: int = 1) -> None:
    doc = pymupdf.open()
    for _ in range(n_pages):
        page = doc.new_page(width=300, height=400)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100))
        pix.set_rect(pix.irect, (180, 180, 180))
        page.insert_image(page.rect, pixmap=pix)
    doc.save(path)
    doc.close()


def make_sparse_page_pdf(path: str) -> None:
    """两页满文本 + 一页仅 30 字符 -> 混合型。"""
    doc = pymupdf.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            f"Full page {i} " + "content " * 40,
            fontsize=11,
        )
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), "short note only", fontsize=11)
    doc.save(path)
    doc.close()


class PrecheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.settings = Settings.from_env(env_file=None)
        self.prechecker = PdfPrechecker(self.settings)

    def tearDown(self):
        self.tmp.cleanup()

    def _path(self, name: str) -> str:
        return os.path.join(self.tmp.name, name)

    def test_text_pdf(self):
        p = self._path("text.pdf")
        make_text_pdf(p)
        result = self.prechecker.precheck(p)
        self.assertEqual(result.kind, PdfKind.TEXT.value)
        self.assertEqual(KIND_TO_ENGINE[PdfKind(result.kind)], "pymupdf")
        self.assertEqual(result.blank_pages, 0)
        self.assertGreaterEqual(result.text_ratio, self.settings.text_ratio_threshold)

    def test_scanned_pdf(self):
        p = self._path("scanned.pdf")
        make_image_pdf(p)
        result = self.prechecker.precheck(p)
        self.assertEqual(result.kind, PdfKind.SCANNED.value)
        self.assertEqual(KIND_TO_ENGINE[PdfKind(result.kind)], "mineru")
        self.assertEqual(result.scanned_pages, 1)

    def test_hybrid_pdf(self):
        p = self._path("hybrid.pdf")
        make_sparse_page_pdf(p)
        result = self.prechecker.precheck(p)
        self.assertEqual(result.kind, PdfKind.HYBRID.value)
        self.assertEqual(KIND_TO_ENGINE[PdfKind(result.kind)], "mineru")
        self.assertEqual(result.hybrid_pages, 1)
        self.assertIn("hybrid", result.reason)

    def test_blank_document_defaults_to_text(self):
        doc = pymupdf.open()
        doc.new_page()
        path = self._path("blank.pdf")
        doc.save(path)
        doc.close()
        result = self.prechecker.precheck(path)
        self.assertEqual(result.kind, PdfKind.TEXT.value)
        self.assertEqual(result.blank_pages, 1)


# ---------------------------------------------------------------- 缓存
class CacheTests(unittest.TestCase):
    def test_memory_ttl_and_delete(self):
        cache = MemoryCache()
        cache.set("k", "v", ttl=0)
        self.assertEqual(cache.get("k"), "v")
        cache.delete("k")
        self.assertIsNone(cache.get("k"))

    def test_versioned_key_contains_namespace_and_version(self):
        settings = Settings.from_env(env_file=None)  # 无 .env
        cache = RouteCache(MemoryCache(), settings)
        key = cache.key("abc123")
        self.assertTrue(key.startswith(f"{settings.cache_namespace}:{settings.resolved_cache_version}:route:"))
        self.assertTrue(key.endswith("abc123"))

    def test_version_change_invalidates(self):
        backend = MemoryCache()
        base = Settings.from_env(env_file=None).replace(cache_version="v1")
        v1 = RouteCache(backend, base)
        v1.set("hash", {"decision": "pymupdf"})
        v2 = RouteCache(backend, base.replace(cache_version="v2"))
        self.assertIsNone(v2.get("hash"))
        self.assertEqual(v1.get("hash"), {"decision": "pymupdf"})

    def test_auto_falls_back_to_memory(self):
        # redis 未安装或未启动时，auto 必须降级为内存缓存而不是抛错
        settings = Settings.from_env(env_file=None).replace(
            cache_backend="auto", redis_host="127.0.0.1", redis_port=1,
            redis_socket_timeout=0.2,
        )
        cache = build_cache(settings)
        self.assertIn(cache.name, ("memory", "redis"))
        cache.close()

    def test_none_backend(self):
        self.assertIsInstance(
            build_cache(Settings.from_env(env_file=None).replace(cache_backend="none")),
            NullCache,
        )


# ---------------------------------------------------------------- 路由
class _FakePrechecker:
    def __init__(self, kind: PdfKind):
        self.kind = kind
        self.calls = 0

    def precheck(self, pdf_path, pdf_hash=""):
        from pdfagent.models import PrecheckResult

        self.calls += 1
        return PrecheckResult(
            pdf_hash=pdf_hash, pages=1, text_pages=1, hybrid_pages=0,
            scanned_pages=0, blank_pages=0, text_ratio=1.0, scanned_ratio=0.0,
            kind=self.kind.value, reason=f"fake:{self.kind.value}",
        )


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf = os.path.join(self.tmp.name, "a.pdf")
        make_text_pdf(self.pdf, n_pages=1)
        self.settings = Settings.from_env(env_file=None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_hit(self):
        pc = _FakePrechecker(PdfKind.TEXT)
        router = PdfRouter(pc, RouteCache(MemoryCache(), self.settings))
        first = router.decide(self.pdf)
        second = router.decide(self.pdf)
        self.assertEqual(first.decision, "pymupdf")
        self.assertFalse(first.from_cache)
        self.assertTrue(second.from_cache)
        self.assertEqual(pc.calls, 1)

    def test_force_mineru(self):
        pc = _FakePrechecker(PdfKind.TEXT)
        router = PdfRouter(pc, RouteCache(MemoryCache(), self.settings))
        result = router.decide(self.pdf, force="mineru")
        self.assertEqual(result.decision, "mineru")
        self.assertTrue(result.forced)
        self.assertEqual(pc.calls, 1)  # 仍做预检以记录证据

    def test_unknown_force(self):
        router = PdfRouter(_FakePrechecker(PdfKind.TEXT), RouteCache(MemoryCache(), self.settings))
        with self.assertRaises(ValueError):
            router.decide(self.pdf, force="nope")


# ---------------------------------------------------------------- 图片客户端
class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _batch_stdout(paths):
    results = []
    for i, path in enumerate(paths):
        stem = os.path.basename(path)
        if "bad" in stem:
            results.append({"index": i, "filename": stem, "ok": False,
                            "data": None, "error": "cannot read image"})
        else:
            results.append({"index": i, "filename": stem, "ok": True,
                            "data": {"image_hash": f"h{i}", "decision": "hybrid",
                                     "reason": "", "features": None,
                                     "result": {"engine": "hybrid", "data": {
                                         "ocr_text": f"ocr-{i}",
                                         "vlm_description": f"vlm-{i}",
                                         "merged": "m"}},
                                     "from_cache": False},
                            "error": None})
    return json.dumps({"count": len(paths), "ok_count": len(paths) - 1,
                       "failed_count": 1, "results": results})


class ImageClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.from_env(env_file=None).replace(
            pdfimg_batch_size=2, pdfimg_max_concurrency=2, pdfimg_max_retries=0,
        )

    def test_to_description_variants(self):
        self.assertEqual(
            to_description({"result": {"engine": "ocr", "data": {"text": "hi"}}}),
            {"ocr_text": "hi", "vlm_description": ""},
        )
        self.assertEqual(
            to_description({"result": {"engine": "vlm", "data": {"text": "desc"}}}),
            {"ocr_text": "", "vlm_description": "desc"},
        )
        self.assertEqual(
            to_description({"result": {"engine": "hybrid", "data": {
                "ocr_text": "o", "vlm_description": "v", "merged": "m"}}}),
            {"ocr_text": "o", "vlm_description": "v"},
        )

    def test_batch_grouping_and_failure_isolation(self):
        client = PdfImgPipelineClient(self.settings)
        calls = []

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            files = [a.split("@", 1)[1] for a in cmd if a.startswith("files=@")]
            calls.append(files)
            return _FakeProc(0, _batch_stdout(files), "")

        original = subprocess.run
        subprocess.run = fake_run
        try:
            outcomes = client.extract_many(["a.png", "bad.png", "c.png"])
        finally:
            subprocess.run = original

        self.assertEqual(len(outcomes), 3)
        self.assertTrue(outcomes[0].ok)
        self.assertFalse(outcomes[1].ok)
        self.assertEqual(outcomes[1].error, "cannot read image")
        self.assertTrue(outcomes[2].ok)
        self.assertEqual(len(calls), 2)  # batch_size=2 -> 两组

    def test_describe_batch_normalizes(self):
        client = PdfImgPipelineClient(self.settings)

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            files = [a.split("@", 1)[1] for a in cmd if a.startswith("files=@")]
            return _FakeProc(0, _batch_stdout(files), "")

        original = subprocess.run
        subprocess.run = fake_run
        try:
            descs = client.describe_batch([{"path": "a.png", "caption": None},
                                           {"path": "bad.png", "caption": None}])
        finally:
            subprocess.run = original

        self.assertEqual(descs[0]["ocr_text"], "ocr-0")
        self.assertEqual(descs[0]["vlm_description"], "vlm-0")
        self.assertIsNone(descs[0]["error"])
        self.assertEqual(descs[1]["error"], "cannot read image")

    def test_transport_failure_marks_group(self):
        client = PdfImgPipelineClient(self.settings)

        def fake_run(cmd, capture_output=True, text=True, timeout=None):
            return _FakeProc(7, "", "connection refused")

        original = subprocess.run
        subprocess.run = fake_run
        try:
            outcomes = client.extract_many(["a.png", "b.png"])
        finally:
            subprocess.run = original

        self.assertTrue(all(not o.ok for o in outcomes))
        self.assertIn("connection refused", outcomes[0].error)


# ---------------------------------------------------------------- Pipeline
@dataclass
class _FakeParser:
    name: str
    available_flag: bool = True
    calls: int = 0

    def available(self):
        return self.available_flag

    def parse(self, pdf_path, out_dir, *, chunk_limit=600, image_describer=None):
        self.calls += 1
        return {
            "meta": {"file": os.path.basename(pdf_path)},
            "sections": [], "tables": [], "figures": [],
            "chunks": [], "elements": [],
            "engine": self.name,
        }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf = os.path.join(self.tmp.name, "a.pdf")
        make_text_pdf(self.pdf, n_pages=1)
        self.out = os.path.join(self.tmp.name, "out")
        self.settings = Settings.from_env(env_file=None)

    def tearDown(self):
        self.tmp.cleanup()

    def _pipeline(self, kind=PdfKind.TEXT, mineru_available=True):
        backend = MemoryCache()
        router = PdfRouter(_FakePrechecker(kind), RouteCache(backend, self.settings))
        parsers = {
            "pymupdf": _FakeParser("pymupdf"),
            "mineru": _FakeParser("mineru", available_flag=mineru_available),
        }
        return PdfPipeline(router, parsers, ResultCache(backend, self.settings), self.settings), parsers

    def test_routes_text_to_pymupdf_and_caches(self):
        pipeline, parsers = self._pipeline(PdfKind.TEXT)
        first = pipeline.process(self.pdf, self.out)
        second = pipeline.process(self.pdf, self.out)
        self.assertEqual(first["engine"], "pymupdf")
        self.assertFalse(first["from_cache"])
        self.assertTrue(second["from_cache"])
        self.assertEqual(parsers["pymupdf"].calls, 1)

    def test_routes_scanned_to_mineru(self):
        pipeline, parsers = self._pipeline(PdfKind.SCANNED)
        result = pipeline.process(self.pdf, self.out)
        self.assertEqual(result["engine"], "mineru")
        self.assertEqual(parsers["mineru"].calls, 1)

    def test_cache_key_varies_with_chunk_limit(self):
        pipeline, parsers = self._pipeline(PdfKind.TEXT)
        pipeline.process(self.pdf, self.out, chunk_limit=600)
        pipeline.process(self.pdf, self.out, chunk_limit=200)
        self.assertEqual(parsers["pymupdf"].calls, 2)

    def test_unavailable_mineru_raises_without_fallback(self):
        pipeline, _ = self._pipeline(PdfKind.SCANNED, mineru_available=False)
        with self.assertRaises(RuntimeError):
            pipeline.process(self.pdf, self.out)

    def test_unavailable_mineru_fallback(self):
        settings = self.settings.replace(mineru_fallback_text=True)
        backend = MemoryCache()
        router = PdfRouter(_FakePrechecker(PdfKind.SCANNED), RouteCache(backend, settings))
        parsers = {
            "pymupdf": _FakeParser("pymupdf"),
            "mineru": _FakeParser("mineru", available_flag=False),
        }
        pipeline = PdfPipeline(router, parsers, ResultCache(backend, settings), settings)
        result = pipeline.process(self.pdf, self.out)
        self.assertEqual(result["engine"], "pymupdf")
        self.assertIn("fallback", result["route"]["reason"])

    def test_no_cache_reruns(self):
        pipeline, parsers = self._pipeline(PdfKind.TEXT)
        pipeline.process(self.pdf, self.out, use_cache=False)
        pipeline.process(self.pdf, self.out, use_cache=False)
        self.assertEqual(parsers["pymupdf"].calls, 2)


# ---------------------------------------------------------------- 图语义化钩子
class DescribeHookTests(unittest.TestCase):
    def test_layout_parser_batch_hook(self):
        import layout_parser

        pending = [
            ({"image": b"a", "caption": "图1"}, "figure-01", "/tmp/a.png"),
            ({"image": b"b", "caption": None}, "figure-02", "/tmp/b.png"),
            ({"image": b"c", "caption": None}, "figure-03", "/tmp/c.png"),
        ]

        class Describer:
            def __init__(self):
                self.seen = None

            def describe_batch(self, items):
                self.seen = items
                return [{"ocr_text": "o1", "vlm_description": "v1", "error": None},
                        {"ocr_text": "o2", "vlm_description": "v2", "error": None},
                        {"ocr_text": "", "vlm_description": "", "error": "curl failed"}]

        d = Describer()
        layout_parser._describe_figures(pending, d)
        self.assertEqual([i["path"] for i in d.seen],
                         ["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"])
        self.assertEqual(pending[0][0]["description"]["ocr_text"], "o1")
        self.assertEqual(pending[1][0]["description"]["vlm_description"], "v2")
        # 失败项：description 置空，错误单独记录
        self.assertEqual(pending[2][0]["description"], {})
        self.assertEqual(pending[2][0]["description_error"], "curl failed")

    def test_layout_parser_falls_back_to_single(self):
        import layout_parser

        pending = [({"image": b"a", "caption": None}, "figure-01", "/tmp/a.png")]
        calls = []

        class Describer:
            def __call__(self, image_bytes, caption):
                calls.append((image_bytes, caption))
                return {"ocr_text": "x", "vlm_description": ""}

        layout_parser._describe_figures(pending, Describer())
        self.assertEqual(calls, [(b"a", None)])
        self.assertEqual(pending[0][0]["description"]["ocr_text"], "x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
