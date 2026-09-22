from dataclasses import asdict

from fastapi.testclient import TestClient

from pdfimgpipeline.config import Settings
from pdfimgpipeline.models import BatchOutcome, RouteResult
from pdfimgpipeline.service.app import create_app


class _InnerRouter:
    async def decide(self, path):
        return RouteResult(decision="ocr", reason="test|stub", features={"x": 1})

    async def decide_many(self, paths, *, max_concurrency=8):
        return [BatchOutcome(ok=True, data=asdict(await self.decide(p))) for p in paths]


class FakePipeline:
    def __init__(self):
        self.router = _InnerRouter()

    async def process(self, path):
        return {
            "image_hash": "deadbeef",
            "decision": "ocr",
            "reason": "test|stub",
            "features": {"x": 1},
            "ocr_probe": {"chars": 0, "conf": 0.0},
            "result": {"engine": "ocr", "data": {"text": "hi"}},
            "from_cache": False,
        }

    async def process_many(self, paths, *, max_concurrency=8):
        return [BatchOutcome(ok=True, data=await self.process(p)) for p in paths]


def _make_app(**settings_kw):
    settings = Settings(cache_backend="none", pdfimg_api_key="", **settings_kw)
    return create_app(settings, pipeline=FakePipeline())


def test_health_no_auth():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["cache_backend"] == "injected"


def test_route_endpoint():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post("/route", files={"file": ("a.png", b"fake", "image/png")})
        assert resp.status_code == 200
        assert resp.json()["decision"] == "ocr"


def test_extract_endpoint():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post("/extract", files={"file": ("a.png", b"fake", "image/png")})
        assert resp.status_code == 200
        body = resp.json()
        assert body["result"]["data"]["text"] == "hi"
        assert body["decision"] == "ocr"


def test_metrics_endpoint():
    app = _make_app()
    with TestClient(app) as client:
        client.post("/extract", files={"file": ("a.png", b"fake", "image/png")})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "pdfimg_extract_ok_total" in resp.text


def test_api_key_enforced():
    settings = Settings(cache_backend="none", pdfimg_api_key="secret")
    app = create_app(settings, pipeline=FakePipeline())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200          # 探活不鉴权
        assert client.post(
            "/extract", files={"file": ("a.png", b"fake", "image/png")}
        ).status_code == 401
        assert client.post(
            "/extract",
            files={"file": ("a.png", b"fake", "image/png")},
            headers={"X-API-Key": "secret"},
        ).status_code == 200


def test_upload_size_limit():
    app = _make_app(max_upload_mb=0)
    with TestClient(app) as client:
        resp = client.post("/extract", files={"file": ("a.png", b"x" * 1024, "image/png")})
        assert resp.status_code == 413


def _two_files():
    return [
        ("files", ("a.png", b"aaa", "image/png")),
        ("files", ("b.png", b"bbb", "image/png")),
    ]


def test_extract_batch():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post("/extract:batch", files=_two_files())
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["ok_count"] == 2
        assert body["failed_count"] == 0
        assert body["results"][0]["filename"] == "a.png"
        assert body["results"][0]["data"]["result"]["data"]["text"] == "hi"


def test_route_batch():
    app = _make_app()
    with TestClient(app) as client:
        resp = client.post("/route:batch", files=_two_files())
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["results"][1]["data"]["decision"] == "ocr"


def test_batch_size_limit():
    app = _make_app(max_batch_size=1)
    with TestClient(app) as client:
        resp = client.post("/extract:batch", files=_two_files())
        assert resp.status_code == 413


def test_batch_metrics():
    app = _make_app()
    with TestClient(app) as client:
        client.post("/extract:batch", files=_two_files())
        body = client.get("/metrics").text
        assert "pdfimg_extract_batch_ok_total" in body
