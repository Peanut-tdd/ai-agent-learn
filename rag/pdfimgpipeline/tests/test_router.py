from pdfimgpipeline.cache import MemoryCache, RouteCache
from pdfimgpipeline.features import PDFImageRouter
from pdfimgpipeline.models import ImageFeatures
from pdfimgpipeline.router import HybridRouter


class FakeProbe:
    def __init__(self, chars, conf):
        self._chars = chars
        self._conf = conf

    async def probe(self, path):
        return self._chars, self._conf


def _router_with(feats: ImageFeatures) -> PDFImageRouter:
    router = PDFImageRouter()
    router.extract_features = lambda path: feats  # type: ignore[assignment]
    return router


def _feats(**kw) -> ImageFeatures:
    base = dict(
        text_ratio=0.2, image_entropy=5.0, edge_density=0.05,
        color_variance=300.0, num_connected_components=100, aspect_ratio=1.0,
    )
    base.update(kw)
    return ImageFeatures(**base)


def _img(tmp_path, name="a.png") -> str:
    p = tmp_path / name
    p.write_bytes(b"fake-image-bytes")
    return str(p)


async def _decide(tmp_path, feats, probe):
    cache = RouteCache(MemoryCache(), ttl=60)
    router = HybridRouter(_router_with(feats), cache, probe)
    return await router.decide(_img(tmp_path))


async def test_high_score_routes_vlm(tmp_path):
    result = await _decide(
        tmp_path,
        _feats(text_ratio=0.05, image_entropy=7.0, color_variance=900.0),
        FakeProbe(0, 0.0),
    )
    assert result.decision == "vlm"


async def test_low_score_routes_ocr(tmp_path):
    result = await _decide(
        tmp_path,
        _feats(text_ratio=0.6, image_entropy=4.0, color_variance=50.0),
        FakeProbe(0, 0.0),
    )
    assert result.decision == "ocr"


async def test_mid_score_no_text_routes_vlm(tmp_path):
    result = await _decide(tmp_path, _feats(), FakeProbe(0, 0.0))
    assert result.decision == "vlm"
    assert "probe_no_text->vlm" in result.reason


async def test_mid_score_high_quality_text_routes_ocr(tmp_path):
    result = await _decide(tmp_path, _feats(), FakeProbe(80, 0.95))
    assert result.decision == "ocr"


async def test_mid_score_medium_text_routes_hybrid(tmp_path):
    result = await _decide(tmp_path, _feats(), FakeProbe(30, 0.70))
    assert result.decision == "hybrid"


async def test_decision_is_cached(tmp_path):
    cache = RouteCache(MemoryCache(), ttl=60)
    router = HybridRouter(_router_with(_feats()), cache, FakeProbe(30, 0.70))
    path = _img(tmp_path)
    first = await router.decide(path)
    second = await router.decide(path)
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.decision == first.decision
