from pdfimgpipeline.features import PDFImageRouter
from pdfimgpipeline.models import ImageFeatures


def _features(**kw) -> ImageFeatures:
    base = dict(
        text_ratio=0.2, image_entropy=5.0, edge_density=0.05,
        color_variance=300.0, num_connected_components=100, aspect_ratio=1.0,
    )
    base.update(kw)
    return ImageFeatures(**base)


def test_low_text_high_entropy_scores_vlm():
    score, reasons = PDFImageRouter().rule_score(
        _features(text_ratio=0.05, image_entropy=7.0, edge_density=0.2,
                  color_variance=900.0))
    assert score >= 3
    assert {"low_text_ratio", "high_entropy", "rich_color"} <= set(reasons)


def test_high_text_low_entropy_scores_ocr():
    score, reasons = PDFImageRouter().rule_score(
        _features(text_ratio=0.6, image_entropy=4.0, color_variance=50.0))
    assert score <= -1
    assert {"high_text_ratio", "low_entropy", "gray_image"} <= set(reasons)


def test_mid_image_lands_in_hybrid_band():
    score, _ = PDFImageRouter().rule_score(_features())
    assert -1 < score < 3
