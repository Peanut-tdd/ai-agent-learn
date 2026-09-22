"""图像特征提取 + 规则打分（纯 OpenCV，不依赖模型）。"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .models import ImageFeatures


class PDFImageRouter:
    """提取 6 维轻量特征，并按阈值给出路由分数与命中原因。"""

    def __init__(
        self,
        text_ratio_threshold: float = 0.15,
        entropy_threshold: float = 6.5,
        edge_density_threshold: float = 0.12,
        color_var_threshold: float = 800.0,
    ):
        self.text_ratio_threshold = text_ratio_threshold
        self.entropy_threshold = entropy_threshold
        self.edge_density_threshold = edge_density_threshold
        self.color_var_threshold = color_var_threshold

    # ---------------------------------------------------------------- 特征
    def extract_features(self, image_path: str) -> ImageFeatures:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        edges = cv2.Canny(gray, 50, 150)
        return ImageFeatures(
            text_ratio=self._estimate_text_ratio(gray),
            image_entropy=self._calc_entropy(gray),
            edge_density=float(np.count_nonzero(edges)) / (h * w),
            color_variance=float(np.var(img.astype(np.float32))),
            num_connected_components=self._count_components(gray),
            aspect_ratio=w / h,
        )

    def _estimate_text_ratio(self, gray) -> float:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        h, w = gray.shape
        text_pixels = 0
        for c in contours:
            _, _, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < (h * w) * 0.5 and 1.5 < (cw / max(ch, 1)) < 50:
                text_pixels += area
        return text_pixels / (h * w)

    def _calc_entropy(self, gray) -> float:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    def _count_components(self, gray) -> int:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 8,
        )
        num, _ = cv2.connectedComponents(binary)
        return int(num)

    # ---------------------------------------------------------------- 打分
    def rule_score(self, f: ImageFeatures) -> Tuple[int, List[str]]:
        """分数越高越像「需要 VLM 的复杂图」。"""
        score = 0
        reasons: List[str] = []

        if f.text_ratio < self.text_ratio_threshold:
            score += 2
            reasons.append("low_text_ratio")
        elif f.text_ratio > 0.4:
            score -= 2
            reasons.append("high_text_ratio")

        if f.image_entropy > self.entropy_threshold:
            score += 2
            reasons.append("high_entropy")
        elif f.image_entropy < 4.5:
            score -= 1
            reasons.append("low_entropy")

        if f.edge_density > self.edge_density_threshold:
            score += 1
            reasons.append("high_edge_density")

        if f.color_variance > self.color_var_threshold:
            score += 2
            reasons.append("rich_color")
        elif f.color_variance < 100:
            score -= 1
            reasons.append("gray_image")

        if f.num_connected_components > 500:
            score -= 1
            reasons.append("many_components")

        return score, reasons
