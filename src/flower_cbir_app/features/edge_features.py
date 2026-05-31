from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label, regionprops

from flower_cbir_app.features.base import FeatureResult
from flower_cbir_app.utils.common import normalize_vector


def _gradient(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # Dùng hướng gradient không dấu [0, 180) vì biên ảnh không có chiều cố định.
    ang = ((np.arctan2(gy, gx) + np.pi) % np.pi) * (180.0 / np.pi)
    return gx, gy, mag, ang


def extract_edge_orientation_hist(gray: np.ndarray, mask: np.ndarray, bins: int = 36) -> FeatureResult:
    # Edge orientation histogram: chỉ lấy hướng gradient tại các pixel được Canny
    # phát hiện là biên. Sobel histogram bên dưới lấy toàn bộ gradient foreground,
    # vì vậy hai feature không còn trùng logic.
    _, _, mag, ang = _gradient(gray)
    edge = cv2.Canny(gray, 100, 200)
    edge[mask == 0] = 0
    valid = (edge > 0) & (mask > 0)
    if not np.any(valid):
        valid = mask > 0
    hist, _ = np.histogram(ang[valid], bins=bins, range=(0, 180), weights=mag[valid])
    hist = normalize_vector(hist.astype(np.float32))
    return FeatureResult(hist.astype('float32'), {'images': {'Edge map': cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)}, 'plots': {'edge_orientation_hist': {'y': hist.tolist()}}, 'tables': {}}, {})


def extract_canny_derived(gray: np.ndarray, mask: np.ndarray) -> FeatureResult:
    edge = cv2.Canny(gray, 100, 200)
    edge[mask == 0] = 0
    lbl = label(edge > 0)
    props = regionprops(lbl)
    lengths = [p.area for p in props] if props else [0]
    edge_ratio = float(np.count_nonzero(edge) / (np.count_nonzero(mask) + 1e-12))
    component_count = float(len(props))
    mean_len = float(np.mean(lengths))
    max_len = float(np.max(lengths))
    std_len = float(np.std(lengths))
    center_density = 0.0
    if edge.size:
        h, w = edge.shape
        center = edge[h//4:3*h//4, w//4:3*w//4]
        center_density = float(np.count_nonzero(center) / (center.size + 1e-12))
    vector = np.array([edge_ratio, component_count, mean_len, max_len, std_len, center_density], dtype=np.float32)
    return FeatureResult(vector, {'images': {'Canny': cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)}, 'plots': {}, 'tables': {'canny_stats': {'edge_ratio': edge_ratio, 'component_count': component_count}}}, {})


def extract_sobel_hist(gray: np.ndarray, mask: np.ndarray, bins: int = 36) -> FeatureResult:
    _, _, mag, ang = _gradient(gray)
    valid = mask > 0
    hist, _ = np.histogram(ang[valid], bins=bins, range=(0, 180), weights=mag[valid])
    hist = normalize_vector(hist.astype(np.float32))
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'sobel_hist': {'y': hist.tolist()}}, 'tables': {}}, {})
