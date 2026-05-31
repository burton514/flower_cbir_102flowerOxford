from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import skew

from flower_cbir_app.features.base import FeatureResult
from flower_cbir_app.utils.common import normalize_vector


def _masked_pixels(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = image_rgb[mask > 0]
    if len(pixels) == 0:
        return image_rgb.reshape(-1, 3)
    return pixels


def extract_hsv_hist(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], (mask > 0).astype('uint8'), [16, 6, 3], [0, 180, 0, 256, 0, 256]).flatten()
    hist = normalize_vector(hist)
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'HSV histogram': {'y': hist.tolist()}}, 'tables': {}}, {})


def extract_rgb_hist(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    hist = cv2.calcHist([image_rgb], [0, 1, 2], (mask > 0).astype('uint8'), [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    hist = normalize_vector(hist)
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'RGB histogram': {'y': hist.tolist()}}, 'tables': {}}, {})


def extract_hue_hist(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0], (mask > 0).astype('uint8'), [36], [0, 180]).flatten()
    hist = normalize_vector(hist)
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'Hue histogram': {'y': hist.tolist()}}, 'tables': {}}, {})


def extract_dominant_colors(image_rgb: np.ndarray, mask: np.ndarray, clusters: int = 5) -> FeatureResult:
    # Feature này là mô tả palette màu bằng KMeans. Vì so sánh palette bằng
    # vector nối trực tiếp chỉ là heuristic, feature được để mặc định tắt.
    # Đặt seed OpenCV để kết quả tái lập khi chạy lại cùng dữ liệu.
    cv2.setRNGSeed(42)
    pixels = _masked_pixels(image_rgb, mask).astype(np.float32)
    if len(pixels) < clusters:
        if len(pixels) == 0:
            pixels = np.zeros((clusters, 3), dtype=np.float32)
        else:
            pixels = np.vstack([pixels, np.repeat(pixels[:1], clusters - len(pixels), axis=0)])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    ratios = np.array([(labels == i).mean() for i in range(clusters)], dtype=np.float32)
    order = np.argsort(-ratios)
    centers = centers[order] / 255.0
    ratios = ratios[order]
    vector = np.hstack([np.hstack([centers[i], ratios[i]]) for i in range(clusters)]).astype('float32')
    table = {}
    for i in range(clusters):
        table[f'color_{i+1}'] = centers[i].tolist()
        table[f'ratio_{i+1}'] = float(ratios[i])
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'dominant_colors': table}}, {})


def extract_color_moments(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    # Color moments chuẩn: mean, standard deviation, skewness theo từng kênh màu.
    # Dùng RGB để tránh vấn đề Hue là đại lượng vòng tròn trong HSV.
    rgb = image_rgb.astype(np.float32)
    pixels = rgb[mask > 0]
    if len(pixels) == 0:
        pixels = rgb.reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    skews = np.array([float(skew(pixels[:, i])) if len(np.unique(pixels[:, i])) > 1 else 0.0 for i in range(3)], dtype=np.float32)
    vector = np.hstack([means, stds, skews]).astype('float32')
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'color_moments': {'mean_r': float(means[0]), 'mean_g': float(means[1]), 'mean_b': float(means[2])}}}, {})


def extract_lab_moments(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[mask > 0]
    if len(pixels) == 0:
        pixels = lab.reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    vector = np.hstack([means, stds]).astype('float32')
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'lab_moments': {'mean_l': float(means[0]), 'std_l': float(stds[0])}}}, {})
