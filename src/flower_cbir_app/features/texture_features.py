from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from flower_cbir_app.features.base import FeatureResult
from flower_cbir_app.utils.common import normalize_vector


def extract_lbp(gray: np.ndarray, mask: np.ndarray, radius: int = 3, points: int = 24) -> FeatureResult:
    gray_norm = gray.copy()
    gray_norm[mask == 0] = 255
    lbp = local_binary_pattern(gray_norm, P=points, R=radius, method='uniform')
    values = lbp[mask > 0]
    if len(values) == 0:
        values = lbp.reshape(-1)
    bins = points + 2
    hist, _ = np.histogram(values, bins=np.arange(0, bins + 1), range=(0, bins))
    hist = normalize_vector(hist.astype(np.float32))
    lbp_vis = (255 * (lbp / (lbp.max() + 1e-12))).astype(np.uint8)
    return FeatureResult(hist.astype('float32'), {'images': {'LBP map': cv2.cvtColor(lbp_vis, cv2.COLOR_GRAY2RGB)}, 'plots': {'LBP histogram': {'y': hist.tolist()}}, 'tables': {}}, {})


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _masked_graycomatrix(reduced: np.ndarray, valid_mask: np.ndarray, distances, angles, levels: int = 8) -> np.ndarray:
    h, w = reduced.shape
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    for di, d in enumerate(distances):
        for ai, angle in enumerate(angles):
            dx = int(round(np.cos(angle) * d))
            dy = -int(round(np.sin(angle) * d))
            x0_src, x1_src = max(0, -dx), min(w, w - dx)
            y0_src, y1_src = max(0, -dy), min(h, h - dy)
            x0_dst, x1_dst = x0_src + dx, x1_src + dx
            y0_dst, y1_dst = y0_src + dy, y1_src + dy
            src = reduced[y0_src:y1_src, x0_src:x1_src]
            dst = reduced[y0_dst:y1_dst, x0_dst:x1_dst]
            valid = valid_mask[y0_src:y1_src, x0_src:x1_src] & valid_mask[y0_dst:y1_dst, x0_dst:x1_dst]
            if not np.any(valid):
                continue
            src_vals = src[valid].ravel()
            dst_vals = dst[valid].ravel()
            counts = np.bincount(src_vals * levels + dst_vals, minlength=levels * levels).reshape(levels, levels)
            counts = counts + counts.T  # symmetric=True
            total = counts.sum()
            if total > 0:
                glcm[:, :, di, ai] = counts / total
    return glcm


def extract_glcm(gray: np.ndarray, mask: np.ndarray) -> FeatureResult:
    # Tính GLCM chỉ trên ROI foreground; không để nền trắng chi phối thống kê texture.
    x0, y0, x1, y1 = _bbox_from_mask(mask)
    gray_crop = gray[y0:y1, x0:x1]
    mask_crop = (mask[y0:y1, x0:x1] > 0)
    if mask_crop.size == 0 or not np.any(mask_crop):
        return FeatureResult(np.zeros(6, dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {}}, {})

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_crop.astype(np.uint8), kernel, iterations=1).astype(bool)
    valid_mask = eroded if np.any(eroded) else mask_crop

    reduced = np.clip(gray_crop // 32, 0, 7).astype(np.uint8)
    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = _masked_graycomatrix(reduced, valid_mask, distances=distances, angles=angles, levels=8)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for prop in properties:
        values = graycoprops(glcm, prop)
        feats.append(float(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).mean()))
    return FeatureResult(np.asarray(feats, dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {'glcm': {p: float(v) for p, v in zip(properties, feats)}}}, {})


def extract_hog(gray: np.ndarray, mask: np.ndarray) -> FeatureResult:
    gray_roi = gray.copy()
    gray_roi[mask == 0] = 255
    feat, hog_image = hog(
        gray_roi,
        orientations=9,
        pixels_per_cell=(64, 64),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True,
    )
    hog_image = (255 * (hog_image - hog_image.min()) / (np.ptp(hog_image) + 1e-12)).astype(np.uint8)
    return FeatureResult(feat.astype('float32'), {'images': {'HOG image': cv2.cvtColor(hog_image, cv2.COLOR_GRAY2RGB)}, 'plots': {}, 'tables': {'hog': {'dimension': int(feat.size)}}}, {})
