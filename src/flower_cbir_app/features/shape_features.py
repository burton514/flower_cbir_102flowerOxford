from __future__ import annotations

import cv2
import numpy as np

from flower_cbir_app.features.base import FeatureResult


def _main_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([[[0, 0]]], dtype=np.int32)
    return max(contours, key=cv2.contourArea)


def extract_hu_moments(mask: np.ndarray) -> FeatureResult:
    # Hu Moments là shape descriptor, vì vậy phải tính trên mask nhị phân
    # với binaryImage=True thay vì coi mask 0/255 như ảnh cường độ xám.
    mask_bin = (mask > 0).astype(np.uint8)
    moments = cv2.moments(mask_bin, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return FeatureResult(hu.astype('float32'), {'images': {}, 'plots': {'Hu moments': {'y': hu.tolist()}}, 'tables': {}}, {})


def extract_geometric_shape(mask: np.ndarray) -> FeatureResult:
    contour = _main_contour(mask)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    img_h, img_w = mask.shape[:2]
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    # Các đại lượng hình học cơ bản. Những đại lượng theo pixel được chuẩn hóa
    # theo kích thước canvas để giảm phụ thuộc scale tuyệt đối.
    area_ratio = float(area / (img_h * img_w + 1e-12))
    perimeter_norm = float(perimeter / (2.0 * (img_h + img_w) + 1e-12))
    aspect_ratio = float(w / h) if h > 0 else 0.0
    circularity = float((4.0 * np.pi * area) / (perimeter ** 2 + 1e-12))
    solidity = float(area / (hull_area + 1e-12))
    extent = float(area / (w * h + 1e-12))
    equivalent_diameter_norm = float(np.sqrt(4 * area / np.pi) / max(img_h, img_w)) if area > 0 else 0.0
    hull_area_ratio = float(hull_area / (img_h * img_w + 1e-12))
    hull_perimeter_norm = float(hull_perimeter / (2.0 * (img_h + img_w) + 1e-12))
    compactness = float((perimeter ** 2) / (4 * np.pi * area + 1e-12))
    convexity = float(hull_perimeter / (perimeter + 1e-12)) if perimeter > 0 else 0.0

    eccentricity = 0.0
    roundness = circularity
    if len(contour) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(contour)
        major = max(MA, ma)
        minor = min(MA, ma)
        a = major / 2
        b = minor / 2
        if a > 0:
            eccentricity = float(np.sqrt(max(0.0, 1 - (b * b) / (a * a + 1e-12))))
            roundness = float((4.0 * area) / (np.pi * major * major + 1e-12))

    vector = np.array([
        area_ratio, perimeter_norm, aspect_ratio, circularity, solidity, extent,
        eccentricity, equivalent_diameter_norm, hull_area_ratio, hull_perimeter_norm,
        convexity, compactness, roundness,
    ], dtype=np.float32)
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'shape_stats': {'area_ratio': area_ratio, 'perimeter_norm': perimeter_norm, 'circularity': circularity}}}, {})


def extract_contour_basic(mask: np.ndarray) -> FeatureResult:
    contour = _main_contour(mask)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    hull = cv2.convexHull(contour)
    roughness = float(cv2.arcLength(contour, True) / (cv2.arcLength(hull, True) + 1e-12))
    defects_count = max(0, len(contour) - len(approx))
    area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(hull))
    vector = np.array([
        len(contour), len(approx), roughness, defects_count,
        float(area / (hull_area + 1e-12)),
    ], dtype=np.float32)
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'contour_basic': {'points': int(len(contour)), 'approx_points': int(len(approx))}}}, {})


def extract_radial_signature(mask: np.ndarray, bins: int = 36) -> FeatureResult:
    contour = _main_contour(mask)
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) == 0:
        return FeatureResult(np.zeros(bins, dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {}}, {})
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if abs(m['m00']) < 1e-12:
        cx, cy = mask.shape[1] / 2, mask.shape[0] / 2
    else:
        cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    vectors = pts - np.array([[cx, cy]], dtype=np.float32)
    angles = (np.arctan2(vectors[:, 1], vectors[:, 0]) + 2 * np.pi) % (2 * np.pi)
    radii = np.sqrt(np.sum(vectors ** 2, axis=1))
    signature = np.zeros(bins, dtype=np.float32)
    for b in range(bins):
        lo = 2 * np.pi * b / bins
        hi = 2 * np.pi * (b + 1) / bins
        sel = (angles >= lo) & (angles < hi)
        signature[b] = radii[sel].max() if np.any(sel) else 0.0
    if signature.max() > 0:
        signature = signature / signature.max()
    return FeatureResult(signature.astype('float32'), {'images': {}, 'plots': {'radial_signature': {'y': signature.tolist()}}, 'tables': {}}, {})


def _resample_closed_contour(points: np.ndarray, n_points: int = 128) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    pts_closed = np.vstack([pts, pts[:1]])
    seg = np.sqrt(np.sum(np.diff(pts_closed, axis=0) ** 2, axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total < 1e-12:
        return np.repeat(pts[:1], n_points, axis=0).astype(np.float32)
    targets = np.linspace(0, total, n_points, endpoint=False)
    x = np.interp(targets, cumulative, pts_closed[:, 0])
    y = np.interp(targets, cumulative, pts_closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def extract_fourier_shape(mask: np.ndarray, coeffs: int = 32) -> FeatureResult:
    contour = _main_contour(mask).reshape(-1, 2).astype(np.float32)
    if len(contour) < 4:
        return FeatureResult(np.zeros(coeffs, dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {}}, {})
    contour = _resample_closed_contour(contour, n_points=max(128, coeffs * 4))
    contour = contour - contour.mean(axis=0, keepdims=True)
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    spectrum = np.abs(np.fft.fft(complex_contour))[1: coeffs + 1]
    if len(spectrum) < coeffs:
        spectrum = np.pad(spectrum, (0, coeffs - len(spectrum)))
    norm = float(np.linalg.norm(spectrum) + 1e-12)
    spectrum = spectrum / norm
    return FeatureResult(spectrum.astype('float32'), {'images': {}, 'plots': {'fourier_shape': {'y': spectrum.tolist()}}, 'tables': {}}, {})


def _foreground_overlap_score(a: np.ndarray, b: np.ndarray) -> float:
    # Jaccard overlap trên foreground, tránh việc nền trắng lớn làm score giả cao.
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a, b).sum()
    return float(inter / (union + 1e-12))


def extract_symmetry_score(mask: np.ndarray) -> FeatureResult:
    mask_bin = mask > 0
    lr = _foreground_overlap_score(mask_bin, np.fliplr(mask_bin))
    ud = _foreground_overlap_score(mask_bin, np.flipud(mask_bin))
    vector = np.array([lr, ud], dtype=np.float32)
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'symmetry': {'left_right': lr, 'up_down': ud}}}, {})


# ── Feature mới: Rotational Symmetry Order ──────────────────────────────────
def extract_rotational_symmetry(mask: np.ndarray, max_order: int = 8) -> FeatureResult:
    """Bậc đối xứng xoay — ước lượng số cánh hoa.

    Ý nghĩa từng chiều vector (max_order chiều):
      - vector[k] = overlap score khi xoay mask đi (k+1) × (360 / max_order) độ.
      - Overlap = Jaccard(mask, rotated_mask) ∈ [0, 1].
      - Nếu hoa có n cánh, xoay 360°/n sẽ cho overlap cao → vector[n-1] lớn.

    Ví dụ:
      - Hoa 5 cánh: vector[4] (xoay 72°) sẽ cao nhất.
      - Hoa 6 cánh: vector[5] (xoay 60°) sẽ cao nhất.
      - Bậc đối xứng ước lượng = argmax(vector) + 1.

    Tại sao hữu ích:
      - Số cánh là đặc trưng phân loại hoa rất tự nhiên, giải thích được trực tiếp.
      - Khác với symmetry_score (chỉ đo lật ngang/dọc), feature này đo đối xứng xoay.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    h, w = mask_bin.shape
    cy, cx = h / 2.0, w / 2.0

    scores = []
    for k in range(1, max_order + 1):
        angle = k * 360.0 / max_order
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(mask_bin, M, (w, h), flags=cv2.INTER_NEAREST)
        a = mask_bin.astype(bool)
        b = rotated.astype(bool)
        union = np.logical_or(a, b).sum()
        inter = np.logical_and(a, b).sum()
        scores.append(float(inter / (union + 1e-12)))

    vector = np.array(scores, dtype=np.float32)
    best_order = int(np.argmax(vector)) + 1
    return FeatureResult(
        vector,
        {'images': {}, 'plots': {'Rotational overlap': {'y': vector.tolist()}},
         'tables': {'rotational_symmetry': {
             'estimated_petal_count': best_order,
             'best_overlap': float(vector[best_order - 1]),
         }}},
        {}
    )
