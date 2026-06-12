from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import label, regionprops

from flower_cbir_app.features.base import FeatureResult
from flower_cbir_app.utils.common import normalize_vector


def _gradient(gray: np.ndarray):
    """Tính gradient ảnh xám bằng Sobel: trả về (gx, gy, magnitude, angle).

    Hướng gradient dùng dạng không dấu [0,180) vì biên ảnh không có chiều cố
    định (cạnh dốc lên hay xuống đều là cùng một hướng cạnh).
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # Dùng hướng gradient không dấu [0, 180) vì biên ảnh không có chiều cố định.
    ang = ((np.arctan2(gy, gx) + np.pi) % np.pi) * (180.0 / np.pi)
    return gx, gy, mag, ang


def extract_edge_orientation_hist(gray: np.ndarray, mask: np.ndarray, bins: int = 36) -> FeatureResult:
    """Histogram hướng cạnh, chỉ tại các pixel Canny xác định là BIÊN (36 chiều).

    Mỗi bin cộng dồn độ lớn gradient theo hướng, rồi L1-normalize. Khác với
    sobel_hist (lấy toàn bộ gradient foreground), feature này chỉ xét pixel biên
    nên mô tả hướng của các đường nét rõ.
    """
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
    """6 thống kê suy ra từ bản đồ cạnh Canny (6 chiều).

    Vector = [tỉ lệ pixel cạnh, số thành phần cạnh, độ dài cạnh trung bình, dài
    nhất, độ lệch chuẩn độ dài, mật độ cạnh vùng trung tâm]. Mô tả "lượng" và
    phân bố chi tiết đường nét của vật.
    """
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
    """Histogram hướng gradient trên TOÀN BỘ vùng vật (36 chiều).

    Cộng dồn độ lớn gradient Sobel theo hướng tại mọi pixel foreground (không chỉ
    pixel biên), L1-normalize. Phản ánh xu hướng kết cấu/đường nét tổng thể.
    """
    _, _, mag, ang = _gradient(gray)
    valid = mask > 0
    hist, _ = np.histogram(ang[valid], bins=bins, range=(0, 180), weights=mag[valid])
    hist = normalize_vector(hist.astype(np.float32))
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'sobel_hist': {'y': hist.tolist()}}, 'tables': {}}, {})
