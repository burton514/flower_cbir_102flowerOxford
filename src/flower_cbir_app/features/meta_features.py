from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops

from flower_cbir_app.features.base import FeatureResult


def extract_foreground_occupancy(mask: np.ndarray) -> FeatureResult:
    """Tỉ lệ diện tích vật chiếm trong khung ảnh (1 chiều).

    Feature META — dùng để kiểm tra chất lượng tiền xử lý, không dùng để xếp
    hạng truy hồi (bị loại bởi exclude_meta_from_retrieval).
    """
    occ = float(np.count_nonzero(mask) / mask.size)
    return FeatureResult(np.array([occ], dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {'occupancy': {'value': occ}}}, {})


def extract_centroid_offset(mask: np.ndarray) -> FeatureResult:
    """Độ lệch trọng tâm vật so với tâm ảnh, chuẩn hóa [0,1] (1 chiều).

    Feature META kiểm tra việc căn giữa khi tiền xử lý — gần 0 nghĩa là vật đã
    được đặt giữa khung tốt. Không dùng để xếp hạng truy hồi.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return FeatureResult(np.array([0.0], dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {}}, {})
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    dist = float(np.sqrt((cx - mask.shape[1] / 2) ** 2 + (cy - mask.shape[0] / 2) ** 2))
    max_dist = float(np.sqrt(mask.shape[0] ** 2 + mask.shape[1] ** 2) / 2.0 + 1e-12)
    offset = dist / max_dist
    return FeatureResult(np.array([offset], dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {'centroid_offset': {'value': offset}}}, {})


def extract_mask_quality(mask: np.ndarray) -> FeatureResult:
    """3 chỉ số chất lượng mask: [số thành phần, chạm biên?, tỉ lệ lỗ] (3 chiều).

    Feature META đánh giá mask tách nền có sạch không: nhiều thành phần hoặc
    chạm biên ảnh hoặc nhiều lỗ -> mask kém. hole_ratio = lỗ bên trong / diện
    tích đã lấp lỗ (không phải 1-occupancy). Không dùng để xếp hạng truy hồi.
    """
    mask_bool = mask > 0
    lbl = label(mask_bool)
    props = regionprops(lbl)
    components = len(props)
    touch_border = int(mask_bool[0].any() or mask_bool[-1].any() or mask_bool[:, 0].any() or mask_bool[:, -1].any())

    # Hole ratio đúng nghĩa: phần lỗ nằm bên trong foreground sau khi fill holes,
    # chia cho diện tích foreground đã fill. Không dùng 1 - occupancy vì đó là tỉ lệ nền.
    if np.any(mask_bool):
        filled = ndimage.binary_fill_holes(mask_bool)
        holes = filled & (~mask_bool)
        hole_ratio = float(np.count_nonzero(holes) / (np.count_nonzero(filled) + 1e-12))
    else:
        hole_ratio = 0.0

    return FeatureResult(np.array([components, touch_border, hole_ratio], dtype=np.float32), {'images': {}, 'plots': {}, 'tables': {'mask_quality': {'components': components, 'touch_border': touch_border, 'hole_ratio': hole_ratio}}}, {})
