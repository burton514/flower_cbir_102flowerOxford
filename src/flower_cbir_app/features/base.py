from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureResult:
    """Kết quả trả về của MỌI hàm extract_* — hộp chứa dữ liệu (không phải ORM).

    - vector: vector đặc trưng (np.ndarray) dùng để so khớp.
    - debug_bundle: ảnh/biểu đồ/bảng để hiển thị debug trên UI (không vào DB).
    - extra: thông tin phụ tùy feature.
    Nhờ khuôn chung này, pipeline xử lý mọi feature theo cùng một cách.
    """
    vector: np.ndarray
    debug_bundle: dict
    extra: dict


@dataclass
class FeatureSpec:
    """Bản khai metadata TĨNH của một feature (tên, nhóm, distance, cờ...).

    Không gắn với DB. registry.get_feature_catalog() trả về danh sách các spec
    này; UI đọc để vẽ checkbox, pipeline đọc để chọn cách chuẩn hóa (is_histogram
    -> L1, ngược lại -> z-score) và distance hợp lệ.
    """
    key: str
    name: str
    group: str
    description: str
    default_distance: str
    enabled_by_default: bool
    output_dim_display: str
    is_meta: bool = False
    is_histogram: bool = False  # True → dùng L1-normalize thay vì z-score khi lưu vector
    supports_chi_square: bool = False  # True khi vector là histogram không âm phù hợp χ²
    default_weight: float = 1.0  # Trọng số mặc định khi auto_weight=False
