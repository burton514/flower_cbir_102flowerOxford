from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureResult:
    vector: np.ndarray
    debug_bundle: dict
    extra: dict


@dataclass
class FeatureSpec:
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
