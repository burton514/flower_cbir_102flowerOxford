from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def zscore_apply(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (vector - mean) / (std + 1e-12)


def is_histogram_feature(config: dict) -> bool:
    """Trả về True nếu feature được lưu bằng chuẩn hóa L1 dạng histogram."""
    extra = config.get('extra') or {}
    return bool(extra.get('is_histogram', False))


def supports_chi_square(config: dict) -> bool:
    """Chi-square chỉ dùng khi vector là histogram không âm phù hợp χ².

    Một số descriptor như HOG có bản chất histogram nhưng đã qua block normalization
    L2-Hys; project không cho chọn χ² cho những descriptor này để tránh diễn giải
    sai lý thuyết.
    """
    extra = config.get('extra') or {}
    if 'supports_chi_square' in extra:
        return bool(extra.get('supports_chi_square'))
    return is_histogram_feature(config)


def resolve_distance_type(config: dict) -> str:
    """Không cho dùng chi-square với vector không phù hợp."""
    metric = str(config.get('distance_type', 'l2'))
    if metric == 'chi_square' and not supports_chi_square(config):
        return 'l2'
    return metric


def compute_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if metric == 'l2':
        return float(np.linalg.norm(a - b))
    if metric == 'cosine':
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 1.0
        value = 1.0 - np.dot(a, b) / (na * nb + 1e-12)
        return float(np.clip(value, 0.0, 2.0))
    if metric == 'chi_square':
        # χ²(x,y) = ½ Σ (xᵢ - yᵢ)² / (xᵢ + yᵢ + ε)
        # Chỉ dùng cho histogram không âm đã L1-normalize.
        a = np.maximum(a, 0.0)
        b = np.maximum(b, 0.0)
        return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-10)))
    raise ValueError(metric)


def normalize_distance_values(distances: Iterable[float]) -> np.ndarray:
    """Min-max normalize distance của một feature trên cùng tập ứng viên.

    Việc này giúp fusion không bị feature có thang đo lớn áp đảo feature khác.
    Nếu mọi khoảng cách bằng nhau, feature đó không đóng góp phân biệt nên trả 0.
    """
    arr = np.asarray(list(distances), dtype=np.float32)
    if arr.size == 0:
        return arr
    d_min = float(np.min(arr))
    d_max = float(np.max(arr))
    if d_max - d_min < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - d_min) / (d_max - d_min + 1e-12)


def aggregate_weighted_distances_with_details(scores_by_feature: dict, weights: dict) -> tuple[dict, dict]:
    """Chuẩn hóa distance theo từng feature rồi cộng trọng số, kèm chi tiết.

    scores_by_feature có dạng:
        {feature_key: [(image_id, raw_distance), ...]}

    Trả về:
        aggregated: {image_id: final_distance_score}
        details: {image_id: [per-feature contribution rows]}

    final_distance_score càng nhỏ thì ảnh càng giống query.
    """
    aggregated = defaultdict(float)
    details = defaultdict(list)
    for key, rows in scores_by_feature.items():
        if not rows or key not in weights:
            continue
        image_ids = [image_id for image_id, _ in rows]
        raw_distances = [float(dist) for _, dist in rows]
        normalized = normalize_distance_values(raw_distances)
        weight = float(weights.get(key, 0.0))
        for image_id, raw_dist, norm_dist in zip(image_ids, raw_distances, normalized):
            contribution = weight * float(norm_dist)
            aggregated[image_id] += contribution
            details[image_id].append({
                'feature_key': key,
                'raw_distance': float(raw_dist),
                'normalized_distance': float(norm_dist),
                'weight': weight,
                'contribution': float(contribution),
            })
    return dict(aggregated), dict(details)


def aggregate_weighted_distances(scores_by_feature: dict, weights: dict) -> dict:
    """Chuẩn hóa distance theo từng feature rồi cộng trọng số."""
    aggregated, _ = aggregate_weighted_distances_with_details(scores_by_feature, weights)
    return aggregated


def build_effective_weights(feature_configs: dict, auto_weight: bool = True, exclude_meta_from_retrieval: bool = True) -> dict:
    active = {k: v for k, v in feature_configs.items() if v['enabled'] and not (exclude_meta_from_retrieval and v['is_meta'])}
    if not active:
        return {}
    if not auto_weight:
        total = sum(max(0.0, float(v['weight'])) for v in active.values())
        if total <= 0:
            total = len(active)
            return {k: 1.0 / total for k in active.keys()}
        return {k: max(0.0, float(v['weight'])) / total for k, v in active.items()}
    groups = {}
    for key, cfg in active.items():
        groups.setdefault(cfg['group_name'], []).append(key)
    group_weight = 1.0 / len(groups)
    out = {}
    for group_keys in groups.values():
        each = group_weight / len(group_keys)
        for key in group_keys:
            out[key] = each
    return out


def build_fisher_weights(feature_configs: dict, feature_matrices: dict, labels: np.ndarray,
                         exclude_meta_from_retrieval: bool = True) -> dict:
    """Trọng số tỉ lệ với Fisher ratio của từng feature — minh bạch, giải thích được.

    Fisher ratio = S_B / S_W:
      S_B = phương sai giữa các lớp (between-class scatter)
      S_W = phương sai trong từng lớp (within-class scatter)

    Feature nào tách lớp tốt hơn (Fisher ratio cao hơn) sẽ được weight cao hơn.
    Đây là thống kê có giám sát nhưng hoàn toàn giải thích được — không phải học sâu.

    Trả về dict {feature_key: weight} đã normalize tổng = 1.
    """
    active = {
        k: v for k, v in feature_configs.items()
        if v['enabled'] and not (exclude_meta_from_retrieval and v['is_meta'])
        and k in feature_matrices and not feature_matrices[k].empty
    }
    if not active:
        return {}

    label_set = sorted(set(labels.tolist()))
    overall_mean_cache: dict = {}
    fisher_scores: dict = {}

    for key in active:
        matrix = feature_matrices[key]
        X = np.stack(matrix['vector'].tolist(), axis=0).astype(np.float32)
        overall_mean = X.mean(axis=0)
        overall_mean_cache[key] = overall_mean

        sw = 0.0
        sb = 0.0
        for lbl in label_set:
            cls = X[labels == lbl]
            if len(cls) == 0:
                continue
            m = cls.mean(axis=0)
            sw += float(np.sum((cls - m) ** 2))
            sb += float(len(cls) * np.sum((m - overall_mean) ** 2))
        fisher_scores[key] = float(sb / (sw + 1e-12))

    total = sum(max(0.0, v) for v in fisher_scores.values())
    if total < 1e-12:
        # Fallback: chia đều nếu mọi feature đều không phân biệt được
        n = len(active)
        return {k: 1.0 / n for k in active}
    return {k: max(0.0, fisher_scores[k]) / total for k in active}
