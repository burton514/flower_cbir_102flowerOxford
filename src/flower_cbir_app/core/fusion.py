from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def zscore_apply(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Áp z-score lên 1 vector query bằng mean/std đã đo sẵn lúc extract.

    Online dùng đúng mean/std của dataset (lưu trong feature_config) để query
    nằm cùng thang đo với các ảnh trong DB, không tự tính lại theo riêng query.
    """
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
    """Khoảng cách giữa HAI vector theo metric chọn (l2 / cosine / chi_square).

    - l2        : khoảng cách Euclid (càng nhỏ càng giống).
    - cosine    : 1 - cos(góc), clip [0,2]; vector 0 thì trả 1.0.
    - chi_square: ½ Σ (a-b)²/(a+b), chỉ cho histogram không âm đã L1-normalize.
    Trả về 1 số float. Phiên bản 1-query-nhiều-ảnh xem distances_to_matrix.
    """
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


def normalize_distance_fixed(distances, d_min: float, d_max: float) -> np.ndarray:
    """Normalize distance bằng min/max CỐ ĐỊNH (đo sẵn trên toàn dataset lúc extract).

    Khác với normalize_distance_values (min-max theo từng query → thang đo trôi),
    hàm này dùng thang đo cố định nên kết quả ổn định giữa các truy vấn khác nhau.
    Giá trị được clip về [0, 1] để query lạ không phá thang đo.
    """
    arr = np.asarray(distances, dtype=np.float32)
    if arr.size == 0:
        return arr
    rng = float(d_max) - float(d_min)
    if rng < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - float(d_min)) / (rng + 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def pairwise_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    """Ma trận khoảng cách N×N vectorized — dùng chung cho evaluation và stats.

    - cosine    : 1 - (X_norm @ X_normᵀ), clip [0, 2]
    - l2        : scipy cdist euclidean
    - chi_square: broadcasting theo hàng (histogram không âm)
    """
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=np.float32)
    if metric == 'cosine':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        Xn = X / norms
        D = 1.0 - (Xn @ Xn.T)
        return np.clip(D, 0.0, 2.0).astype(np.float32)
    if metric == 'l2':
        return cdist(X, X, metric='euclidean').astype(np.float32)
    if metric == 'chi_square':
        Xp = np.maximum(X, 0.0)
        n = len(Xp)
        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            diff = Xp[i] - Xp
            sumv = Xp[i] + Xp + 1e-10
            D[i] = 0.5 * np.sum(diff ** 2 / sumv, axis=1)
        return D
    raise ValueError(f'Unknown metric: {metric}')


def distances_to_matrix(query_vector: np.ndarray, matrix: np.ndarray, metric: str) -> np.ndarray:
    """Khoảng cách từ 1 query vector tới toàn bộ N hàng của matrix (vectorized).

    Trả về mảng shape (N,). Dùng trong online pipeline để bỏ vòng lặp Python.
    """
    q = np.asarray(query_vector, dtype=np.float32).ravel()
    M = np.asarray(matrix, dtype=np.float32)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    if metric == 'l2':
        return np.linalg.norm(M - q[None, :], axis=1).astype(np.float32)
    if metric == 'cosine':
        qn = np.linalg.norm(q)
        Mn = np.linalg.norm(M, axis=1)
        denom = (Mn * qn) + 1e-12
        sim = (M @ q) / denom
        out = 1.0 - sim
        out[(Mn < 1e-12) | (qn < 1e-12)] = 1.0
        return np.clip(out, 0.0, 2.0).astype(np.float32)
    if metric == 'chi_square':
        qp = np.maximum(q, 0.0)
        Mp = np.maximum(M, 0.0)
        diff = Mp - qp[None, :]
        sumv = Mp + qp[None, :] + 1e-10
        return (0.5 * np.sum(diff ** 2 / sumv, axis=1)).astype(np.float32)
    raise ValueError(f'Unknown metric: {metric}')


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


def aggregate_fixed_scale_with_details(dist_by_feature: dict, scale_by_feature: dict, weights: dict) -> tuple[dict, dict]:
    """Fusion dùng thang chuẩn hóa CỐ ĐỊNH (đo sẵn lúc extract), vectorized.

    dist_by_feature: {feature_key: (image_ids: np.ndarray[N], raw_dists: np.ndarray[N])}
    scale_by_feature: {feature_key: (d_min, d_max)}
    weights: {feature_key: weight}

    Trả về:
        aggregated: {image_id: final_distance_score}  (càng nhỏ càng giống)
        details:    {image_id: [contribution rows]}

    Khác aggregate_weighted_distances_with_details ở chỗ dùng min/max cố định
    (ổn định giữa các query) thay vì min-max theo từng tập ứng viên.
    """
    aggregated: dict = defaultdict(float)
    details: dict = defaultdict(list)
    for key, (image_ids, raw_dists) in dist_by_feature.items():
        if key not in weights or len(image_ids) == 0:
            continue
        d_min, d_max = scale_by_feature.get(key, (0.0, 1.0))
        normalized = normalize_distance_fixed(raw_dists, d_min, d_max)
        weight = float(weights.get(key, 0.0))
        contrib = weight * normalized
        for idx in range(len(image_ids)):
            image_id = int(image_ids[idx])
            aggregated[image_id] += float(contrib[idx])
            details[image_id].append({
                'feature_key': key,
                'raw_distance': float(raw_dists[idx]),
                'normalized_distance': float(normalized[idx]),
                'weight': weight,
                'contribution': float(contrib[idx]),
            })
    return dict(aggregated), dict(details)


def build_effective_weights(feature_configs: dict, auto_weight: bool = True, exclude_meta_from_retrieval: bool = True) -> dict:
    """Tính trọng số cuối cùng của mỗi feature dùng cho fusion (tổng = 1).

    - exclude_meta_from_retrieval: loại các feature meta khỏi việc xếp hạng.
    - auto_weight=True : chia đều theo NHÓM trước (mỗi nhóm 1/số_nhóm), rồi chia
      đều trong nhóm -> mọi nhóm đóng góp ngang nhau dù số feature khác nhau.
    - auto_weight=False: dùng weight tay người chỉnh, normalize tổng = 1 (nếu
      tổng <= 0 thì chia đều).

    Trả về dict {feature_key: weight}. Bản theo Fisher ratio xem build_fisher_weights.
    """
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
