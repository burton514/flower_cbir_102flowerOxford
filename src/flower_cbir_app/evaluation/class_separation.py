from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score

from flower_cbir_app.core.fusion import (
    build_effective_weights,
    normalize_distance_values,
    pairwise_distance_matrix,
    resolve_distance_type,
)


def _get_fusion_config(db, extraction_run_id: int) -> dict:
    """Đọc phần config 'fusion' (auto_weight, exclude_meta...) của extraction run."""
    run_config = db.get_extraction_run_config(extraction_run_id) if hasattr(db, 'get_extraction_run_config') else {}
    system_config = run_config.get('system', run_config) if isinstance(run_config, dict) else {}
    return system_config.get('fusion', {}) if isinstance(system_config, dict) else {}


def _pairwise_feature_distance(vectors: list[np.ndarray], metric: str) -> np.ndarray:
    """Tính ma trận khoảng cách N×N bằng vectorization rồi normalize.

    Dùng chung pairwise_distance_matrix với fusion/retrieval.
    """
    X = np.stack(vectors, axis=0).astype(np.float32)
    n = len(X)
    D = pairwise_distance_matrix(X, metric)
    np.fill_diagonal(D, 0.0)

    if n > 1:
        tri = np.triu_indices(n, k=1)
        norm_vals = normalize_distance_values(D[tri])
        D_norm = np.zeros_like(D)
        D_norm[tri] = norm_vals
        D_norm[(tri[1], tri[0])] = norm_vals
        return D_norm
    return D


def evaluate_class_separation(db, extraction_run_id: int) -> dict:
    """Đo mức độ TÁCH LỚP của không gian đặc trưng fused (không cần truy vấn).

    Dựng ma trận khoảng cách fused N×N rồi tính:
      - intra/inter_class_distance: khoảng cách trung bình trong-lớp và giữa-lớp.
      - separation_ratio = inter/intra (càng lớn càng tách tốt).
      - silhouette: điểm silhouette trên ma trận khoảng cách precomputed.
      - fisher_ratio = S_B / S_W trên vector nối toàn bộ feature.
    Trả về dict các chỉ số này + số mẫu/feature. Bổ trợ cho retrieval metrics:
    đánh giá feature có gom cùng lớp / tách khác lớp tốt không.
    """
    configs = db.get_extraction_feature_configs(extraction_run_id)
    fusion_config = _get_fusion_config(db, extraction_run_id)
    weights = build_effective_weights(
        configs,
        auto_weight=fusion_config.get('auto_weight', True),
        exclude_meta_from_retrieval=fusion_config.get('exclude_meta_from_retrieval', True),
    )
    if not weights:
        return {
            'intra_class_distance': 0.0,
            'inter_class_distance': 0.0,
            'separation_ratio': 0.0,
            'silhouette': 0.0,
            'fisher_ratio': 0.0,
            'num_samples': 0,
            'num_features': 0,
        }

    matrices = {k: db.get_feature_matrix(extraction_run_id, k) for k in weights.keys()}
    matrices = {k: v for k, v in matrices.items() if not v.empty}
    if not matrices:
        return {
            'intra_class_distance': 0.0,
            'inter_class_distance': 0.0,
            'separation_ratio': 0.0,
            'silhouette': 0.0,
            'fisher_ratio': 0.0,
            'num_samples': 0,
            'num_features': 0,
        }

    base = next(iter(matrices.values()))
    labels = np.asarray(base['label'].tolist())
    n = len(labels)

    D = np.zeros((n, n), dtype=np.float32)
    for key, matrix in matrices.items():
        vectors = matrix['vector'].tolist()
        metric = resolve_distance_type(configs[key])
        D += float(weights.get(key, 0.0)) * _pairwise_feature_distance(vectors, metric)
    np.fill_diagonal(D, 0.0)

    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra.append(D[i, j])
            else:
                inter.append(D[i, j])
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    sep_ratio = float(inter_mean / (intra_mean + 1e-12)) if intra_mean > 0 else 0.0

    label_set = set(labels.tolist())
    label_counts = {lbl: int(np.sum(labels == lbl)) for lbl in label_set}
    can_silhouette = len(label_set) > 1 and n > len(label_set) and min(label_counts.values()) >= 2
    sil = float(silhouette_score(D, labels, metric='precomputed')) if can_silhouette else 0.0

    X = np.concatenate([np.stack(matrices[k]['vector'].to_list(), axis=0) for k in matrices.keys()], axis=1)
    overall_mean = X.mean(axis=0)
    sw = 0.0
    sb = 0.0
    for label in sorted(label_set):
        cls = X[labels == label]
        m = cls.mean(axis=0)
        sw += float(np.sum((cls - m) ** 2))
        sb += float(len(cls) * np.sum((m - overall_mean) ** 2))
    fisher = float(sb / (sw + 1e-12))
    return {
        'intra_class_distance': intra_mean,
        'inter_class_distance': inter_mean,
        'separation_ratio': sep_ratio,
        'silhouette': sil,
        'fisher_ratio': fisher,
        'num_samples': int(n),
        'num_features': int(len(matrices)),
    }
