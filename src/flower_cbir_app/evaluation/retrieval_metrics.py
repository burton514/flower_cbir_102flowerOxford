from __future__ import annotations

from collections import defaultdict

import numpy as np

from flower_cbir_app.core.fusion import (
    build_effective_weights,
    normalize_distance_values,
    pairwise_distance_matrix,
    resolve_distance_type,
)


def _precision_at_k(relevance):
    """Precision@k = tỉ lệ ảnh đúng trong k kết quả trả về (mean mảng 0/1)."""
    return float(np.mean(relevance)) if len(relevance) else 0.0


def _recall_at_k(relevance, total_relevant):
    """Recall@k = số ảnh đúng trong top-k / tổng số ảnh đúng có trong DB."""
    if total_relevant <= 0:
        return 0.0
    return float(np.sum(relevance) / total_relevant)


def _average_precision_at_k(relevance, total_relevant, k=5):
    """Average Precision@k — trung bình precision tại mỗi vị trí có ảnh đúng.

    Cộng (số hit tích lũy / vị trí) ở mỗi kết quả đúng, chia cho
    min(total_relevant, k). Thưởng việc xếp ảnh đúng lên đầu. MAP = trung bình
    AP trên mọi query.
    """
    if total_relevant <= 0:
        return 0.0
    hits = 0
    ap = 0.0
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            ap += hits / idx
    return float(ap / max(1, min(total_relevant, k)))


def _mrr_at_k(relevance):
    """MRR = 1 / vị trí của ảnh ĐÚNG đầu tiên trong kết quả (0 nếu không có).

    Đo hệ thống đưa được 1 ảnh đúng lên sớm tới mức nào. MRR cuối = trung bình
    trên mọi query.
    """
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            return float(1.0 / idx)
    return 0.0


def _get_fusion_config(db, extraction_run_id: int) -> dict:
    """Đọc phần config 'fusion' (auto_weight, exclude_meta...) của extraction run."""
    run_config = db.get_extraction_run_config(extraction_run_id) if hasattr(db, 'get_extraction_run_config') else {}
    system_config = run_config.get('system', run_config) if isinstance(run_config, dict) else {}
    return system_config.get('fusion', {}) if isinstance(system_config, dict) else {}


def _compute_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    """Wrapper sang pairwise_distance_matrix dùng chung trong fusion."""
    return pairwise_distance_matrix(X, metric)


def evaluate_dataset_retrieval(db, extraction_run_id: int, progress_callback=None) -> dict:
    """Đánh giá chất lượng truy hồi trên TOÀN dataset (mỗi ảnh làm 1 query).

    Cách làm: dựng ma trận khoảng cách fused N×N (mỗi feature: distance matrix ->
    normalize min-max -> nhân trọng số -> cộng), đặt đường chéo = vô cực để loại
    chính ảnh query. Với mỗi ảnh: lấy top-5, đối chiếu nhãn để tính
    Precision/Recall/MAP/MRR @5. Bỏ qua query mà nhãn của nó chỉ có 1 ảnh.

    Trả về dict gồm 4 metric trung bình toàn cục + bảng per_label + số query
    đã đánh giá/bỏ qua. KHÔNG lưu vào DB (kết quả chỉ về session_state của UI).
    """
    configs = db.get_extraction_feature_configs(extraction_run_id)
    fusion_config = _get_fusion_config(db, extraction_run_id)
    weights = build_effective_weights(
        configs,
        auto_weight=fusion_config.get('auto_weight', True),
        exclude_meta_from_retrieval=fusion_config.get('exclude_meta_from_retrieval', True),
    )
    matrices = {k: db.get_feature_matrix(extraction_run_id, k) for k in weights.keys()}
    matrices = {k: v for k, v in matrices.items() if not v.empty}
    if not matrices:
        return {
            'precision_at_5': 0.0, 'recall_at_5': 0.0,
            'map_at_5': 0.0, 'mrr_at_5': 0.0,
            'evaluated_queries': 0, 'skipped_queries': 0,
            'per_label': [],
        }

    base = next(iter(matrices.values()))
    all_labels    = np.asarray(base['label'].tolist())
    all_image_ids = np.asarray(base['image_id'].tolist(), dtype=np.int64)
    total = len(base)

    # ── Tính ma trận khoảng cách N×N cho từng feature một lần ──────────────
    # Mỗi feature: stack vector → distance matrix → normalize upper-tri
    # → nhân trọng số → cộng vào D_fused
    # Thay thế 2 vòng lặp lồng nhau O(N²) Python calls trước đây.
    D_fused = np.zeros((total, total), dtype=np.float32)
    for key, matrix in matrices.items():
        X = np.stack(matrix['vector'].tolist(), axis=0).astype(np.float32)
        metric = resolve_distance_type(configs[key])
        D_key = _compute_distance_matrix(X, metric)

        tri = np.triu_indices(total, k=1)
        norm_vals = normalize_distance_values(D_key[tri])
        D_norm = np.zeros((total, total), dtype=np.float32)
        D_norm[tri] = norm_vals
        D_norm[(tri[1], tri[0])] = norm_vals

        D_fused += float(weights.get(key, 0.0)) * D_norm

    np.fill_diagonal(D_fused, np.inf)  # loại query khỏi kết quả

    label_precisions: dict = defaultdict(list)
    label_recalls:    dict = defaultdict(list)
    label_maps:       dict = defaultdict(list)
    label_mrrs:       dict = defaultdict(list)
    label_skipped:    dict = defaultdict(int)
    evaluated_queries = 0
    skipped_queries = 0

    file_names = base.get('file_name', None)

    for i in range(total):
        if progress_callback:
            fname = file_names.iloc[i] if file_names is not None else str(i)
            progress_callback(i / max(total, 1), f'[Đánh giá {i + 1}/{total}] {fname}')

        label = all_labels[i]
        total_relevant = int(np.sum(all_labels == label)) - 1
        if total_relevant <= 0:
            skipped_queries += 1
            label_skipped[label] += 1
            continue

        row = D_fused[i]
        sorted_idx = np.argsort(row)[:5]
        top_labels = all_labels[sorted_idx]
        relevance  = (top_labels == label).astype(np.float32)

        label_precisions[label].append(_precision_at_k(relevance))
        label_recalls[label].append(_recall_at_k(relevance, total_relevant))
        label_maps[label].append(_average_precision_at_k(relevance, total_relevant, k=5))
        label_mrrs[label].append(_mrr_at_k(relevance))
        evaluated_queries += 1

    if progress_callback:
        progress_callback(1.0, 'Hoàn tất đánh giá.')

    all_unique_labels = sorted(set(all_labels.tolist()))
    per_label = []
    all_p, all_r, all_m, all_mrr_list = [], [], [], []
    for lbl in all_unique_labels:
        values = label_precisions.get(lbl, [])
        cnt = len(values)
        if cnt > 0:
            p  = float(np.mean(label_precisions[lbl]))
            r  = float(np.mean(label_recalls[lbl]))
            m  = float(np.mean(label_maps[lbl]))
            mr = float(np.mean(label_mrrs[lbl]))
            all_p.extend(label_precisions[lbl])
            all_r.extend(label_recalls[lbl])
            all_m.extend(label_maps[lbl])
            all_mrr_list.extend(label_mrrs[lbl])
        else:
            p = r = m = mr = 0.0
        per_label.append({
            'label':          lbl,
            'count':          cnt,
            'skipped':        int(label_skipped.get(lbl, 0)),
            'precision_at_5': round(p,  4),
            'recall_at_5':    round(r,  4),
            'map_at_5':       round(m,  4),
            'mrr_at_5':       round(mr, 4),
        })

    return {
        'precision_at_5': float(np.mean(all_p)) if all_p else 0.0,
        'recall_at_5':    float(np.mean(all_r)) if all_r else 0.0,
        'map_at_5':       float(np.mean(all_m)) if all_m else 0.0,
        'mrr_at_5':       float(np.mean(all_mrr_list)) if all_mrr_list else 0.0,
        'evaluated_queries': int(evaluated_queries),
        'skipped_queries': int(skipped_queries),
        'per_label':      per_label,
    }
