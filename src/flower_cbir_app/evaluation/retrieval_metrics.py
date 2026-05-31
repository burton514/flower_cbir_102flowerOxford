from __future__ import annotations

from collections import defaultdict

import numpy as np

from flower_cbir_app.core.fusion import (
    aggregate_weighted_distances,
    build_effective_weights,
    compute_distance,
    resolve_distance_type,
)


def _precision_at_k(relevance):
    return float(np.mean(relevance)) if len(relevance) else 0.0


def _recall_at_k(relevance, total_relevant):
    if total_relevant <= 0:
        return 0.0
    return float(np.sum(relevance) / total_relevant)


def _average_precision_at_k(relevance, total_relevant, k=5):
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
    for idx, rel in enumerate(relevance, start=1):
        if rel:
            return float(1.0 / idx)
    return 0.0


def _get_fusion_config(db, extraction_run_id: int) -> dict:
    run_config = db.get_extraction_run_config(extraction_run_id) if hasattr(db, 'get_extraction_run_config') else {}
    system_config = run_config.get('system', run_config) if isinstance(run_config, dict) else {}
    return system_config.get('fusion', {}) if isinstance(system_config, dict) else {}


def evaluate_dataset_retrieval(db, extraction_run_id: int, progress_callback=None) -> dict:
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
    all_labels = np.asarray(base['label'].tolist())
    total = len(base)

    label_precisions: dict = defaultdict(list)
    label_recalls:    dict = defaultdict(list)
    label_maps:       dict = defaultdict(list)
    label_mrrs:       dict = defaultdict(list)
    label_skipped:    dict = defaultdict(int)

    cached_vectors = {key: matrices[key]['vector'].tolist() for key in matrices.keys()}
    metric_by_key = {key: resolve_distance_type(configs[key]) for key in matrices.keys()}
    evaluated_queries = 0
    skipped_queries = 0

    for i, (_, query_row) in enumerate(base.iterrows()):
        if progress_callback:
            fname = query_row.get('file_name', str(i))
            progress_callback(i / max(total, 1), f'[Đánh giá {i + 1}/{total}] {fname}')
        label = query_row['label']
        total_relevant = int(np.sum(all_labels == label)) - 1
        if total_relevant <= 0:
            skipped_queries += 1
            label_skipped[label] += 1
            continue

        candidate_info = []
        scores_by_feature = {key: [] for key in matrices.keys()}
        for j, (_, cand_row) in enumerate(base.iterrows()):
            if i == j:
                continue
            image_id = int(cand_row['image_id'])
            candidate_info.append((image_id, cand_row['label']))
            for key in matrices.keys():
                qv = cached_vectors[key][i]
                cv = cached_vectors[key][j]
                dist = compute_distance(qv, cv, metric_by_key[key])
                scores_by_feature[key].append((image_id, dist))

        aggregated = aggregate_weighted_distances(scores_by_feature, weights)
        label_by_image_id = {image_id: cand_label for image_id, cand_label in candidate_info}
        ranked = sorted(aggregated.items(), key=lambda x: x[1])
        top = ranked[:5]
        relevance = np.array([1 if label_by_image_id[image_id] == label else 0 for image_id, _ in top], dtype=np.float32)

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
