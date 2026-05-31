from __future__ import annotations

from pathlib import Path

from flower_cbir_app.core.fusion import aggregate_weighted_distances_with_details, build_effective_weights, compute_distance, resolve_distance_type, zscore_apply
from flower_cbir_app.core.preprocessing import preprocess_image
from flower_cbir_app.features.local_features import bovw_feature_result, descriptors_to_bovw, extract_local_descriptors
from flower_cbir_app.features.registry import EXTRACTORS, LOCAL_FEATURE_MAP


def run_query(system_config: dict, feature_state: dict, db, query_image_path: str | None = None, query_file_bytes: bytes | None = None, top_k: int = 5) -> dict:
    extraction_run_id = db.get_latest_extraction_run_id()
    if extraction_run_id is None:
        raise ValueError('Chưa có extraction run trong SQLite.')
    configs = db.get_extraction_feature_configs(extraction_run_id)
    if not configs:
        raise ValueError('Không có feature config trong extraction run.')

    # Query phải dùng lại cấu hình preprocessing/fusion của extraction run mới nhất
    # để tránh lệch pipeline nếu người dùng chỉnh config sau khi đã extract.
    run_config = db.get_extraction_run_config(extraction_run_id) if hasattr(db, 'get_extraction_run_config') else {}
    query_system_config = run_config.get('system', system_config) if isinstance(run_config, dict) else system_config

    processed = preprocess_image(query_image_path, query_system_config, file_bytes=query_file_bytes)
    query_vectors = {}
    for key, cfg in configs.items():
        if key in EXTRACTORS:
            result = EXTRACTORS[key](processed)
            query_vectors[key] = zscore_apply(result.vector.astype('float32'), cfg['mean'], cfg['std'])
        elif key in LOCAL_FEATURE_MAP:
            desc, extra = extract_local_descriptors(
                processed['gray'], processed['mask'], LOCAL_FEATURE_MAP[key],
                max_desc_per_image=int(query_system_config['local_bovw'].get('max_descriptors_per_image', 300)),
            )
            hist = descriptors_to_bovw(desc, cfg['vocab'])
            result = bovw_feature_result(hist, LOCAL_FEATURE_MAP[key], extra.get('keypoint_count', 0))
            query_vectors[key] = zscore_apply(result.vector.astype('float32'), cfg['mean'], cfg['std'])

    weights = build_effective_weights(
        configs,
        auto_weight=query_system_config['fusion'].get('auto_weight', True),
        exclude_meta_from_retrieval=query_system_config['fusion'].get('exclude_meta_from_retrieval', True),
    )
    scores_by_feature = {}
    image_meta = {}
    for key in weights.keys():
        if key not in query_vectors:
            continue
        matrix = db.get_feature_matrix(extraction_run_id, key)
        if matrix.empty:
            continue
        rows = []
        metric = resolve_distance_type(configs[key])
        for _, row in matrix.iterrows():
            distance = compute_distance(query_vectors[key], row['vector'], metric)
            rows.append((row['image_id'], distance))
            image_meta[row['image_id']] = {'file_name': row['file_name'], 'label': row['label'], 'file_path': row['file_path']}
        scores_by_feature[key] = rows

    # Chuẩn hóa distance theo từng feature rồi mới cộng trọng số để fusion đúng thang đo.
    aggregated, details_by_image = aggregate_weighted_distances_with_details(scores_by_feature, weights)
    ranked = sorted(aggregated.items(), key=lambda x: x[1])
    results = []
    query_path_norm = str(Path(query_image_path).resolve()) if query_image_path else None
    for image_id, score in ranked:
        meta = image_meta[image_id]
        if query_path_norm is not None and str(Path(meta['file_path']).resolve()) == query_path_norm:
            continue
        distance_score = float(score)
        similarity = float(max(0.0, 1.0 - distance_score))
        rank = len(results) + 1
        feature_details = details_by_image.get(image_id, [])
        for item in feature_details:
            item['rank'] = rank
            item['file_name'] = meta['file_name']
        results.append({
            'image_id': image_id,
            'distance_score': distance_score,
            'similarity': similarity,
            'score': distance_score,  # giữ tương thích ngược; score ở đây là distance
            'feature_details': feature_details,
            **meta,
        })
        if len(results) >= top_k:
            break

    contribution_rows = []
    for row in results:
        contribution_rows.extend(row.get('feature_details', []))

    return {
        'query_debug_bundle': processed['debug_bundle'],
        'results': results,
        'per_feature_contributions': contribution_rows,
        'weights': weights,
    }
