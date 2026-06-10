from __future__ import annotations

from pathlib import Path

import cv2
import hashlib
import re
import numpy as np

from flower_cbir_app.core.fusion import pairwise_distance_matrix, resolve_distance_type, zscore_apply
from flower_cbir_app.core.preprocessing import load_preprocessed_from_disk, preprocess_image
from flower_cbir_app.features.local_features import bovw_feature_result, descriptors_to_bovw, extract_local_descriptors, fit_bovw_vocabulary, get_descriptor_dim
from flower_cbir_app.features.registry import EXTRACTORS, LOCAL_FEATURE_MAP, get_feature_catalog
from flower_cbir_app.storage.sqlite_manager import SQLiteManager
from flower_cbir_app.utils.common import ensure_dir, parse_label_from_path


def _list_images(dataset_root: str, exts: list[str]):
    dataset_root = Path(dataset_root)
    allowed = {str(ext).lower() for ext in exts}
    return sorted([p for p in dataset_root.rglob('*') if p.is_file() and p.suffix.lower() in allowed])



def _safe_output_stem(relative_name: str) -> str:
    rel = Path(relative_name)
    no_suffix = rel.with_suffix('')
    raw = '__'.join(no_suffix.parts)
    safe = re.sub(r'[^0-9A-Za-zÀ-ỹ_-]+', '_', raw).strip('_') or 'image'
    short_hash = hashlib.md5(str(rel).encode('utf-8')).hexdigest()[:8]
    return f'{safe}_{short_hash}'

def _save_preprocessed_outputs(workspace_root: str, relative_name: str, processed: dict):
    pre_dir = ensure_dir(Path(workspace_root) / 'preprocessed')
    stem = _safe_output_stem(relative_name)

    norm_path = pre_dir / f'{stem}_normalized.png'
    mask_path = pre_dir / f'{stem}_mask.png'
    gray_path = pre_dir / f'{stem}_gray.png'
    edge_path = pre_dir / f'{stem}_edge.png'

    cv2.imwrite(str(norm_path), cv2.cvtColor(processed['image_rgb'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), processed['mask'])
    cv2.imwrite(str(gray_path), processed['gray'])
    cv2.imwrite(str(edge_path), processed['edge'])

    # Lưu thêm debug trung gian để soi lỗi mask/rembg/crop khi cần.
    debug_dir = ensure_dir(pre_dir / 'debug')
    images = processed.get('debug_bundle', {}).get('images', {})
    extra_map = {
        'input_rgb': f'{stem}_input.png',
        'bg_removed_preview': f'{stem}_bg_removed_preview.png',
        'mask_raw': f'{stem}_mask_raw.png',
        'mask_clean': f'{stem}_mask_clean.png',
        'crop_rgb': f'{stem}_crop.png',
        'normalized_rgb': f'{stem}_normalized_debug.png',
        'normalized_mask': f'{stem}_normalized_mask_debug.png',
        'gray': f'{stem}_gray_debug.png',
        'canny_edge': f'{stem}_edge_debug.png',
    }
    for key, name in extra_map.items():
        if key not in images:
            continue
        img = images[key]
        out = debug_dir / name
        if img.ndim == 3:
            cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(out), img)

    return str(norm_path), str(mask_path), str(gray_path), str(edge_path)


def inspect_dataset(system_config: dict, max_read: int | None = None) -> dict:
    """Thống kê nhanh dataset để phục vụ kiểm tra yêu cầu đề bài.

    Không thay đổi CSDL. Hàm này giúp chứng minh số lượng ảnh, nhãn, kích thước,
    định dạng và quy ước tên file `nhãn_####`.
    """
    from collections import Counter
    from PIL import Image

    dataset_root = system_config['dataset_root']
    files = _list_images(dataset_root, system_config['image_extensions'])
    if max_read is not None:
        files_to_read = files[:max_read]
    else:
        files_to_read = files

    label_source = system_config.get('label_source', 'auto')
    labels = [parse_label_from_path(p, dataset_root, label_source) for p in files]
    label_counts = Counter(labels)
    extensions = Counter(p.suffix.lower() for p in files)
    sizes = Counter()
    modes = Counter()
    invalid_names = []
    alpha_count = 0
    for p in files_to_read:
        if '_' not in p.stem:
            invalid_names.append(p.name)
        try:
            with Image.open(p) as img:
                sizes[f'{img.size[0]}x{img.size[1]}'] += 1
                modes[img.mode] += 1
                if img.mode in ('RGBA', 'LA') or ('transparency' in img.info):
                    alpha_count += 1
        except Exception:
            sizes['unreadable'] += 1

    return {
        'num_images': len(files),
        'num_labels': len(label_counts),
        'image_extensions': dict(sorted(extensions.items())),
        'image_sizes': dict(sizes.most_common()),
        'image_modes': dict(modes.most_common()),
        'alpha_or_transparency_count_checked': int(alpha_count),
        'num_checked_for_size_mode': len(files_to_read),
        'min_images_per_label': int(min(label_counts.values())) if label_counts else 0,
        'max_images_per_label': int(max(label_counts.values())) if label_counts else 0,
        'label_counts': dict(sorted(label_counts.items(), key=lambda x: str(x[0]))),
        'invalid_name_examples': invalid_names[:20],
        'label_source': label_source,
    }


def run_offline_preprocess(system_config: dict, sample_limit: int = 5, progress_callback=None) -> dict:
    dataset_root = system_config['dataset_root']
    if not dataset_root:
        raise ValueError('Dataset root đang trống.')
    files = _list_images(dataset_root, system_config['image_extensions'])
    if not files:
        raise ValueError('Không tìm thấy ảnh nào trong dataset root.')

    db = SQLiteManager(system_config['db_path'])
    db.reset_preprocess_data()  # Xóa sạch dữ liệu preprocess + extraction cũ trước khi chạy lại
    preprocess_run_id = db.create_preprocess_run(system_config)
    samples = []
    total = len(files)
    for idx, path in enumerate(files):
        if progress_callback:
            progress_callback(idx / total, f'[{idx + 1}/{total}] Tiền xử lí: {path.name}')
        processed = preprocess_image(str(path), system_config)
        label = parse_label_from_path(path, dataset_root, system_config.get('label_source', 'auto'))
        image_id = db.upsert_image(str(path), path.name, label)
        try:
            relative_name = str(path.relative_to(Path(dataset_root)))
        except ValueError:
            relative_name = path.name
        norm_path, mask_path, gray_path, edge_path = _save_preprocessed_outputs(system_config['workspace_root'], relative_name, processed)
        db.insert_preprocess_output(preprocess_run_id, image_id, norm_path, mask_path, gray_path, edge_path, processed['stats'])
        if idx < sample_limit:
            samples.append({'file_name': path.name, 'debug_bundle': processed['debug_bundle']})
    if progress_callback:
        progress_callback(1.0, f'Hoàn tất {total} ảnh.')
    return {
        'preprocess_run_id': preprocess_run_id,
        'num_images': len(files),
        'message': f'Đã tiền xử lí offline {len(files)} ảnh.',
        'samples': samples,
    }


def run_feature_extraction(system_config: dict, feature_state: dict, progress_callback=None) -> dict:
    dataset_root = system_config['dataset_root']
    if not dataset_root:
        raise ValueError('Dataset root đang trống.')
    files = _list_images(dataset_root, system_config['image_extensions'])
    if not files:
        raise ValueError('Không tìm thấy ảnh nào trong dataset root.')

    catalog = {spec.key: spec for spec in get_feature_catalog()}
    db = SQLiteManager(system_config['db_path'])
    db.reset_extraction_data()  # Xóa sạch dữ liệu extraction cũ trước khi chạy lại
    extraction_run_id = db.create_extraction_run({'system': system_config, 'features': feature_state})

    # Tái dùng ảnh đã tiền xử lý ở bước offline (nếu có) để không preprocess 2 lần.
    preprocessed_map = db.get_preprocess_outputs_map()
    reused_count = 0

    enabled_features = [k for k, v in feature_state.items() if v['enabled']]
    enabled_standard = [k for k in enabled_features if k in EXTRACTORS]
    enabled_local = [k for k in enabled_features if k in LOCAL_FEATURE_MAP]

    all_processed = []
    sample_debug = []
    total = len(files)
    for idx, path in enumerate(files):
        if progress_callback:
            progress_callback(idx / total * 0.7, f'[{idx + 1}/{total}] Trích xuất: {path.name}')
        label = parse_label_from_path(path, dataset_root, system_config.get('label_source', 'auto'))
        image_id = db.upsert_image(str(path), path.name, label)

        # Ưu tiên nạp lại ảnh chuẩn hóa từ đĩa; chỉ preprocess lại nếu thiếu file.
        processed = None
        paths = preprocessed_map.get(image_id)
        if paths is not None:
            processed = load_preprocessed_from_disk(
                paths['normalized_path'], paths['mask_path'], paths['gray_path'], paths['edge_path']
            )
        if processed is None:
            processed = preprocess_image(str(path), system_config)
        else:
            reused_count += 1

        record = {
            'image_id': image_id,
            'file_name': path.name,
            'file_path': str(path),
            'label': label,
            'processed': processed,
            'feature_vectors': {},
            'local_desc': {},
        }
        for key in enabled_standard:
            result = EXTRACTORS[key](processed)
            record['feature_vectors'][key] = result.vector.astype(np.float32)
            if idx < 3:
                processed['debug_bundle']['tables'][f'feature_{key}'] = {'dim': int(result.vector.size)}
        for key in enabled_local:
            method = LOCAL_FEATURE_MAP[key]
            desc, extra = extract_local_descriptors(
                processed['gray'],
                processed['mask'],
                method=method,
                max_desc_per_image=int(system_config['local_bovw'].get('max_descriptors_per_image', 300)),
            )
            record['local_desc'][key] = (desc, extra)
        if idx < 3:
            sample_debug.append({'file_name': path.name, 'debug_bundle': processed['debug_bundle']})
        all_processed.append(record)

    # Fit vocabularies and convert local descriptors to fixed-size histograms.
    vocab_size = int(system_config['local_bovw'].get('vocab_size', 32))
    max_fit = int(system_config['local_bovw'].get('max_descriptors_fit', 12000))
    vocabularies = {}
    for bovw_idx, key in enumerate(enabled_local):
        if progress_callback:
            progress_callback(0.7 + bovw_idx / max(len(enabled_local), 1) * 0.1, f'Fitting BoVW vocabulary: {key}...')
        descriptor_list = []
        total_desc = 0
        for item in all_processed:
            desc, _ = item['local_desc'][key]
            if len(desc) == 0:
                continue
            if total_desc >= max_fit:
                break
            take = min(len(desc), max_fit - total_desc)
            descriptor_list.append(desc[:take])
            total_desc += take
        descriptor_dim = get_descriptor_dim(LOCAL_FEATURE_MAP[key])
        vocab = fit_bovw_vocabulary(
            descriptor_list,
            vocab_size=vocab_size,
            random_state=int(system_config.get('random_state', 42)),
            descriptor_dim=descriptor_dim,
        ) if descriptor_list else np.zeros((vocab_size, descriptor_dim), dtype=np.float32)
        vocabularies[key] = vocab
        for item in all_processed:
            desc, extra = item['local_desc'][key]
            hist = descriptors_to_bovw(desc, vocab)
            result = bovw_feature_result(hist, LOCAL_FEATURE_MAP[key], extra.get('keypoint_count', 0))
            item['feature_vectors'][key] = result.vector.astype(np.float32)

    # Mảng image_id theo đúng thứ tự xử lý (dùng chung cho mọi feature matrix).
    image_ids = np.asarray([item['image_id'] for item in all_processed], dtype=np.int64)

    # Normalize per feature: L1 cho histogram, z-score cho non-histogram
    if progress_callback:
        progress_callback(0.82, 'Chuẩn hoá vector...')
    feature_configs_saved = {}
    normalized_matrices = {}  # key → ma trận N×D đã chuẩn hóa (để lưu gộp + tính scale)
    for key in enabled_features:
        spec = catalog[key]
        vectors = np.stack([item['feature_vectors'][key] for item in all_processed], axis=0).astype(np.float32)

        if spec.is_histogram:
            # L1-normalize: chia cho tổng mỗi hàng, đảm bảo sum=1 và giá trị >=0
            row_sums = vectors.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
            vectors_norm = vectors / row_sums
            mean = np.zeros(vectors.shape[1], dtype=np.float32)
            std  = np.ones(vectors.shape[1], dtype=np.float32)
        else:
            # Z-score normalize
            mean = vectors.mean(axis=0)
            std  = vectors.std(axis=0)
            std[std < 1e-8] = 1.0
            vectors_norm = (vectors - mean) / std

        vectors_norm = vectors_norm.astype(np.float32)
        normalized_matrices[key] = vectors_norm

        # Distance type thực tế (fallback chi_square→l2 nếu không hợp lệ).
        distance_type = ('l2' if feature_state[key].get('distance') == 'chi_square' and not spec.supports_chi_square
                         else feature_state[key].get('distance', spec.default_distance))

        # Tính thang đo distance CỐ ĐỊNH trên toàn dataset (min/max của upper-triangle)
        # để online normalize ổn định, không bị trôi theo từng query.
        d_min, d_max = 0.0, 1.0
        n = len(vectors_norm)
        if n > 1:
            D = pairwise_distance_matrix(vectors_norm, distance_type)
            tri = np.triu_indices(n, k=1)
            tri_vals = D[tri]
            if tri_vals.size:
                d_min = float(np.min(tri_vals))
                d_max = float(np.max(tri_vals))

        db.insert_feature_config(
            extraction_run_id,
            feature_key=key,
            group_name=spec.group,
            enabled=True,
            distance_type=distance_type,
            weight=float(feature_state[key]['weight']),
            is_meta=spec.is_meta,
            mean=mean,
            std=std,
            vocab=vocabularies.get(key),
            d_min=d_min,
            d_max=d_max,
            extra={'output_dim': int(vectors.shape[1]), 'is_histogram': spec.is_histogram, 'supports_chi_square': spec.supports_chi_square},
        )
        feature_configs_saved[key] = {
            'dim': int(vectors.shape[1]),
            'norm': 'l1' if spec.is_histogram else 'zscore',
            'distance': distance_type,
            'd_min': round(d_min, 6),
            'd_max': round(d_max, 6),
        }

    # Lưu GỘP ma trận từng feature (một blob/feature) thay vì 1 dòng/ảnh/feature.
    if progress_callback:
        progress_callback(0.92, 'Lưu ma trận đặc trưng vào SQLite...')
    for key in enabled_features:
        db.insert_feature_matrix(extraction_run_id, key, image_ids, normalized_matrices[key])

    summary = {
        'num_images': len(all_processed),
        'num_enabled_features': len(enabled_features),
        'reused_preprocessed': int(reused_count),
        'features': feature_configs_saved,
    }
    db.update_extraction_run_summary(extraction_run_id, summary)
    if progress_callback:
        progress_callback(1.0, f'Hoàn tất {len(all_processed)} ảnh, {len(enabled_features)} feature.')
    return {
        'extraction_run_id': extraction_run_id,
        'num_images': len(all_processed),
        'enabled_features': enabled_features,
        'reused_preprocessed': int(reused_count),
        'message': f'Đã trích xuất đặc trưng cho {len(all_processed)} ảnh với {len(enabled_features)} feature (tái dùng {reused_count} ảnh tiền xử lý).',
        'sample_debug': sample_debug,
    }
