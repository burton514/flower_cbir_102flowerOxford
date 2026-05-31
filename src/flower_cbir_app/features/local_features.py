from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from flower_cbir_app.features.base import FeatureResult
from flower_cbir_app.utils.common import normalize_vector


def _get_detector(name: str):
    name = name.lower()
    if name == 'sift':
        if hasattr(cv2, 'SIFT_create'):
            return cv2.SIFT_create(nfeatures=300)
        raise RuntimeError('OpenCV hiện tại không có SIFT_create.')
    if name == 'orb':
        return cv2.ORB_create(nfeatures=300)
    if name == 'akaze':
        return cv2.AKAZE_create()
    if name == 'brisk':
        return cv2.BRISK_create()
    raise ValueError(name)


def get_descriptor_dim(method: str) -> int:
    detector = _get_detector(method)
    try:
        dim = int(detector.descriptorSize())
        if dim > 0:
            return dim
    except Exception:
        pass
    # fallback theo descriptor phổ biến của OpenCV
    return {'sift': 128, 'orb': 32, 'akaze': 61, 'brisk': 64}.get(method.lower(), 32)


def extract_local_descriptors(gray: np.ndarray, mask: np.ndarray, method: str, max_desc_per_image: int = 300):
    detector = _get_detector(method)
    descriptor_dim = get_descriptor_dim(method)
    keypoints = detector.detect(gray, mask=mask)
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_desc_per_image]
    if not keypoints:
        return np.empty((0, descriptor_dim), dtype=np.float32), {'keypoint_count': 0, 'descriptor_dim': descriptor_dim}
    keypoints, descriptors = detector.compute(gray, keypoints)
    if descriptors is None or len(descriptors) == 0:
        return np.empty((0, descriptor_dim), dtype=np.float32), {'keypoint_count': 0, 'descriptor_dim': descriptor_dim}
    descriptors = descriptors.astype(np.float32)
    return descriptors, {'keypoint_count': len(keypoints), 'descriptor_dim': int(descriptors.shape[1])}


def fit_bovw_vocabulary(descriptor_list: list[np.ndarray], vocab_size: int, random_state: int = 42, descriptor_dim: int = 32):
    valid = [d for d in descriptor_list if d is not None and len(d) > 0]
    if not valid:
        return np.zeros((vocab_size, descriptor_dim), dtype=np.float32)
    all_desc = np.vstack(valid).astype(np.float32)
    descriptor_dim = int(all_desc.shape[1])
    if len(all_desc) < vocab_size:
        pad = np.repeat(all_desc[:1], vocab_size - len(all_desc), axis=0)
        all_desc = np.vstack([all_desc, pad])
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=min(2048, max(vocab_size * 8, 256)), random_state=random_state, n_init=10)
    kmeans.fit(all_desc)
    return kmeans.cluster_centers_.astype(np.float32)


def descriptors_to_bovw(descriptors: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    if vocab is None or len(vocab) == 0:
        return np.zeros(0, dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(vocab), dtype=np.float32)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    vocab = np.asarray(vocab, dtype=np.float32)
    if descriptors.shape[1] != vocab.shape[1]:
        # DB cũ hoặc config cũ có thể lưu vocab sai chiều; trả histogram rỗng thay vì crash.
        return np.zeros(len(vocab), dtype=np.float32)
    dists = np.linalg.norm(descriptors[:, None, :] - vocab[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    hist = np.bincount(nearest, minlength=len(vocab)).astype(np.float32)
    return normalize_vector(hist)


def bovw_feature_result(hist: np.ndarray, method: str, keypoint_count: int) -> FeatureResult:
    return FeatureResult(
        hist.astype(np.float32),
        {'images': {}, 'plots': {f'{method.upper()} BoVW histogram': {'y': hist.tolist()}}, 'tables': {f'{method.upper()} local': {'keypoints': int(keypoint_count)}}},
        {'keypoint_count': int(keypoint_count)},
    )
