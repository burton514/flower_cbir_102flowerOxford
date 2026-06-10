from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage import morphology

from flower_cbir_app.utils.common import gray_to_rgb


def load_input_image(path: str | None = None, file_bytes: bytes | None = None) -> np.ndarray:
    if path is not None:
        image = Image.open(path)
    elif file_bytes is not None:
        image = Image.open(io.BytesIO(file_bytes))
    else:
        raise ValueError('Cần path hoặc file_bytes.')
    return np.array(image.convert('RGBA'))


def try_rembg(rgba_image: np.ndarray) -> np.ndarray:
    try:
        from rembg import remove
    except Exception as exc:
        raise RuntimeError('Chưa cài rembg/onnxruntime nhưng config đang yêu cầu dùng rembg.') from exc

    pil_image = Image.fromarray(rgba_image)
    out = remove(pil_image)
    return np.array(out.convert('RGBA'))


def _has_useful_alpha(alpha: np.ndarray) -> bool:
    # Ảnh RGB convert sang RGBA sẽ có alpha=255 toàn ảnh -> không có thông tin tách nền thật.
    return bool(np.any(alpha < 250))


def _fallback_white_background_mask(rgb: np.ndarray) -> np.ndarray:
    # Phương án cứu hộ nhẹ khi rembg thất bại hoặc không có alpha hữu ích.
    bg = (
        (rgb[..., 0] > 245) &
        (rgb[..., 1] > 245) &
        (rgb[..., 2] > 245)
    )
    return np.where(bg, 0, 255).astype(np.uint8)


def build_initial_mask(rgba_image: np.ndarray, use_rembg: bool, alpha_threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    rgba = rgba_image.copy()
    alpha = rgba[..., 3]

    # Nếu ảnh chưa có alpha thật thì mới gọi rembg.
    if use_rembg and not _has_useful_alpha(alpha):
        rgba = try_rembg(rgba)
        alpha = rgba[..., 3]

    # Nếu vẫn chưa có alpha hữu ích thì fallback tách nền trắng nhẹ.
    if not _has_useful_alpha(alpha):
        alpha = _fallback_white_background_mask(rgba[..., :3])

    mask = (alpha >= alpha_threshold).astype(np.uint8) * 255
    return rgba, mask




def _remove_small_holes(mask_bool: np.ndarray, area_threshold: int) -> np.ndarray:
    try:
        return morphology.remove_small_holes(mask_bool, max_size=area_threshold)
    except TypeError:
        return morphology.remove_small_holes(mask_bool, area_threshold=area_threshold)


def _remove_small_objects(mask_bool: np.ndarray, min_size: int) -> np.ndarray:
    try:
        return morphology.remove_small_objects(mask_bool, max_size=min_size)
    except TypeError:
        return morphology.remove_small_objects(mask_bool, min_size=min_size)

def clean_mask(mask: np.ndarray, kernel_size: int, min_area_ratio: float) -> np.ndarray:
    mask_bool = mask > 0
    mask_bool = _remove_small_holes(mask_bool, area_threshold=max(64, kernel_size * kernel_size * 4))
    min_size = max(32, int(mask.size * min_area_ratio))
    mask_bool = _remove_small_objects(mask_bool, min_size=min_size)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    if np.count_nonzero(mask_uint8) == 0:
        return mask

    # Giữ thành phần liên thông lớn nhất để bỏ rác quanh nền.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask_uint8 > 0).astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return mask_uint8

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = (labels == largest_label).astype(np.uint8) * 255
    return largest


def compute_bbox(mask: np.ndarray, margin_ratio: float) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    margin = int(max(bw, bh) * margin_ratio)

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(mask.shape[1] - 1, x1 + margin)
    y1 = min(mask.shape[0] - 1, y1 + margin)
    return x0, y0, x1 + 1, y1 + 1


def crop_by_bbox(image_rgba: np.ndarray, mask: np.ndarray, bbox) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox
    return image_rgba[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def center_and_scale_object(
    crop_rgba: np.ndarray,
    crop_mask: np.ndarray,
    target_size: int,
    target_object_ratio: float,
    white_background: bool = True,
):
    ys, xs = np.where(crop_mask > 0)
    h, w = crop_mask.shape

    if len(xs) == 0 or len(ys) == 0:
        canvas_rgb = np.full((target_size, target_size, 3), 255 if white_background else 0, dtype=np.uint8)
        canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)
        return canvas_rgb, canvas_mask

    crop_rgba = crop_rgba.copy()
    crop_rgba[..., 3] = np.where(crop_mask > 0, 255, 0).astype(np.uint8)

    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    desired = max(1, int(target_size * target_object_ratio))
    scale = desired / max(bbox_w, bbox_h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized_rgba = cv2.resize(crop_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    resized_mask = np.where(resized_mask > 0, 255, 0).astype(np.uint8)
    resized_rgba[..., 3] = np.where(resized_mask > 0, 255, 0).astype(np.uint8)

    canvas_rgb = np.full((target_size, target_size, 3), 255 if white_background else 0, dtype=np.uint8)
    canvas_mask = np.zeros((target_size, target_size), dtype=np.uint8)

    ys2, xs2 = np.where(resized_mask > 0)
    cy = int(np.mean(ys2)) if len(ys2) else new_h // 2
    cx = int(np.mean(xs2)) if len(xs2) else new_w // 2

    target_cx = target_size // 2
    target_cy = target_size // 2

    x0 = target_cx - cx
    y0 = target_cy - cy
    x1 = x0 + new_w
    y1 = y0 + new_h

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    dst_x1 = min(target_size, x1)
    dst_y1 = min(target_size, y1)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    src_rgb = resized_rgba[..., :3]
    src_mask = resized_mask > 0

    roi_rgb = canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1]
    roi_src_rgb = src_rgb[src_y0:src_y1, src_x0:src_x1]
    roi_src_mask = src_mask[src_y0:src_y1, src_x0:src_x1]

    roi_rgb[roi_src_mask] = roi_src_rgb[roi_src_mask]
    canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1] = roi_rgb
    canvas_mask[dst_y0:dst_y1, dst_x0:dst_x1] = resized_mask[src_y0:src_y1, src_x0:src_x1]
    return canvas_rgb, canvas_mask


def build_gray_and_edge(image_rgb: np.ndarray, mask: np.ndarray, canny_low: int = 80, canny_high: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    masked_gray = gray.copy()
    masked_gray[mask == 0] = 255

    blur = cv2.GaussianBlur(masked_gray, (3, 3), 0)
    edge = cv2.Canny(blur, canny_low, canny_high)
    edge[mask == 0] = 0
    return masked_gray, edge


def preprocess_image(path: str | None, config: dict, file_bytes: bytes | None = None) -> Dict[str, np.ndarray | dict]:
    rgba = load_input_image(path=path, file_bytes=file_bytes)

    rgba_bg, mask0 = build_initial_mask(
        rgba,
        use_rembg=config['preprocessing'].get('use_rembg', True),
        alpha_threshold=int(config['preprocessing'].get('alpha_threshold', 8)),
    )
    mask1 = clean_mask(
        mask0,
        kernel_size=int(config['preprocessing'].get('mask_kernel_size', 5)),
        min_area_ratio=float(config['preprocessing'].get('mask_min_area_ratio', 0.01)),
    )

    bbox = compute_bbox(mask1, margin_ratio=float(config['preprocessing'].get('bbox_margin_ratio', 0.08)))
    crop_rgba, crop_mask = crop_by_bbox(rgba_bg, mask1, bbox)

    image_rgb, mask_norm = center_and_scale_object(
        crop_rgba,
        crop_mask,
        target_size=int(config['preprocessing'].get('target_size', 256)),
        target_object_ratio=float(config['preprocessing'].get('target_object_ratio', 0.78)),
        white_background=bool(config['preprocessing'].get('white_background', True)),
    )
    gray, edge = build_gray_and_edge(
        image_rgb,
        mask_norm,
        canny_low=int(config['preprocessing'].get('canny_low', 80)),
        canny_high=int(config['preprocessing'].get('canny_high', 160)),
    )

    occupancy = float(np.count_nonzero(mask_norm) / mask_norm.size)
    ys, xs = np.where(mask_norm > 0)
    centroid_offset = 0.0
    if len(xs):
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        centroid_offset = float(np.sqrt((cx - mask_norm.shape[1] / 2) ** 2 + (cy - mask_norm.shape[0] / 2) ** 2))

    h, w = mask_norm.shape
    bbox_is_full = tuple(map(int, bbox)) == (0, 0, int(rgba.shape[1]), int(rgba.shape[0]))
    bad_mask = bool(occupancy > 0.9 or occupancy < 0.01 or bbox_is_full)

    # Ảnh preview sau xóa nền để debug rõ bước rembg/alpha.
    bg_removed_preview = np.full_like(rgba_bg[..., :3], 255, dtype=np.uint8)
    fg_mask = mask0 > 0
    bg_removed_preview[fg_mask] = rgba_bg[..., :3][fg_mask]

    debug_bundle = {
        'images': {
            'input_rgb': rgba[..., :3],
            'bg_removed_preview': bg_removed_preview,
            'mask_raw': gray_to_rgb(mask0),
            'mask_clean': gray_to_rgb(mask1),
            'crop_rgb': crop_rgba[..., :3],
            'normalized_rgb': image_rgb,
            'normalized_mask': gray_to_rgb(mask_norm),
            'gray': gray_to_rgb(gray),
            'canny_edge': gray_to_rgb(edge),
        },
        'plots': {},
        'tables': {
            'preprocess_stats': {
                'bbox': list(map(int, bbox)),
                'occupancy_ratio': occupancy,
                'centroid_offset': centroid_offset,
                'mask_pixels': int(np.count_nonzero(mask_norm)),
                'bad_mask': bad_mask,
            }
        },
    }

    return {
        'image_rgb': image_rgb,
        'mask': mask_norm,
        'gray': gray,
        'edge': edge,
        'debug_bundle': debug_bundle,
        'stats': {
            'bbox': bbox,
            'occupancy_ratio': occupancy,
            'centroid_offset': centroid_offset,
            'mask_pixels': int(np.count_nonzero(mask_norm)),
            'bad_mask': bad_mask,
        },
    }


def load_preprocessed_from_disk(normalized_path: str, mask_path: str, gray_path: str, edge_path: str) -> Dict[str, np.ndarray | dict] | None:
    """Nạp lại kết quả tiền xử lý đã lưu ra đĩa thay vì chạy lại preprocess_image().

    Trả về dict tối thiểu đủ cho các EXTRACTORS (image_rgb, mask, gray, edge) kèm
    debug_bundle/stats rỗng. Trả None nếu thiếu file để caller fallback sang
    preprocess_image() (tính lại từ ảnh gốc).

    Đây là tối ưu để extraction không tiền xử lý lại các ảnh đã xử lý ở bước offline.
    """
    try:
        if not all(p and Path(p).exists() for p in [normalized_path, mask_path, gray_path, edge_path]):
            return None
        norm_bgr = cv2.imread(normalized_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        if norm_bgr is None or mask is None or gray is None or edge is None:
            return None
        image_rgb = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.uint8) * 255

        occupancy = float(np.count_nonzero(mask) / mask.size) if mask.size else 0.0
        return {
            'image_rgb': image_rgb,
            'mask': mask,
            'gray': gray,
            'edge': edge,
            'debug_bundle': {'images': {}, 'plots': {}, 'tables': {}},
            'stats': {'occupancy_ratio': occupancy, 'loaded_from_disk': True},
        }
    except Exception:
        return None
