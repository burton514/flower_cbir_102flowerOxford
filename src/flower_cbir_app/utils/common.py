
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class DebugBundle:
    images: Dict[str, np.ndarray]
    plots: Dict[str, dict]
    tables: Dict[str, dict]


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    gray = image_to_uint8(gray)
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    s = np.sum(vector)
    if s > eps:
        vector = vector / s
    return vector


def np_to_blob(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(array), allow_pickle=False)
    return buf.getvalue()


def blob_to_np(blob: bytes) -> np.ndarray:
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf, allow_pickle=False)


# ── Lưu trữ mảng dạng raw bytes (gọn hơn .npy ~128 byte header/mảng) ──────────
# Dùng cho ma trận feature lớn (N×D) và mảng image_id. Phải tự quản lý dtype/shape.

def array_to_raw_blob(array: np.ndarray) -> bytes:
    """Serialize mảng float32 thành raw bytes (không header).

    Gọn hơn np.save khi lưu hàng nghìn vector. Caller phải lưu kèm shape + dtype.
    """
    arr = np.ascontiguousarray(array, dtype=np.float32)
    return arr.tobytes()


def raw_blob_to_matrix(blob: bytes, rows: int, dim: int) -> np.ndarray:
    """Giải nén raw bytes thành ma trận float32 (rows × dim)."""
    arr = np.frombuffer(blob, dtype=np.float32)
    if rows * dim != arr.size:
        # Fallback an toàn: suy ra rows từ dim nếu lệch metadata.
        rows = arr.size // max(dim, 1)
    return arr.reshape(rows, dim).copy()


def ids_to_blob(ids) -> bytes:
    """Serialize danh sách image_id thành raw int64 bytes."""
    return np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes()


def blob_to_ids(blob: bytes) -> np.ndarray:
    """Giải nén raw bytes thành mảng int64 image_id."""
    return np.frombuffer(blob, dtype=np.int64).copy()


def _json_default(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


def serialize_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def deserialize_json(data: Optional[str]) -> dict:
    if not data:
        return {}
    return json.loads(data)


def parse_label_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    if '_' in stem:
        return stem.split('_')[0]
    return stem




def parse_label_from_path(file_path: str | Path, dataset_root: str | Path | None = None, mode: str = 'auto') -> str:
    """Lấy nhãn ảnh theo cấu trúc dataset.

    mode:
    - 'parent_folder': nhãn là tên thư mục cha của ảnh.
    - 'filename_prefix': nhãn là phần trước dấu '_' trong tên file.
    - 'auto': nếu ảnh nằm trong thư mục con của dataset_root thì lấy thư mục cha,
      ngược lại fallback về filename_prefix.
    """
    path = Path(file_path)
    mode = (mode or 'auto').lower()

    if mode == 'filename_prefix':
        return parse_label_from_filename(path.name)

    if mode == 'parent_folder':
        return path.parent.name or parse_label_from_filename(path.name)

    # auto
    if dataset_root is not None:
        try:
            root = Path(dataset_root).resolve()
            resolved = path.resolve()
            rel = resolved.relative_to(root)
            if len(rel.parts) >= 2:
                return rel.parts[0]
        except Exception:
            pass
    if path.parent.name and dataset_root is not None:
        try:
            if path.parent.resolve() != Path(dataset_root).resolve():
                return path.parent.name
        except Exception:
            pass
    return parse_label_from_filename(path.name)

def read_image_rgb(path: str) -> np.ndarray:
    image = Image.open(path)
    return np.array(image.convert('RGB'))


def read_image_rgba(path: str) -> np.ndarray:
    image = Image.open(path)
    return np.array(image.convert('RGBA'))
