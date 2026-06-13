from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flower_cbir_app.utils.common import (
    blob_to_np,
    deserialize_json,
    serialize_json,
)

SCHEMA_VERSION = 7


def _arr_to_json(arr, decimals: int = 6) -> str:
    """Tuần tự hóa numpy array thành chuỗi JSON đọc được (list/list-of-list).

    Làm tròn `decimals` chữ số để xem gọn và không lộ sai số float32 khi mở DB.
    """
    a = np.round(np.asarray(arr, dtype=np.float64), decimals)
    return json.dumps(a.tolist())


def _json_to_arr(text: str, dtype=np.float32) -> np.ndarray:
    """Giải mã chuỗi JSON về numpy array với dtype mong muốn."""
    return np.asarray(json.loads(text), dtype=dtype)

# Dữ liệu được reset mỗi lần chạy lại (chỉ giữ 1 bộ tại một thời điểm), nên không
# còn bảng *_runs. Ta dùng một run_id cố định cho cache + cho các API cũ vẫn
# nhận tham số run_id (giữ tương thích chữ ký hàm cho app.py/pipeline).
_RUN_ID = 1

# ── Cache ma trận feature ở mức module ───────────────────────────────────────
# Ma trận feature là immutable cho tới khi reset/trích xuất lại. Vì Streamlit tạo
# SQLiteManager mới ở mỗi rerun, ta cache ở mức module theo (db_path, run_id,
# feature_key) để query online không phải đọc lại DB và giải nén blob mỗi lần.
_MATRIX_CACHE: Dict[Tuple[str, int, str], Tuple[np.ndarray, np.ndarray]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(db_path: str, run_id: int, feature_key: str):
    """Lấy (image_ids, matrix) đã cache cho 1 feature; None nếu chưa cache."""
    with _CACHE_LOCK:
        return _MATRIX_CACHE.get((db_path, run_id, feature_key))


def _cache_put(db_path: str, run_id: int, feature_key: str, image_ids: np.ndarray, matrix: np.ndarray):
    """Lưu ma trận feature vào cache mức module (sống qua các lần Streamlit rerun)."""
    with _CACHE_LOCK:
        _MATRIX_CACHE[(db_path, run_id, feature_key)] = (image_ids, matrix)


def _cache_clear(db_path: str):
    """Xóa toàn bộ cache của 1 database (gọi khi reset preprocess/extraction)."""
    with _CACHE_LOCK:
        for k in [key for key in _MATRIX_CACHE if key[0] == db_path]:
            _MATRIX_CACHE.pop(k, None)


class SQLiteManager:
    """Lớp truy cập SQLite — toàn bộ thao tác DB của project (SQL viết tay, KHÔNG ORM).

    Schema rút gọn còn 3 bảng:
      - images: ảnh gốc + nhãn.
      - preprocess_outputs: đường dẫn ảnh đã tiền xử lý của mỗi ảnh (FK → images, 1:1).
      - feature_matrices: GỘP ma trận N×D của mỗi feature + toàn bộ cấu hình/tham
        số chuẩn hóa (mean/std/vocab/d_min/d_max/distance/weight...) trong cùng 1
        dòng/feature. feature_key là khóa chính.

    Kết quả đánh giá KHÔNG lưu DB nữa — chỉ tính & hiển thị tại chỗ trong phiên.

    Mỗi lần Streamlit rerun lại tạo 1 instance mới nên kết nối là ngắn hạn; ma
    trận feature được cache ở mức module (_MATRIX_CACHE) để khỏi đọc blob lại.
    """

    def __init__(self, db_path: str):
        """Mở/khởi tạo file SQLite tại db_path (tạo thư mục cha + bảng nếu chưa có)."""
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        """Tạo kết nối SQLite mới: row trả về dạng Row (truy cập theo tên cột) + bật FK."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')  # SQLite mặc định TẮT ràng buộc FK
        return conn

    def _init_db(self):
        """Tạo 3 bảng nếu chưa có. Migrate sạch schema cũ bằng user_version."""
        with self._connect() as conn:
            ver = int(conn.execute('PRAGMA user_version').fetchone()[0])
            if ver < SCHEMA_VERSION:
                # Bỏ các bảng cũ không còn dùng / đã đổi cấu trúc. Dữ liệu cũ sẽ
                # được sinh lại khi chạy lại tiền xử lý + trích xuất.
                conn.executescript(
                    """
                    DROP TABLE IF EXISTS meta;
                    DROP TABLE IF EXISTS preprocess_runs;
                    DROP TABLE IF EXISTS extraction_runs;
                    DROP TABLE IF EXISTS feature_configs;
                    DROP TABLE IF EXISTS feature_matrices;
                    DROP TABLE IF EXISTS preprocess_outputs;
                    DROP TABLE IF EXISTS evaluations;
                    DROP TABLE IF EXISTS evaluation_runs;
                    DROP TABLE IF EXISTS feature_vectors;
                    """
                )
                conn.execute(f'PRAGMA user_version = {SCHEMA_VERSION}')

            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    file_name TEXT,
                    label TEXT
                );

                -- Mỗi ảnh có ĐÚNG 1 bộ ảnh tiền xử lý (quan hệ 1:1). image_id là
                -- UNIQUE để DB ép buộc: 1 ảnh không thể có 2 dòng output. FK → images.
                CREATE TABLE IF NOT EXISTS preprocess_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER UNIQUE,
                    normalized_path TEXT,
                    mask_path TEXT,
                    gray_path TEXT,
                    edge_path TEXT,
                    debug_json TEXT,
                    FOREIGN KEY(image_id) REFERENCES images(image_id)
                );

                -- GỘP ma trận N×D của mỗi feature + toàn bộ cấu hình/tham số chuẩn
                -- hóa trong cùng 1 dòng. feature_key là khóa chính (mỗi feature 1
                -- dòng). Các cột *_json lưu dạng TEXT (JSON) để mở bằng trình xem
                -- SQLite là thấy số trực tiếp. image_ids_json nối các hàng của ma
                -- trận với images.image_id.
                CREATE TABLE IF NOT EXISTS feature_matrices (
                    feature_key TEXT PRIMARY KEY,
                    group_name TEXT,
                    enabled INTEGER,
                    distance_type TEXT,
                    weight REAL,
                    is_meta INTEGER,
                    d_min REAL,
                    d_max REAL,
                    mean_json TEXT,
                    std_json TEXT,
                    vocab_json TEXT,
                    extra_json TEXT,
                    num_rows INTEGER,
                    dim INTEGER,
                    matrix_json TEXT,
                    image_ids_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_po_image ON preprocess_outputs(image_id);
                """
            )

    # ── Schema version ────────────────────────────────────────────────────────
    def get_schema_version(self) -> int:
        """Đọc số phiên bản schema (PRAGMA user_version)."""
        with self._connect() as conn:
            return int(conn.execute('PRAGMA user_version').fetchone()[0])

    # ── Runs (giữ API cũ, không còn bảng *_runs) ──────────────────────────────
    def create_preprocess_run(self, config: dict, summary: dict | None = None) -> int:
        """Tương thích API cũ: không còn bảng runs, trả về run_id cố định."""
        return _RUN_ID

    def create_extraction_run(self, config: dict, summary: dict | None = None) -> int:
        """Tương thích API cũ: không còn bảng runs, trả về run_id cố định."""
        return _RUN_ID

    def update_extraction_run_summary(self, run_id: int, summary: dict):
        """Tương thích API cũ: không lưu summary run (đã bỏ bảng extraction_runs)."""
        return None

    def get_extraction_run_config(self, run_id: int) -> dict:
        """Tương thích API cũ: không còn lưu config theo run.

        Online/đánh giá sẽ tự fallback sang cấu hình hiện hành được truyền vào.
        """
        return {}

    def reset_preprocess_data(self):
        """Xóa toàn bộ dữ liệu preprocess VÀ feature (preprocess đổi thì feature cũ vô hiệu).
        Xóa luôn images để tránh tích lũy ảnh từ nhiều dataset_root khác nhau."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM feature_matrices;
                DELETE FROM preprocess_outputs;
                DELETE FROM images;
            """)
        _cache_clear(self.db_path)

    def reset_extraction_data(self):
        """Xóa toàn bộ dữ liệu feature (giữ lại preprocess)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM feature_matrices")
        _cache_clear(self.db_path)

    # ── Images ────────────────────────────────────────────────────────────────
    def upsert_image(self, file_path: str, file_name: str, label: str) -> int:
        """Thêm ảnh vào bảng images, hoặc cập nhật nếu file_path đã có. Trả image_id.

        file_path là khóa UNIQUE nên cùng 1 ảnh không bị tạo nhiều bản ghi.
        """
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO images(file_path, file_name, label)
                VALUES (?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET file_name = excluded.file_name, label = excluded.label
                ''',
                (file_path, file_name, label),
            )
            row = conn.execute('SELECT image_id FROM images WHERE file_path = ?', (file_path,)).fetchone()
            return int(row['image_id'])

    def insert_preprocess_output(self, preprocess_run_id: int, image_id: int, normalized_path: str, mask_path: str, gray_path: str, edge_path: str, debug_json: dict):
        """Lưu ĐƯỜNG DẪN 4 ảnh đã tiền xử lý của 1 ảnh (ảnh nằm trên đĩa, DB chỉ giữ path).

        UPSERT theo image_id (quan hệ 1:1): chạy lại tiền xử lý cùng ảnh thì ghi đè,
        không tạo dòng thứ 2. preprocess_run_id giữ trong chữ ký cho tương thích.
        """
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO preprocess_outputs(image_id, normalized_path, mask_path, gray_path, edge_path, debug_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    normalized_path = excluded.normalized_path,
                    mask_path = excluded.mask_path,
                    gray_path = excluded.gray_path,
                    edge_path = excluded.edge_path,
                    debug_json = excluded.debug_json
                ''',
                (image_id, normalized_path, mask_path, gray_path, edge_path, serialize_json(debug_json)),
            )

    def get_preprocess_outputs_map(self, preprocess_run_id: int | None = None) -> Dict[int, dict]:
        """Map image_id → đường dẫn ảnh trung gian, để extraction tái dùng không phải tiền xử lý lại."""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT image_id, normalized_path, mask_path, gray_path, edge_path FROM preprocess_outputs',
            ).fetchall()
        return {
            int(r['image_id']): {
                'normalized_path': r['normalized_path'],
                'mask_path': r['mask_path'],
                'gray_path': r['gray_path'],
                'edge_path': r['edge_path'],
            }
            for r in rows
        }

    # ── Feature config (gộp vào feature_matrices) ─────────────────────────────
    def insert_feature_config(self, extraction_run_id: int, feature_key: str, group_name: str, enabled: bool,
                              distance_type: str, weight: float, is_meta: bool, mean, std, vocab,
                              d_min: float = 0.0, d_max: float = 1.0, extra: dict | None = None):
        """Lưu cấu hình + tham số chuẩn hóa của 1 feature vào dòng feature_matrices của nó.

        Gồm distance_type, weight, is_meta, thang đo cố định (d_min/d_max) và các
        blob mean/std (z-score), vocab (BoVW) để online tái dùng đúng tham số.
        UPSERT theo feature_key: phần ma trận do insert_feature_matrix ghi sau.
        """
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO feature_matrices(feature_key, group_name, enabled, distance_type, weight, is_meta, d_min, d_max, mean_json, std_json, vocab_json, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(feature_key) DO UPDATE SET
                    group_name = excluded.group_name,
                    enabled = excluded.enabled,
                    distance_type = excluded.distance_type,
                    weight = excluded.weight,
                    is_meta = excluded.is_meta,
                    d_min = excluded.d_min,
                    d_max = excluded.d_max,
                    mean_json = excluded.mean_json,
                    std_json = excluded.std_json,
                    vocab_json = excluded.vocab_json,
                    extra_json = excluded.extra_json
                ''',
                (
                    feature_key,
                    group_name,
                    int(enabled),
                    distance_type,
                    float(weight),
                    int(is_meta),
                    float(d_min),
                    float(d_max),
                    _arr_to_json(mean),
                    _arr_to_json(std),
                    _arr_to_json(vocab) if vocab is not None else None,
                    serialize_json(extra or {}),
                ),
            )

    def get_extraction_feature_configs(self, extraction_run_id: int) -> Dict[str, dict]:
        """Đọc config mọi feature, trả dict {feature_key: {...}}.

        Tự giải nén các blob mean/std/vocab về numpy. Đây là dữ liệu fusion/online
        cần để chuẩn hóa và chọn distance.
        """
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT feature_key, group_name, enabled, distance_type, weight, is_meta, d_min, d_max, mean_json, std_json, vocab_json, extra_json FROM feature_matrices'
            ).fetchall()
        out = {}
        for row in rows:
            if row['distance_type'] is None:
                continue  # dòng chưa có config (chỉ mới có ma trận)
            out[row['feature_key']] = {
                'feature_key': row['feature_key'],
                'group_name': row['group_name'],
                'enabled': bool(row['enabled']),
                'distance_type': row['distance_type'],
                'weight': float(row['weight']),
                'is_meta': bool(row['is_meta']),
                'd_min': float(row['d_min']) if row['d_min'] is not None else 0.0,
                'd_max': float(row['d_max']) if row['d_max'] is not None else 1.0,
                'mean': _json_to_arr(row['mean_json']) if row['mean_json'] is not None else None,
                'std': _json_to_arr(row['std_json']) if row['std_json'] is not None else None,
                'vocab': _json_to_arr(row['vocab_json']) if row['vocab_json'] is not None else None,
                'extra': deserialize_json(row['extra_json']),
            }
        return out

    # ── Feature matrices (gộp) ────────────────────────────────────────────────
    def insert_feature_matrix(self, extraction_run_id: int, feature_key: str, image_ids, matrix: np.ndarray):
        """Lưu GỘP ma trận N×D của một feature + mảng image_id vào dòng feature_matrices.

        UPSERT theo feature_key: phần config do insert_feature_config ghi trước.
        """
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        num_rows, dim = int(matrix.shape[0]), int(matrix.shape[1])
        image_ids_arr = np.asarray(image_ids, dtype=np.int64)
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO feature_matrices(feature_key, num_rows, dim, matrix_json, image_ids_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(feature_key) DO UPDATE SET
                    num_rows = excluded.num_rows,
                    dim = excluded.dim,
                    matrix_json = excluded.matrix_json,
                    image_ids_json = excluded.image_ids_json
                ''',
                (feature_key, num_rows, dim, _arr_to_json(matrix), json.dumps(image_ids_arr.tolist())),
            )
        _cache_put(self.db_path, _RUN_ID, feature_key, image_ids_arr, matrix)

    def get_feature_matrix_raw(self, extraction_run_id: int, feature_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (image_ids int64 [N], matrix float32 [N×D]). Có cache ở mức module."""
        cached = _cache_get(self.db_path, _RUN_ID, feature_key)
        if cached is not None:
            return cached
        with self._connect() as conn:
            row = conn.execute(
                'SELECT num_rows, dim, matrix_json, image_ids_json FROM feature_matrices WHERE feature_key = ?',
                (feature_key,),
            ).fetchone()
        if row is None or row['matrix_json'] is None:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        matrix = _json_to_arr(row['matrix_json'], dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(int(row['num_rows']), int(row['dim']))
        image_ids = _json_to_arr(row['image_ids_json'], dtype=np.int64)
        _cache_put(self.db_path, _RUN_ID, feature_key, image_ids, matrix)
        return image_ids, matrix

    def _images_lookup(self) -> Dict[int, dict]:
        """Trả map image_id -> {file_path, file_name, label} cho toàn bộ ảnh (1 truy vấn)."""
        with self._connect() as conn:
            rows = conn.execute('SELECT image_id, file_path, file_name, label FROM images').fetchall()
        return {int(r['image_id']): {'file_path': r['file_path'], 'file_name': r['file_name'], 'label': r['label']} for r in rows}

    def get_feature_matrix(self, extraction_run_id: int, feature_key: str) -> pd.DataFrame:
        """Tương thích ngược: dựng lại DataFrame (image_id, file_path, file_name, label, vector, dim)
        từ ma trận gộp + bảng images. Giữ nguyên contract cho evaluation/online cũ."""
        image_ids, matrix = self.get_feature_matrix_raw(extraction_run_id, feature_key)
        if len(image_ids) == 0:
            return pd.DataFrame(columns=['image_id', 'file_path', 'file_name', 'label', 'vector', 'dim', 'extra'])
        lookup = self._images_lookup()
        dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        data = []
        for idx, image_id in enumerate(image_ids):
            meta = lookup.get(int(image_id), {'file_path': '', 'file_name': '', 'label': ''})
            data.append({
                'image_id': int(image_id),
                'file_path': meta['file_path'],
                'file_name': meta['file_name'],
                'label': meta['label'],
                'vector': matrix[idx],
                'dim': dim,
                'extra': {},
            })
        return pd.DataFrame(data)

    def get_all_feature_matrices(self, extraction_run_id: int) -> Dict[str, pd.DataFrame]:
        """Trả về toàn bộ feature matrix dưới dạng {feature_key: DataFrame}."""
        configs = self.get_extraction_feature_configs(extraction_run_id)
        return {k: self.get_feature_matrix(extraction_run_id, k) for k in configs.keys()}

    def get_feature_matrices_summary(self, extraction_run_id: int) -> List[dict]:
        """Tóm tắt các feature matrix đã lưu: feature_key, num_rows, dim.

        Dùng cho UI product để hiển thị 'dữ liệu đã trích xuất trong DB' mà không
        phải tải toàn bộ blob ma trận về.
        """
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT feature_key, num_rows, dim FROM feature_matrices WHERE matrix_json IS NOT NULL ORDER BY feature_key',
            ).fetchall()
        return [
            {'feature_key': r['feature_key'], 'num_rows': int(r['num_rows']), 'dim': int(r['dim'])}
            for r in rows
        ]

    def get_features_per_image_table(self, extraction_run_id: int, feature_keys: List[str],
                                     limit: int | None = None, precision: int = 4) -> pd.DataFrame:
        """Dựng bảng ảnh × feature: mỗi HÀNG là 1 ảnh, mỗi CỘT là 1 feature,
        giá trị ô = vector feature của ảnh đó (rút gọn dạng chuỗi).

        limit=None lấy toàn bộ ảnh. Vector được làm tròn `precision` chữ số và
        cắt bớt nếu quá dài để hiển thị gọn trên UI.
        """
        # Đọc ma trận của từng feature (image_ids đồng nhất giữa các feature
        # vì cùng thứ tự lúc extract).
        feature_data = {}
        image_ids_ref = None
        for key in feature_keys:
            ids, matrix = self.get_feature_matrix_raw(extraction_run_id, key)
            if len(ids) == 0:
                continue
            feature_data[key] = (ids, matrix)
            if image_ids_ref is None:
                image_ids_ref = ids
        if image_ids_ref is None:
            return pd.DataFrame()

        lookup = self._images_lookup()
        n = len(image_ids_ref) if limit is None else min(limit, len(image_ids_ref))

        def _fmt(vec) -> str:
            arr = np.round(np.asarray(vec, dtype=np.float32), precision)
            return '[' + ', '.join(str(v) for v in arr.tolist()) + ']'

        rows = []
        for i in range(n):
            image_id = int(image_ids_ref[i])
            meta = lookup.get(image_id, {'file_name': '', 'label': ''})
            row = {'image_id': image_id, 'file_name': meta['file_name'], 'label': meta['label']}
            for key in feature_keys:
                if key not in feature_data:
                    continue
                ids, matrix = feature_data[key]
                row[key] = _fmt(matrix[i]) if i < len(matrix) else ''
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Latest run ids (suy ra từ dữ liệu hiện có) ────────────────────────────
    def get_latest_preprocess_run_id(self) -> Optional[int]:
        """Trả run_id cố định nếu đã có dữ liệu tiền xử lý, None nếu chưa."""
        with self._connect() as conn:
            row = conn.execute('SELECT 1 FROM preprocess_outputs LIMIT 1').fetchone()
        return _RUN_ID if row else None

    def get_latest_extraction_run_id(self) -> Optional[int]:
        """Trả run_id cố định nếu đã có ma trận feature, None nếu chưa. Online/đánh giá dùng run này."""
        with self._connect() as conn:
            row = conn.execute('SELECT 1 FROM feature_matrices WHERE matrix_json IS NOT NULL LIMIT 1').fetchone()
        return _RUN_ID if row else None

    def list_images(self, limit: int = 100) -> List[dict]:
        """Liệt kê các ảnh mới nhất trong bảng images (tối đa `limit` dòng)."""
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM images ORDER BY image_id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]

    # ── Dump bảng thô (cho UI xem DB) ─────────────────────────────────────────
    def list_tables(self) -> List[str]:
        """Liệt kê tên các bảng dữ liệu trong DB (bỏ bảng hệ thống sqlite_*)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r['name'] for r in rows]

    def dump_table(self, table: str, limit: int = 500) -> pd.DataFrame:
        """Đọc toàn bộ 1 bảng ra DataFrame, CHUYỂN cột blob/json về dạng đọc được.

        - Cột *_blob (BLOB nhị phân): hiển thị tóm tắt 'numpy[shape] | head: ...'.
        - Cột *_json: giữ nguyên chuỗi JSON (đã là text đọc được).
        Chỉ cho phép tên bảng có trong list_tables để tránh injection.
        """
        if table not in self.list_tables():
            return pd.DataFrame()
        with self._connect() as conn:
            rows = conn.execute(f'SELECT * FROM "{table}" LIMIT ?', (limit,)).fetchall()
        if not rows:
            return pd.DataFrame()

        def _blob_repr(b: bytes) -> str:
            try:
                arr = blob_to_np(b)
                flat = arr.ravel()
                vals = ', '.join(str(x) for x in np.round(flat, 4).tolist())
                return f'numpy{list(arr.shape)} [{vals}]'
            except Exception:
                return f'<blob {len(b)} bytes>'

        data = []
        for row in rows:
            d = {}
            for k in row.keys():
                v = row[k]
                if isinstance(v, (bytes, bytearray)):
                    d[k] = _blob_repr(bytes(v))
                else:
                    d[k] = v
            data.append(d)
        return pd.DataFrame(data)

    def dump_feature_matrices_readable(self, hidden_keys=None, order_keys=None, limit_rows: int = 500) -> pd.DataFrame:
        """Dump bảng feature_matrices nhưng chuyển blob ma trận thành mô tả đọc được,
        ẩn các feature trong hidden_keys, và sắp theo order_keys (thứ tự registry).

        Cột matrix_json -> mảng của mảng (ma trận N×D); image_ids_json -> 'N ids: ...'.
        """
        hidden_keys = set(hidden_keys or [])
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT feature_key, num_rows, dim, matrix_json, image_ids_json FROM feature_matrices WHERE matrix_json IS NOT NULL LIMIT ?',
                (limit_rows,),
            ).fetchall()
        data = []
        for r in rows:
            key = r['feature_key']
            if key in hidden_keys:
                continue
            # matrix_json/image_ids_json đã là TEXT đọc được; hiển thị trực tiếp.
            mat_repr = r['matrix_json'] if r['matrix_json'] is not None else '[]'
            try:
                ids = json.loads(r['image_ids_json'])
                ids_repr = f'{len(ids)} ids: {ids}'
            except Exception:
                ids_repr = '[]'
            data.append({
                'feature_key': key,
                'num_rows': int(r['num_rows']),
                'dim': int(r['dim']),
                'matrix_json': mat_repr,
                'image_ids_json': ids_repr,
            })
        df = pd.DataFrame(data)
        if order_keys and not df.empty:
            order_map = {k: i for i, k in enumerate(order_keys)}
            df['_o'] = df['feature_key'].map(lambda k: order_map.get(k, 999))
            df = df.sort_values('_o').drop(columns='_o').reset_index(drop=True)
        return df
