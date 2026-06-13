from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flower_cbir_app.utils.common import (
    array_to_raw_blob,
    blob_to_ids,
    blob_to_np,
    deserialize_json,
    ids_to_blob,
    np_to_blob,
    raw_blob_to_matrix,
    serialize_json,
)

SCHEMA_VERSION = 3

# ── Cache ma trận feature ở mức module ───────────────────────────────────────
# Ma trận feature là immutable theo từng extraction_run_id (run append-only,
# reset sẽ tạo run_id mới). Vì Streamlit tạo SQLiteManager mới ở mỗi rerun, ta
# cache ở mức module theo (db_path, run_id, feature_key) để query online không
# phải đọc lại DB và giải nén blob mỗi lần.
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

    Quản lý 8 bảng: meta, preprocess_runs, extraction_runs, images,
    preprocess_outputs, feature_configs, feature_matrices, evaluations.
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
        """Tạo toàn bộ bảng + index nếu chưa tồn tại và ghi schema_version vào meta."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS preprocess_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT,
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS extraction_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    config_json TEXT,
                    summary_json TEXT
                );

                CREATE TABLE IF NOT EXISTS images (
                    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    file_name TEXT,
                    label TEXT
                );

                CREATE TABLE IF NOT EXISTS preprocess_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    preprocess_run_id INTEGER,
                    image_id INTEGER,
                    normalized_path TEXT,
                    mask_path TEXT,
                    gray_path TEXT,
                    edge_path TEXT,
                    debug_json TEXT,
                    FOREIGN KEY(image_id) REFERENCES images(image_id)
                );

                CREATE TABLE IF NOT EXISTS feature_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_run_id INTEGER,
                    feature_key TEXT,
                    group_name TEXT,
                    enabled INTEGER,
                    distance_type TEXT,
                    weight REAL,
                    is_meta INTEGER,
                    d_min REAL,
                    d_max REAL,
                    mean_blob BLOB,
                    std_blob BLOB,
                    vocab_blob BLOB,
                    extra_json TEXT
                );

                -- Lưu GỘP toàn bộ vector của một feature trong một extraction run
                -- thành MỘT ma trận (N×D) duy nhất + mảng image_id tương ứng.
                -- Thay cho thiết kế cũ 1 dòng/ảnh/feature (hàng nghìn dòng rời rạc).
                CREATE TABLE IF NOT EXISTS feature_matrices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_run_id INTEGER,
                    feature_key TEXT,
                    num_rows INTEGER,
                    dim INTEGER,
                    matrix_blob BLOB,
                    image_ids_blob BLOB
                );

                CREATE INDEX IF NOT EXISTS idx_fm_run_key ON feature_matrices(extraction_run_id, feature_key);
                CREATE INDEX IF NOT EXISTS idx_fc_run ON feature_configs(extraction_run_id);
                CREATE INDEX IF NOT EXISTS idx_po_run ON preprocess_outputs(preprocess_run_id);
                CREATE INDEX IF NOT EXISTS idx_po_image ON preprocess_outputs(image_id);

                -- Lưu KẾT QUẢ đánh giá để mở lại xem lần gần nhất đã chạy.
                -- Chỉ giữ kết quả mới nhất (ghi đè), không lưu lịch sử.
                CREATE TABLE IF NOT EXISTS evaluations (
                    extraction_run_id INTEGER PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metrics_json TEXT,
                    separation_json TEXT
                );

                -- Dọn bảng cũ không còn dùng (migrate từ schema cũ).
                DROP TABLE IF EXISTS evaluation_runs;
                DROP TABLE IF EXISTS feature_vectors;
                """
            )
            conn.execute(
                'INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
                ('schema_version', str(SCHEMA_VERSION)),
            )

    # ── Schema version ────────────────────────────────────────────────────────
    def get_schema_version(self) -> int:
        """Đọc số phiên bản schema đang lưu trong bảng meta (0 nếu chưa có)."""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return int(row['value']) if row else 0

    # ── Runs ────────────────────────────────────────────────────────────────
    def create_preprocess_run(self, config: dict, summary: dict | None = None) -> int:
        """Tạo 1 bản ghi preprocess_run mới (lưu config), trả về run_id vừa sinh."""
        with self._connect() as conn:
            cur = conn.execute(
                'INSERT INTO preprocess_runs(config_json, summary_json) VALUES (?, ?)',
                (serialize_json(config), serialize_json(summary or {})),
            )
            return int(cur.lastrowid)

    def create_extraction_run(self, config: dict, summary: dict | None = None) -> int:
        """Tạo 1 bản ghi extraction_run mới (lưu config), trả về run_id vừa sinh."""
        with self._connect() as conn:
            cur = conn.execute(
                'INSERT INTO extraction_runs(config_json, summary_json) VALUES (?, ?)',
                (serialize_json(config), serialize_json(summary or {})),
            )
            return int(cur.lastrowid)

    def update_extraction_run_summary(self, run_id: int, summary: dict):
        """Cập nhật phần summary_json (số ảnh, feature...) cho 1 extraction run."""
        with self._connect() as conn:
            conn.execute('UPDATE extraction_runs SET summary_json = ? WHERE run_id = ?', (serialize_json(summary), run_id))

    def get_extraction_run_config(self, run_id: int) -> dict:
        """Đọc lại config (system + features) đã lưu của 1 extraction run."""
        with self._connect() as conn:
            row = conn.execute('SELECT config_json FROM extraction_runs WHERE run_id = ?', (run_id,)).fetchone()
        return deserialize_json(row['config_json']) if row else {}

    def reset_preprocess_data(self):
        """Xóa toàn bộ dữ liệu preprocess VÀ extraction (preprocess đổi thì extraction cũ vô hiệu).
        Xóa luôn images để tránh tích lũy ảnh từ nhiều dataset_root khác nhau."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM evaluations;
                DELETE FROM feature_matrices;
                DELETE FROM feature_configs;
                DELETE FROM extraction_runs;
                DELETE FROM preprocess_outputs;
                DELETE FROM preprocess_runs;
                DELETE FROM images;
            """)
        _cache_clear(self.db_path)

    def reset_extraction_data(self):
        """Xóa toàn bộ dữ liệu extraction (giữ lại preprocess)."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM evaluations;
                DELETE FROM feature_matrices;
                DELETE FROM feature_configs;
                DELETE FROM extraction_runs;
            """)
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
        """Lưu ĐƯỜNG DẪN 4 ảnh đã tiền xử lý của 1 ảnh (ảnh nằm trên đĩa, DB chỉ giữ path)."""
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO preprocess_outputs(preprocess_run_id, image_id, normalized_path, mask_path, gray_path, edge_path, debug_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (preprocess_run_id, image_id, normalized_path, mask_path, gray_path, edge_path, serialize_json(debug_json)),
            )

    def get_preprocess_outputs_map(self, preprocess_run_id: int | None = None) -> Dict[int, dict]:
        """Map image_id → đường dẫn ảnh trung gian, để extraction tái dùng không phải tiền xử lý lại."""
        with self._connect() as conn:
            if preprocess_run_id is None:
                row = conn.execute('SELECT run_id FROM preprocess_runs ORDER BY run_id DESC LIMIT 1').fetchone()
                if row is None:
                    return {}
                preprocess_run_id = int(row['run_id'])
            rows = conn.execute(
                'SELECT image_id, normalized_path, mask_path, gray_path, edge_path FROM preprocess_outputs WHERE preprocess_run_id = ?',
                (preprocess_run_id,),
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

    # ── Feature configs ─────────────────────────────────────────────────────
    def insert_feature_config(self, extraction_run_id: int, feature_key: str, group_name: str, enabled: bool,
                              distance_type: str, weight: float, is_meta: bool, mean, std, vocab,
                              d_min: float = 0.0, d_max: float = 1.0, extra: dict | None = None):
        """Lưu cấu hình + tham số chuẩn hóa của 1 feature trong 1 extraction run.

        Gồm distance_type, weight, is_meta, thang đo cố định (d_min/d_max) và các
        blob mean/std (z-score), vocab (BoVW) để online tái dùng đúng tham số.
        """
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO feature_configs(extraction_run_id, feature_key, group_name, enabled, distance_type, weight, is_meta, d_min, d_max, mean_blob, std_blob, vocab_blob, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    extraction_run_id,
                    feature_key,
                    group_name,
                    int(enabled),
                    distance_type,
                    float(weight),
                    int(is_meta),
                    float(d_min),
                    float(d_max),
                    np_to_blob(mean),
                    np_to_blob(std),
                    np_to_blob(vocab) if vocab is not None else None,
                    serialize_json(extra or {}),
                ),
            )

    def get_extraction_feature_configs(self, extraction_run_id: int) -> Dict[str, dict]:
        """Đọc config mọi feature của 1 run, trả dict {feature_key: {...}}.

        Tự giải nén các blob mean/std/vocab về numpy. Đây là dữ liệu fusion/online
        cần để chuẩn hóa và chọn distance.
        """
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM feature_configs WHERE extraction_run_id = ?', (extraction_run_id,)).fetchall()
        out = {}
        for row in rows:
            keys = row.keys()
            out[row['feature_key']] = {
                'feature_key': row['feature_key'],
                'group_name': row['group_name'],
                'enabled': bool(row['enabled']),
                'distance_type': row['distance_type'],
                'weight': float(row['weight']),
                'is_meta': bool(row['is_meta']),
                'd_min': float(row['d_min']) if 'd_min' in keys and row['d_min'] is not None else 0.0,
                'd_max': float(row['d_max']) if 'd_max' in keys and row['d_max'] is not None else 1.0,
                'mean': blob_to_np(row['mean_blob']) if row['mean_blob'] is not None else None,
                'std': blob_to_np(row['std_blob']) if row['std_blob'] is not None else None,
                'vocab': blob_to_np(row['vocab_blob']) if row['vocab_blob'] is not None else None,
                'extra': deserialize_json(row['extra_json']),
            }
        return out

    # ── Feature matrices (gộp) ────────────────────────────────────────────────
    def insert_feature_matrix(self, extraction_run_id: int, feature_key: str, image_ids, matrix: np.ndarray):
        """Lưu GỘP ma trận N×D của một feature + mảng image_id thành một dòng blob."""
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        num_rows, dim = int(matrix.shape[0]), int(matrix.shape[1])
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO feature_matrices(extraction_run_id, feature_key, num_rows, dim, matrix_blob, image_ids_blob)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (extraction_run_id, feature_key, num_rows, dim, array_to_raw_blob(matrix), ids_to_blob(image_ids)),
            )
        _cache_put(self.db_path, extraction_run_id, feature_key,
                   np.asarray(image_ids, dtype=np.int64), matrix)

    def get_feature_matrix_raw(self, extraction_run_id: int, feature_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (image_ids int64 [N], matrix float32 [N×D]). Có cache ở mức module."""
        cached = _cache_get(self.db_path, extraction_run_id, feature_key)
        if cached is not None:
            return cached
        with self._connect() as conn:
            row = conn.execute(
                'SELECT num_rows, dim, matrix_blob, image_ids_blob FROM feature_matrices WHERE extraction_run_id = ? AND feature_key = ?',
                (extraction_run_id, feature_key),
            ).fetchone()
        if row is None:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        matrix = raw_blob_to_matrix(row['matrix_blob'], int(row['num_rows']), int(row['dim']))
        image_ids = blob_to_ids(row['image_ids_blob'])
        _cache_put(self.db_path, extraction_run_id, feature_key, image_ids, matrix)
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
        """Trả về toàn bộ feature matrix của 1 run dưới dạng {feature_key: DataFrame}."""
        configs = self.get_extraction_feature_configs(extraction_run_id)
        return {k: self.get_feature_matrix(extraction_run_id, k) for k in configs.keys()}

    def get_feature_matrices_summary(self, extraction_run_id: int) -> List[dict]:
        """Tóm tắt các feature matrix đã lưu của 1 run: feature_key, num_rows, dim.

        Dùng cho UI product để hiển thị 'dữ liệu đã trích xuất trong DB' mà không
        phải tải toàn bộ blob ma trận về.
        """
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT feature_key, num_rows, dim FROM feature_matrices WHERE extraction_run_id = ? ORDER BY feature_key',
                (extraction_run_id,),
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

    # ── Latest run ids ────────────────────────────────────────────────────────
    def get_latest_preprocess_run_id(self) -> Optional[int]:
        """Lấy run_id của lần tiền xử lý mới nhất (None nếu chưa có)."""
        with self._connect() as conn:
            row = conn.execute('SELECT run_id FROM preprocess_runs ORDER BY run_id DESC LIMIT 1').fetchone()
            return int(row['run_id']) if row else None

    def get_latest_extraction_run_id(self) -> Optional[int]:
        """Lấy run_id của lần trích xuất mới nhất (None nếu chưa có). Online/đánh giá dùng run này."""
        with self._connect() as conn:
            row = conn.execute('SELECT run_id FROM extraction_runs ORDER BY run_id DESC LIMIT 1').fetchone()
            return int(row['run_id']) if row else None

    def list_images(self, limit: int = 100) -> List[dict]:
        """Liệt kê các ảnh mới nhất trong bảng images (tối đa `limit` dòng)."""
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM images ORDER BY image_id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]

    # ── Evaluations (lưu kết quả đánh giá, chỉ giữ bản mới nhất / run) ─────────
    def save_evaluation(self, extraction_run_id: int, metrics: dict, separation: dict):
        """Lưu (ghi đè) kết quả đánh giá của 1 extraction run.

        Dùng UPSERT theo extraction_run_id: mỗi run chỉ giữ kết quả mới nhất,
        không tích lũy lịch sử. Mở lại app vẫn xem được đánh giá lần gần nhất.
        """
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO evaluations(extraction_run_id, created_at, metrics_json, separation_json)
                VALUES (?, CURRENT_TIMESTAMP, ?, ?)
                ON CONFLICT(extraction_run_id) DO UPDATE SET
                    created_at = CURRENT_TIMESTAMP,
                    metrics_json = excluded.metrics_json,
                    separation_json = excluded.separation_json
                ''',
                (extraction_run_id, serialize_json(metrics), serialize_json(separation)),
            )

    def get_evaluation(self, extraction_run_id: int) -> Optional[dict]:
        """Đọc kết quả đánh giá đã lưu của 1 run; None nếu chưa từng đánh giá."""
        with self._connect() as conn:
            row = conn.execute(
                'SELECT created_at, metrics_json, separation_json FROM evaluations WHERE extraction_run_id = ?',
                (extraction_run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            'created_at': row['created_at'],
            'metrics': deserialize_json(row['metrics_json']),
            'separation': deserialize_json(row['separation_json']),
        }
