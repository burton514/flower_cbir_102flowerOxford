from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from flower_cbir_app.utils.common import blob_to_np, np_to_blob, serialize_json, deserialize_json


class SQLiteManager:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript(
                """
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
                    mean_blob BLOB,
                    std_blob BLOB,
                    vocab_blob BLOB,
                    extra_json TEXT
                );

                CREATE TABLE IF NOT EXISTS feature_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_run_id INTEGER,
                    image_id INTEGER,
                    feature_key TEXT,
                    vector_blob BLOB,
                    dim INTEGER,
                    extra_json TEXT,
                    FOREIGN KEY(image_id) REFERENCES images(image_id)
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    extraction_run_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    metrics_json TEXT,
                    separation_json TEXT
                );
                """
            )

    def create_preprocess_run(self, config: dict, summary: dict | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                'INSERT INTO preprocess_runs(config_json, summary_json) VALUES (?, ?)',
                (serialize_json(config), serialize_json(summary or {})),
            )
            return int(cur.lastrowid)

    def create_extraction_run(self, config: dict, summary: dict | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                'INSERT INTO extraction_runs(config_json, summary_json) VALUES (?, ?)',
                (serialize_json(config), serialize_json(summary or {})),
            )
            return int(cur.lastrowid)

    def update_extraction_run_summary(self, run_id: int, summary: dict):
        with self._connect() as conn:
            conn.execute('UPDATE extraction_runs SET summary_json = ? WHERE run_id = ?', (serialize_json(summary), run_id))

    def get_extraction_run_config(self, run_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute('SELECT config_json FROM extraction_runs WHERE run_id = ?', (run_id,)).fetchone()
        return deserialize_json(row['config_json']) if row else {}

    def reset_preprocess_data(self):
        """Xóa toàn bộ dữ liệu preprocess VÀ extraction (vì preprocess thay đổi thì extraction cũ không còn hợp lệ).
        Xóa luôn images để tránh tích lũy ảnh từ nhiều dataset_root khác nhau qua các lần chạy."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM evaluation_runs;
                DELETE FROM feature_vectors;
                DELETE FROM feature_configs;
                DELETE FROM extraction_runs;
                DELETE FROM preprocess_outputs;
                DELETE FROM preprocess_runs;
                DELETE FROM images;
            """)

    def reset_extraction_data(self):
        """Xóa toàn bộ dữ liệu extraction (giữ lại preprocess)."""
        with self._connect() as conn:
            conn.executescript("""
                DELETE FROM evaluation_runs;
                DELETE FROM feature_vectors;
                DELETE FROM feature_configs;
                DELETE FROM extraction_runs;
            """)

    def upsert_image(self, file_path: str, file_name: str, label: str) -> int:
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
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO preprocess_outputs(preprocess_run_id, image_id, normalized_path, mask_path, gray_path, edge_path, debug_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (preprocess_run_id, image_id, normalized_path, mask_path, gray_path, edge_path, serialize_json(debug_json)),
            )

    def insert_feature_config(self, extraction_run_id: int, feature_key: str, group_name: str, enabled: bool, distance_type: str, weight: float, is_meta: bool, mean, std, vocab, extra: dict | None = None):
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO feature_configs(extraction_run_id, feature_key, group_name, enabled, distance_type, weight, is_meta, mean_blob, std_blob, vocab_blob, extra_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    extraction_run_id,
                    feature_key,
                    group_name,
                    int(enabled),
                    distance_type,
                    float(weight),
                    int(is_meta),
                    np_to_blob(mean),
                    np_to_blob(std),
                    np_to_blob(vocab) if vocab is not None else None,
                    serialize_json(extra or {}),
                ),
            )

    def insert_feature_vector(self, extraction_run_id: int, image_id: int, feature_key: str, vector, extra: dict | None = None):
        vector = vector.astype('float32')
        with self._connect() as conn:
            conn.execute(
                '''
                INSERT INTO feature_vectors(extraction_run_id, image_id, feature_key, vector_blob, dim, extra_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (extraction_run_id, image_id, feature_key, np_to_blob(vector), int(vector.size), serialize_json(extra or {})),
            )

    def get_latest_preprocess_run_id(self) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute('SELECT run_id FROM preprocess_runs ORDER BY run_id DESC LIMIT 1').fetchone()
            return int(row['run_id']) if row else None

    def get_latest_extraction_run_id(self) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute('SELECT run_id FROM extraction_runs ORDER BY run_id DESC LIMIT 1').fetchone()
            return int(row['run_id']) if row else None

    def get_extraction_feature_configs(self, extraction_run_id: int) -> Dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM feature_configs WHERE extraction_run_id = ?', (extraction_run_id,)).fetchall()
        out = {}
        for row in rows:
            out[row['feature_key']] = {
                'feature_key': row['feature_key'],
                'group_name': row['group_name'],
                'enabled': bool(row['enabled']),
                'distance_type': row['distance_type'],
                'weight': float(row['weight']),
                'is_meta': bool(row['is_meta']),
                'mean': blob_to_np(row['mean_blob']) if row['mean_blob'] is not None else None,
                'std': blob_to_np(row['std_blob']) if row['std_blob'] is not None else None,
                'vocab': blob_to_np(row['vocab_blob']) if row['vocab_blob'] is not None else None,
                'extra': deserialize_json(row['extra_json']),
            }
        return out

    def get_feature_matrix(self, extraction_run_id: int, feature_key: str) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                '''
                SELECT fv.image_id, i.file_path, i.file_name, i.label, fv.vector_blob, fv.dim, fv.extra_json
                FROM feature_vectors fv
                JOIN images i ON i.image_id = fv.image_id
                WHERE fv.extraction_run_id = ? AND fv.feature_key = ?
                ORDER BY fv.image_id ASC
                ''',
                (extraction_run_id, feature_key),
            ).fetchall()
        data = []
        for row in rows:
            data.append({
                'image_id': int(row['image_id']),
                'file_path': row['file_path'],
                'file_name': row['file_name'],
                'label': row['label'],
                'vector': blob_to_np(row['vector_blob']),
                'dim': int(row['dim']),
                'extra': deserialize_json(row['extra_json']),
            })
        return pd.DataFrame(data)

    def get_all_feature_matrices(self, extraction_run_id: int) -> Dict[str, pd.DataFrame]:
        configs = self.get_extraction_feature_configs(extraction_run_id)
        return {k: self.get_feature_matrix(extraction_run_id, k) for k in configs.keys()}

    def list_images(self, limit: int = 100) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute('SELECT * FROM images ORDER BY image_id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]

    def insert_evaluation_run(self, extraction_run_id: int, metrics: dict, separation: dict):
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO evaluation_runs(extraction_run_id, metrics_json, separation_json) VALUES (?, ?, ?)',
                (extraction_run_id, serialize_json(metrics), serialize_json(separation)),
            )
