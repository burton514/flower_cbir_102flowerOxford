
import json
from pathlib import Path


def load_json(path):
    """Đọc 1 file JSON (UTF-8) và trả về dict/list. Dùng để nạp config."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, data):
    """Ghi dict ra file JSON (UTF-8, indent 2), tự tạo thư mục cha nếu thiếu."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def deep_update(base: dict, incoming: dict):
    """Gộp ĐỆ QUY incoming vào base (merge dict lồng nhau, không ghi đè cả nhánh).

    Khác dict.update thông thường ở chỗ với giá trị là dict thì đi sâu vào trộn
    từng khóa, giữ lại các khóa cũ. Dùng khi nạp config từ file đè lên config mặc định.
    """
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base
