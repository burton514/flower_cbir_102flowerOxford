import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flower_cbir_app.core.offline_pipeline import run_offline_preprocess, run_feature_extraction
from flower_cbir_app.features.registry import get_default_feature_state
from flower_cbir_app.utils.config_utils import load_json

def print_progress(pct, msg):
    print(f"[{pct*100:.1f}%] {msg}")

def main():
    print("Loading config...")
    config_path = ROOT / "config" / "default_config.json"
    system_config = load_json(config_path)
    feature_state = get_default_feature_state()
    
    print("\n--- BẮT ĐẦU TIỀN XỬ LÝ OFFLINE ---")
    res_prep = run_offline_preprocess(system_config, sample_limit=0, progress_callback=print_progress)
    print(f"\nKẾT QUẢ TIỀN XỬ LÝ: {res_prep['message']}")
    
    print("\n--- BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG ---")
    res_ext = run_feature_extraction(system_config, feature_state, progress_callback=print_progress)
    print(f"\nKẾT QUẢ TRÍCH XUẤT: {res_ext['message']}")
    
    print("\n--- REBUILD HOÀN TẤT ---")

if __name__ == "__main__":
    main()
