import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flower_cbir_app.core.offline_pipeline import inspect_dataset, run_offline_preprocess, run_feature_extraction
from flower_cbir_app.core.online_pipeline import run_query
from flower_cbir_app.evaluation.class_separation import evaluate_class_separation
from flower_cbir_app.evaluation.retrieval_metrics import evaluate_dataset_retrieval
from flower_cbir_app.features.registry import get_default_feature_state, get_feature_catalog
from flower_cbir_app.storage.sqlite_manager import SQLiteManager
from flower_cbir_app.utils.config_utils import load_json, save_json, deep_update
from flower_cbir_app.utils.display import render_debug_bundle, make_feature_config_dataframe, render_feature_glossary

st.set_page_config(page_title="Flower CBIR Workbench", layout="wide")
st.title("Flower CBIR Workbench")
st.caption("CBIR ảnh hoa theo hướng feature-based, offline/online pipeline tách riêng, lưu SQLite.")

DEFAULT_CONFIG_PATH = ROOT / "config" / "default_config.json"


TAB_NAMES = [
    "Feature & Weight",
    "Tiền xử lí offline",
    "Trích xuất đặc trưng",
    "Đánh giá",
    "Truy vấn",
    "SQLite / Xem DB",
]

def ensure_session_state():
    if "system_config" not in st.session_state:
        st.session_state.system_config = load_json(DEFAULT_CONFIG_PATH)
    if "feature_state" not in st.session_state:
        st.session_state.feature_state = get_default_feature_state()
    if "applied_config" not in st.session_state:
        st.session_state.applied_config = {
            "system": json.loads(json.dumps(st.session_state.system_config)),
            "features": json.loads(json.dumps(st.session_state.feature_state)),
        }
    if "last_messages" not in st.session_state:
        st.session_state.last_messages = []
    # Giữ tab đang active
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = TAB_NAMES[0]
    # Lưu kết quả từng tab để không bị mất khi chuyển tab
    if "inspect_result" not in st.session_state:
        st.session_state.inspect_result = None
    if "preprocess_result" not in st.session_state:
        st.session_state.preprocess_result = None
    if "extract_result" not in st.session_state:
        st.session_state.extract_result = None
    if "eval_result" not in st.session_state:
        st.session_state.eval_result = None
    if "fisher_weight_result" not in st.session_state:
        st.session_state.fisher_weight_result = None
    if "query_result" not in st.session_state:
        st.session_state.query_result = None


def append_message(msg: str):
    st.session_state.last_messages.append(msg)
    st.session_state.last_messages = st.session_state.last_messages[-12:]


ensure_session_state()

with st.sidebar:
    st.header("Cấu hình hệ thống")
    cfg = st.session_state.system_config
    cfg["dataset_root"] = st.text_input("Dataset root", value=cfg.get("dataset_root", ""))
    cfg["workspace_root"] = st.text_input("Workspace root", value=cfg.get("workspace_root", "./workspace"))
    cfg["db_path"] = st.text_input("SQLite path", value=cfg.get("db_path", "./workspace/flower_cbir.sqlite"))
    cfg["label_source"] = st.selectbox(
        "Cách lấy nhãn lớp",
        options=["auto", "parent_folder", "filename_prefix"],
        index=["auto", "parent_folder", "filename_prefix"].index(cfg.get("label_source", "auto")) if cfg.get("label_source", "auto") in ["auto", "parent_folder", "filename_prefix"] else 0,
        help="auto: ưu tiên tên thư mục con của dataset, nếu không có thì lấy prefix tên file trước dấu _."
    )
    cfg["preprocessing"]["use_rembg"] = st.checkbox("Dùng rembg nếu ảnh chưa có alpha", value=cfg["preprocessing"].get("use_rembg", True))
    cfg["preprocessing"]["target_size"] = st.number_input("Target size", min_value=64, max_value=1024, value=int(cfg["preprocessing"].get("target_size", 256)), step=32)
    cfg["preprocessing"]["target_object_ratio"] = st.slider("Tỉ lệ chiếm khung của object", min_value=0.4, max_value=0.95, value=float(cfg["preprocessing"].get("target_object_ratio", 0.78)), step=0.01)
    cfg["local_bovw"]["vocab_size"] = st.number_input("BoVW vocab size", min_value=8, max_value=256, value=int(cfg["local_bovw"].get("vocab_size", 32)), step=8)
    cfg["fusion"]["auto_weight"] = st.checkbox("Auto weight theo nhóm", value=cfg["fusion"].get("auto_weight", True))

    uploaded_cfg = st.file_uploader("Nạp config JSON", type=["json"])
    if uploaded_cfg is not None:
        external_cfg = json.load(uploaded_cfg)
        deep_update(st.session_state.system_config, external_cfg.get("system", external_cfg))
        if "features" in external_cfg:
            st.session_state.feature_state = external_cfg["features"]
        append_message("Đã nạp config từ file JSON.")

    if st.button("Áp dụng cấu hình", use_container_width=True, type="primary"):
        st.session_state.applied_config = {
            "system": json.loads(json.dumps(st.session_state.system_config)),
            "features": json.loads(json.dumps(st.session_state.feature_state)),
        }
        workspace = Path(st.session_state.system_config["workspace_root"])
        workspace.mkdir(parents=True, exist_ok=True)
        save_json(workspace / "active_config.json", st.session_state.applied_config)
        append_message("Đã áp dụng cấu hình hiện tại. Các nút xử lí phía dưới sẽ dùng đúng cấu hình này.")

    if st.button("Lưu config JSON", use_container_width=True):
        workspace = Path(st.session_state.system_config["workspace_root"])
        workspace.mkdir(parents=True, exist_ok=True)
        out = workspace / "saved_config.json"
        save_json(out, {
            "system": st.session_state.system_config,
            "features": st.session_state.feature_state,
        })
        append_message(f"Đã lưu config ra {out}")

st.subheader("Nhật ký ngắn")
for msg in st.session_state.last_messages[::-1]:
    st.write(f"- {msg}")

catalog = get_feature_catalog()

# Tab selector — dùng radio nằm ngang để giữ tab khi re-render
st.session_state.active_tab = st.radio(
    "Chọn tab",
    options=TAB_NAMES,
    index=TAB_NAMES.index(st.session_state.active_tab),
    horizontal=True,
    label_visibility="collapsed",
    key="tab_selector",
)
st.divider()
active_tab = st.session_state.active_tab

if active_tab == "Feature & Weight":
    st.markdown("### Danh sách feature")
    groups = {}
    for feature in catalog:
        groups.setdefault(feature.group, []).append(feature)

    for group_name, items in groups.items():
        with st.expander(f"Nhóm: {group_name}", expanded=True):
            for feature in items:
                state = st.session_state.feature_state[feature.key]
                cols = st.columns([2, 1, 1, 1, 2])
                state["enabled"] = cols[0].checkbox(
                    f"{feature.name}",
                    value=state["enabled"],
                    key=f"enabled_{feature.key}",
                    help=feature.description,
                )
                distance_options = ["cosine", "l2", "chi_square"] if feature.supports_chi_square else ["cosine", "l2"]
                if state.get("distance") not in distance_options:
                    state["distance"] = feature.default_distance if feature.default_distance in distance_options else distance_options[0]
                state["distance"] = cols[1].selectbox(
                    "Distance",
                    options=distance_options,
                    index=distance_options.index(state["distance"]),
                    key=f"distance_{feature.key}",
                    label_visibility="collapsed",
                    help="Chi-square chỉ được phép dùng với histogram không âm phù hợp χ².",
                )
                state["weight"] = cols[2].number_input(
                    "Weight",
                    min_value=0.0,
                    value=float(state["weight"]),
                    step=0.1,
                    key=f"weight_{feature.key}",
                    label_visibility="collapsed",
                )
                cols[3].write(f"Dim: {feature.output_dim_display}")
                cols[4].caption(feature.description)

    st.markdown("### Bảng tổng hợp nhanh")
    df = make_feature_config_dataframe(catalog, st.session_state.feature_state)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Fisher Weight ────────────────────────────────────────────────────────
    st.markdown("### ⚖️ Tính trọng số theo Fisher Ratio")
    st.caption(
        "Fisher ratio = S_B / S_W (phương sai giữa lớp / trong lớp). "
        "Feature nào tách lớp tốt hơn sẽ được weight cao hơn. "
        "Cần đã có extraction run trong SQLite."
    )
    if st.button("Tính Fisher Weight từ dữ liệu", use_container_width=True):
        try:
            from flower_cbir_app.core.fusion import build_fisher_weights
            import numpy as _np
            db_fw = SQLiteManager(st.session_state.applied_config["system"]["db_path"])
            run_id = db_fw.get_latest_extraction_run_id()
            if run_id is None:
                st.session_state.fisher_weight_result = {"warning": "Chưa có extraction run. Hãy trích xuất đặc trưng trước."}
            else:
                configs_fw = db_fw.get_extraction_feature_configs(run_id)
                matrices_fw = {k: db_fw.get_feature_matrix(run_id, k) for k in configs_fw}
                matrices_fw = {k: v for k, v in matrices_fw.items() if not v.empty}
                base_fw = next(iter(matrices_fw.values())) if matrices_fw else None
                if base_fw is None:
                    st.session_state.fisher_weight_result = {"warning": "Không có vector trong DB."}
                else:
                    labels_fw = _np.asarray(base_fw['label'].tolist())
                    fw = build_fisher_weights(
                        configs_fw, matrices_fw, labels_fw,
                        exclude_meta_from_retrieval=True,
                    )
                    st.session_state.fisher_weight_result = fw
                    append_message("Đã tính Fisher weight từ dữ liệu.")
        except Exception as exc:
            st.session_state.fisher_weight_result = {"error": str(exc)}

    if st.session_state.fisher_weight_result is not None:
        r = st.session_state.fisher_weight_result
        if isinstance(r, dict) and "error" in r:
            st.exception(Exception(r["error"]))
        elif isinstance(r, dict) and "warning" in r:
            st.warning(r["warning"])
        else:
            fw_df = pd.DataFrame([
                {"feature": k, "fisher_weight": round(v, 6)}
                for k, v in sorted(r.items(), key=lambda x: -x[1])
            ])
            st.dataframe(fw_df, use_container_width=True, hide_index=True)
            if st.button("Áp dụng Fisher Weight vào cấu hình hiện tại", type="primary"):
                for k, v in r.items():
                    if k in st.session_state.feature_state:
                        st.session_state.feature_state[k]["weight"] = float(v)
                append_message("Đã áp dụng Fisher weight vào feature_state. Nhớ bấm 'Áp dụng cấu hình' để lưu.")
                st.rerun()

    st.divider()

    # ── Từ điển chiều vector ─────────────────────────────────────────────────
    render_feature_glossary(catalog, st.session_state.feature_state)

# ── Tab: Tiền xử lí offline ──────────────────────────────────────────────────
if active_tab == "Tiền xử lí offline":
    st.markdown("### Pipeline offline")
    st.write("Tạo bộ ảnh chuẩn và lưu kết quả trung gian vào workspace.")

    if st.button("Kiểm tra nhanh dataset", use_container_width=True):
        try:
            summary = inspect_dataset(st.session_state.applied_config["system"])
            st.session_state.inspect_result = summary
        except Exception as exc:
            st.session_state.inspect_result = {"error": str(exc)}

    if st.session_state.inspect_result is not None:
        r = st.session_state.inspect_result
        if "error" in r:
            st.error(r["error"])
        else:
            st.markdown("#### Thống kê dataset")
            overview = {k: v for k, v in r.items() if k not in {"label_counts"}}
            st.json(overview)
            if r.get("label_counts"):
                label_df = pd.DataFrame([
                    {"label": label, "count": count}
                    for label, count in r["label_counts"].items()
                ])
                st.markdown("#### Số ảnh theo nhãn")
                st.dataframe(label_df, use_container_width=True, hide_index=True)

    st.divider()

    sample_limit = st.number_input("Số ảnh hiển thị log/debug gần nhất", min_value=1, max_value=20, value=5, key="sample_limit_input")

    if st.button("Chạy tiền xử lí offline", type="primary"):
        try:
            progress_bar = st.progress(0, text="Đang khởi động...")
            status_text = st.empty()

            def preprocess_progress(pct: float, msg: str):
                progress_bar.progress(min(pct, 1.0), text=msg)
                status_text.caption(msg)

            result = run_offline_preprocess(
                st.session_state.applied_config["system"],
                sample_limit=int(sample_limit),
                progress_callback=preprocess_progress,
            )
            progress_bar.progress(1.0, text="✅ Hoàn tất!")
            status_text.empty()
            append_message(result["message"])
            st.session_state.preprocess_result = result
        except Exception as exc:
            st.session_state.preprocess_result = {"error": str(exc)}

    if st.session_state.preprocess_result is not None:
        r = st.session_state.preprocess_result
        if "error" in r:
            st.exception(Exception(r["error"]))
        else:
            st.success(r["message"])
            st.json({k: v for k, v in r.items() if k not in {"samples", "message"}})
            for sample in r.get("samples", []):
                st.markdown(f"#### {sample['file_name']}")
                render_debug_bundle(sample["debug_bundle"])

# ── Tab: Trích xuất đặc trưng ────────────────────────────────────────────────
if active_tab == "Trích xuất đặc trưng":
    st.markdown("### Trích xuất đặc trưng")
    st.write("Nút này dùng đúng bộ cấu hình đã được Áp dụng gần nhất.")

    if st.button("Trích xuất đặc trưng", type="primary"):
        try:
            progress_bar = st.progress(0, text="Đang khởi động...")
            status_text = st.empty()

            def extract_progress(pct: float, msg: str):
                progress_bar.progress(min(pct, 1.0), text=msg)
                status_text.caption(msg)

            result = run_feature_extraction(
                st.session_state.applied_config["system"],
                st.session_state.applied_config["features"],
                progress_callback=extract_progress,
            )
            progress_bar.progress(1.0, text="✅ Hoàn tất!")
            status_text.empty()
            append_message(result["message"])
            st.session_state.extract_result = result
        except Exception as exc:
            st.session_state.extract_result = {"error": str(exc)}

    if st.session_state.extract_result is not None:
        r = st.session_state.extract_result
        if "error" in r:
            st.exception(Exception(r["error"]))
        else:
            st.success(r["message"])
            st.json({k: v for k, v in r.items() if k not in {"sample_debug", "message"}})
            if r.get("sample_debug"):
                st.markdown("### Kết quả trung gian từ một số ảnh mẫu")
                for item in r["sample_debug"]:
                    st.markdown(f"#### {item['file_name']}")
                    render_debug_bundle(item["debug_bundle"])

# ── Tab: Đánh giá ────────────────────────────────────────────────────────────
if active_tab == "Đánh giá":
    st.markdown("### Đánh giá")

    if st.button("Đánh giá", type="primary"):
        try:
            db = SQLiteManager(st.session_state.applied_config["system"]["db_path"])
            extraction_run_id = db.get_latest_extraction_run_id()
            if extraction_run_id is None:
                st.session_state.eval_result = {"warning": "Chưa có extraction run nào trong SQLite."}
            else:
                retrieval_bar = st.progress(0, text="Đang tính retrieval metrics...")
                retrieval_status = st.empty()

                def retrieval_progress(pct: float, msg: str):
                    retrieval_bar.progress(min(pct, 1.0), text=msg)
                    retrieval_status.caption(msg)

                metrics = evaluate_dataset_retrieval(db, extraction_run_id, progress_callback=retrieval_progress)
                retrieval_bar.progress(1.0, text="✅ Retrieval metrics xong!")
                retrieval_status.empty()

                with st.spinner("Đang tính class separation metrics..."):
                    separation = evaluate_class_separation(db, extraction_run_id)

                append_message("Đã tính xong metric truy hồi và độ tách lớp.")
                st.session_state.eval_result = {
                    "metrics": metrics,
                    "separation": separation,
                }
        except Exception as exc:
            st.session_state.eval_result = {"error": str(exc)}

    if st.session_state.eval_result is not None:
        r = st.session_state.eval_result
        if "error" in r:
            st.exception(Exception(r["error"]))
        elif "warning" in r:
            st.warning(r["warning"])
        else:
            metrics = r["metrics"]
            separation = r["separation"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Retrieval @5 (tổng thể)")
                overall = {k: v for k, v in metrics.items() if k != "per_label"}
                st.dataframe(pd.DataFrame([overall]))
            with col2:
                st.markdown("#### Class separation")
                st.dataframe(pd.DataFrame([separation]))

            per_label = metrics.get("per_label", [])
            if per_label:
                st.markdown("#### Retrieval @5 theo từng nhãn")
                df_label = pd.DataFrame(per_label)
                df_label = df_label.sort_values("precision_at_5", ascending=False).reset_index(drop=True)
                st.dataframe(
                    df_label.style.background_gradient(
                        subset=["precision_at_5", "recall_at_5", "map_at_5", "mrr_at_5"],
                        cmap="RdYlGn",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Tab: Truy vấn ────────────────────────────────────────────────────────────
if active_tab == "Truy vấn":
    st.markdown("### Truy vấn")
    mode = st.radio("Nguồn query", options=["Chọn ảnh trong dataset", "Upload ảnh ngoài dataset"], horizontal=True)
    db = SQLiteManager(st.session_state.applied_config["system"]["db_path"])
    dataset_images = db.list_images(limit=5000)
    selected_path = None
    upload_file = None
    if mode == "Chọn ảnh trong dataset":
        mapping = {f"{row['label']} | {row['file_name']}": row["file_path"] for row in dataset_images}
        if mapping:
            selected_label = st.selectbox("Ảnh query", options=list(mapping.keys()))
            selected_path = mapping[selected_label]
        else:
            st.info("SQLite chưa có ảnh. Hãy preprocess/extract trước.")
    else:
        upload_file = st.file_uploader("Upload ảnh query", type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"])

    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5)

    if st.button("Truy vấn", type="primary"):
        try:
            if selected_path is None and upload_file is None:
                st.warning("Chưa có ảnh query.")
            else:
                with st.spinner("Đang xử lí ảnh query và tìm kiếm..."):
                    result = run_query(
                        st.session_state.applied_config["system"],
                        st.session_state.applied_config["features"],
                        db,
                        query_image_path=selected_path,
                        query_file_bytes=upload_file.read() if upload_file is not None else None,
                        top_k=int(top_k),
                    )
                append_message("Đã chạy truy vấn theo bộ đặc trưng hiện hành.")
                st.session_state.query_result = result
        except Exception as exc:
            st.session_state.query_result = {"error": str(exc)}

    if st.session_state.query_result is not None:
        r = st.session_state.query_result
        if "error" in r:
            st.exception(Exception(r["error"]))
        else:
            st.success("Truy vấn hoàn tất!")
            st.markdown("### Kết quả trung gian của query")
            render_debug_bundle(r["query_debug_bundle"])
            st.markdown("### Top kết quả")
            results = r["results"]
            if results:
                cols = st.columns(len(results))
                for col, row in zip(cols, results):
                    with col:
                        img_path = row.get("file_path", "")
                        try:
                            col.image(img_path, use_container_width=True)
                        except Exception:
                            col.warning("Không tải được ảnh")
                        rank = results.index(row) + 1
                        distance_val = row.get("distance_score", row.get("score", 0))
                        similarity_val = row.get("similarity", max(0.0, 1.0 - distance_val))
                        lbl = row.get("label", "?")
                        fname = row.get("file_name", "")
                        col.markdown(
                            f"**#{rank}** &nbsp; `{lbl}`  \n"
                            f"Distance: `{distance_val:.4f}`  \n"
                            f"Similarity: `{similarity_val:.4f}`  \n"
                            f"<small>{fname}</small>",
                            unsafe_allow_html=True,
                        )
                st.markdown("---")
                result_df = pd.DataFrame([{k: v for k, v in row.items() if k != "feature_details"} for row in results])
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                contrib = r.get("per_feature_contributions", [])
                if contrib:
                    st.markdown("### Đóng góp từng đặc trưng vào điểm Top-K")
                    st.caption("Distance raw được chuẩn hóa min-max theo từng feature; contribution = weight × normalized_distance. Tổng contribution là distance_score, càng nhỏ càng giống.")
                    st.dataframe(pd.DataFrame(contrib), use_container_width=True, hide_index=True)

# ── Tab: SQLite / Xem DB ─────────────────────────────────────────────────────
if active_tab == "SQLite / Xem DB":
    st.markdown("### SQLite")
    db = SQLiteManager(st.session_state.applied_config["system"]["db_path"])
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Run gần nhất")
        st.write({
            "latest_preprocess_run_id": db.get_latest_preprocess_run_id(),
            "latest_extraction_run_id": db.get_latest_extraction_run_id(),
        })
    with col2:
        st.markdown("#### Ảnh gần nhất")
        st.dataframe(pd.DataFrame(db.list_images(limit=20)), use_container_width=True, hide_index=True)
