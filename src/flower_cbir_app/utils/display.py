
from __future__ import annotations

import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def render_debug_bundle(bundle: dict):
    """Hiển thị 1 debug_bundle lên giao diện Streamlit: ảnh -> biểu đồ -> bảng.

    Ảnh xếp lưới 3 cột, mỗi plot vẽ bằng matplotlib, mỗi table render thành
    dataframe. Dùng chung cho tab tiền xử lý / trích xuất / truy vấn.
    """
    images = bundle.get('images', {})
    if images:
        # Ẩn ảnh edge map khỏi UI; edge vẫn được tính ngầm trong preprocess và
        # các feature biên (canny/sobel) vẫn dùng trong fusion.
        keys = [k for k in images.keys() if 'edge' not in k]
        if keys:
            cols = st.columns(min(3, len(keys)))
            for idx, key in enumerate(keys):
                with cols[idx % len(cols)]:
                    st.image(images[key], caption=key, use_container_width=True)
    plots = bundle.get('plots', {})
    for name, payload in plots.items():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(payload.get('x', list(range(len(payload['y'])))), payload['y'])
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    tables = bundle.get('tables', {})
    for name, payload in tables.items():
        if name == 'preprocess_stats':
            continue  # Ẩn bảng preprocess_stats khỏi UI.
        st.markdown(f"**{name}**")
        st.dataframe(pd.DataFrame([payload]), use_container_width=True, hide_index=True)


def make_feature_config_dataframe(catalog: List, feature_state: dict) -> pd.DataFrame:
    """Tạo bảng tổng hợp cấu hình mọi feature (nhóm, bật/tắt, distance, weight...).

    Ghép thông tin tĩnh từ catalog với trạng thái hiện tại trong feature_state,
    trả về DataFrame để hiển thị nhanh ở tab Feature & Weight.
    """
    rows = []
    for feature in catalog:
        state = feature_state[feature.key]
        rows.append({
            'group': feature.group,
            'feature': feature.name,
            'enabled': state['enabled'],
            'distance': state['distance'],
            'weight': state['weight'],
            'retrieval': not feature.is_meta,
            'default_on': feature.enabled_by_default,
            'dimension': feature.output_dim_display,
        })
    return pd.DataFrame(rows)


# ── Từ điển chiều vector ─────────────────────────────────────────────────────
# Mô tả ý nghĩa từng chiều (hoặc nhóm chiều) của mỗi feature.
# Dùng để giải thích "vector biểu diễn cho gì" trong báo cáo / bảo vệ.

FEATURE_DIM_GLOSSARY: dict[str, list[dict]] = {
    'hsv_hist': [
        {'dims': '0–287', 'meaning': 'Histogram 3D màu HSV: 16 bins Hue × 6 bins Saturation × 3 bins Value. Mỗi bin = tỉ lệ pixel foreground có màu trong khoảng đó. L1-normalize → tổng = 1.'},
    ],
    'rgb_hist': [
        {'dims': '0–511', 'meaning': 'Histogram 3D màu RGB: 8×8×8 bins. Mỗi bin = tỉ lệ pixel foreground trong ô màu đó. L1-normalize.'},
    ],
    'hue_hist': [
        {'dims': '0–35', 'meaning': '36 bins Hue [0°–180°), mỗi bin rộng 5°. Mô tả phân bố màu sắc chủ đạo của bông hoa. L1-normalize.'},
    ],
    'color_moments': [
        {'dims': '0–2',  'meaning': 'Mean R, G, B của pixel foreground (giá trị trung bình màu).'},
        {'dims': '3–5',  'meaning': 'Std R, G, B (độ biến thiên màu).'},
        {'dims': '6–8',  'meaning': 'Skewness R, G, B (độ lệch phân bố màu, âm = lệch trái, dương = lệch phải).'},
    ],
    'lab_moments': [
        {'dims': '0–2',  'meaning': 'Mean L*, a*, b* trong không gian Lab (L=sáng, a=xanh↔đỏ, b=xanh↔vàng).'},
        {'dims': '3–5',  'meaning': 'Std L*, a*, b* (độ biến thiên sáng và màu sắc).'},
    ],
    'radial_color_hist': [
        {'dims': '0–11',  'meaning': 'Histogram Hue 12 bins của vành trong cùng (vùng tâm/nhụy hoa).'},
        {'dims': '12–23', 'meaning': 'Histogram Hue 12 bins của vành giữa.'},
        {'dims': '24–35', 'meaning': 'Histogram Hue 12 bins của vành ngoài cùng (rìa cánh hoa).'},
    ],
    'ccv': [
        {'dims': '0,2,4,...', 'meaning': 'α (coherent): số pixel màu bin k thuộc vùng liền khối lớn (≥25px) — mảng màu đồng đều.'},
        {'dims': '1,3,5,...', 'meaning': 'β (incoherent): số pixel màu bin k rải rác/nhỏ lẻ — đốm màu, vân, nhụy.'},
    ],
    'circular_hue_stats': [
        {'dims': '0', 'meaning': 'cos(circular mean Hue) — thành phần x của hướng màu trung bình.'},
        {'dims': '1', 'meaning': 'sin(circular mean Hue) — thành phần y của hướng màu trung bình.'},
        {'dims': '2', 'meaning': 'Resultant length R ∈ [0,1]: gần 1 = màu rất thuần/đồng đều, gần 0 = màu hỗn tạp.'},
        {'dims': '3', 'meaning': 'Circular std: √(−2 ln R), gần 0 = màu đồng đều.'},
        {'dims': '4', 'meaning': 'Mean Saturation foreground [0,1].'},
        {'dims': '5', 'meaning': 'Mean Value (brightness) foreground [0,1].'},
    ],
    'hu_moments': [
        {'dims': '0–6', 'meaning': '7 Hu Moments sau log-transform: bất biến với tịnh tiến, xoay, tỉ lệ. Mô tả hình dạng tổng thể của vùng hoa. Dim 0 = moment bậc thấp (hình dạng thô), dim 6 = moment bậc cao (chi tiết hình dạng).'},
    ],
    'geometric_shape': [
        {'dims': '0',  'meaning': 'area_ratio: diện tích bông / diện tích canvas.'},
        {'dims': '1',  'meaning': 'perimeter_norm: chu vi / (2×(H+W)).'},
        {'dims': '2',  'meaning': 'aspect_ratio: chiều rộng / chiều cao bounding box.'},
        {'dims': '3',  'meaning': 'circularity: 4π×area / perimeter². Gần 1 = tròn.'},
        {'dims': '4',  'meaning': 'solidity: area / convex_hull_area. Gần 1 = lồi.'},
        {'dims': '5',  'meaning': 'extent: area / (w×h bounding box).'},
        {'dims': '6',  'meaning': 'eccentricity: độ dẹt của ellipse khớp contour [0,1].'},
        {'dims': '7',  'meaning': 'equivalent_diameter_norm: đường kính tương đương / max(H,W).'},
        {'dims': '8',  'meaning': 'hull_area_ratio: diện tích convex hull / canvas.'},
        {'dims': '9',  'meaning': 'hull_perimeter_norm: chu vi hull / (2×(H+W)).'},
        {'dims': '10', 'meaning': 'convexity: hull_perimeter / perimeter.'},
        {'dims': '11', 'meaning': 'compactness: perimeter² / (4π×area).'},
        {'dims': '12', 'meaning': 'roundness: 4×area / (π×major_axis²).'},
    ],
    'radial_signature': [
        {'dims': '0–35', 'meaning': '36 bins góc đều nhau [0°–360°). Mỗi bin = khoảng cách max từ tâm đến biên theo hướng đó, normalize về [0,1]. Mô tả "profile" hình dạng bông hoa theo hướng.'},
    ],
    'fourier_shape': [
        {'dims': '0–31', 'meaning': '32 hệ số magnitude FFT của contour (đã resample, center, normalize). Dim 0 = thành phần tần số thấp (hình dạng thô), dim 31 = tần số cao (chi tiết biên).'},
    ],
    'symmetry_score': [
        {'dims': '0', 'meaning': 'Jaccard overlap khi lật ngang (left-right symmetry).'},
        {'dims': '1', 'meaning': 'Jaccard overlap khi lật dọc (up-down symmetry).'},
    ],
    'rotational_symmetry': [
        {'dims': '0–7', 'meaning': 'Overlap khi xoay mask đi k×45° (k=1..8). Dim k-1 cao → hoa có bậc đối xứng k. Ví dụ: dim 4 cao → hoa 5 cánh (xoay 72°≈45°×1.6, nhưng dim 4 = xoay 225° ≈ không phải; thực tế argmax+1 = số cánh ước lượng).'},
    ],
    'lbp': [
        {'dims': '0–25', 'meaning': '26 bins LBP uniform (P=24, R=3). Mỗi bin = tỉ lệ pixel foreground có pattern LBP tương ứng. Bin 0–24 = 25 pattern uniform, bin 25 = non-uniform. Mô tả kết cấu bề mặt cánh hoa.'},
    ],
    'glcm': [
        {'dims': '0', 'meaning': 'Contrast: đo sự chênh lệch cường độ giữa pixel kề nhau. Cao = kết cấu thô.'},
        {'dims': '1', 'meaning': 'Dissimilarity: tương tự contrast nhưng tuyến tính.'},
        {'dims': '2', 'meaning': 'Homogeneity: đo sự đồng đều. Cao = kết cấu mịn.'},
        {'dims': '3', 'meaning': 'Energy (ASM): tổng bình phương GLCM. Cao = kết cấu đều đặn.'},
        {'dims': '4', 'meaning': 'Correlation: tương quan tuyến tính giữa pixel kề nhau.'},
        {'dims': '5', 'meaning': 'ASM (Angular Second Moment): đo tính đồng nhất.'},
    ],
    'hog': [
        {'dims': '0–323', 'meaning': '324 bins HOG (9 orientations × 2×2 cells/block × nhiều blocks). Mỗi bin = histogram hướng gradient trong một ô nhỏ. Mô tả phân bố hướng cạnh/gradient toàn ảnh.'},
    ],
    'edge_orientation_hist': [
        {'dims': '0–35', 'meaning': '36 bins hướng gradient [0°–180°) tại pixel biên Canny, trọng số theo magnitude. Mô tả hướng chủ đạo của đường biên cánh hoa.'},
    ],
    'canny_derived': [
        {'dims': '0', 'meaning': 'edge_ratio: tỉ lệ pixel biên / pixel foreground.'},
        {'dims': '1', 'meaning': 'component_count: số thành phần liên thông của biên.'},
        {'dims': '2', 'meaning': 'mean_len: độ dài trung bình các đoạn biên.'},
        {'dims': '3', 'meaning': 'max_len: độ dài đoạn biên dài nhất.'},
        {'dims': '4', 'meaning': 'std_len: độ lệch chuẩn độ dài đoạn biên.'},
        {'dims': '5', 'meaning': 'center_density: mật độ biên ở vùng trung tâm ảnh.'},
    ],
    'sobel_hist': [
        {'dims': '0–35', 'meaning': '36 bins hướng gradient Sobel [0°–180°) trên toàn foreground, trọng số theo magnitude. Mô tả hướng gradient tổng thể (không chỉ tại biên).'},
    ],
    'sift_bovw': [
        {'dims': '0–V-1', 'meaning': 'Bag of Visual Words từ SIFT (V = vocab_size). Mỗi bin = tỉ lệ keypoint được gán vào visual word k. Visual word = centroid KMeans của descriptor 128-dim. L1-normalize.'},
    ],
    'orb_bovw': [
        {'dims': '0–V-1', 'meaning': 'BoVW từ ORB descriptor 32-dim. Tương tự SIFT BoVW nhưng dùng descriptor nhị phân nhanh hơn.'},
    ],
    'akaze_bovw': [
        {'dims': '0–V-1', 'meaning': 'BoVW từ AKAZE descriptor 61-dim. Bất biến với tỉ lệ và xoay, mạnh hơn ORB.'},
    ],
    'brisk_bovw': [
        {'dims': '0–V-1', 'meaning': 'BoVW từ BRISK descriptor 64-dim. Descriptor nhị phân, nhanh.'},
    ],
    'foreground_occupancy': [
        {'dims': '0', 'meaning': 'Tỉ lệ pixel foreground / tổng pixel canvas [0,1]. Dùng kiểm tra chất lượng mask.'},
    ],
    'centroid_offset': [
        {'dims': '0', 'meaning': 'Khoảng cách tâm foreground đến tâm canvas, normalize theo đường chéo [0,1]. Gần 0 = bông hoa ở giữa ảnh.'},
    ],
    'mask_quality': [
        {'dims': '0', 'meaning': 'Số thành phần liên thông của mask (lý tưởng = 1).'},
        {'dims': '1', 'meaning': 'Có chạm biên ảnh không (0/1).'},
        {'dims': '2', 'meaning': 'Tỉ lệ lỗ bên trong foreground / diện tích foreground đã fill.'},
    ],
}


def render_feature_glossary(catalog: list, feature_state: dict):
    """Render bảng từ điển chiều vector cho các feature đang bật."""
    st.markdown("### 📖 Từ điển chiều vector")
    st.caption(
        "Mô tả ý nghĩa từng chiều (hoặc nhóm chiều) của mỗi feature. "
        "Dùng để giải thích 'vector biểu diễn cho gì' khi báo cáo."
    )

    enabled_keys = [f.key for f in catalog if feature_state.get(f.key, {}).get('enabled', False) and not getattr(f, 'hidden_in_ui', False)]
    if not enabled_keys:
        st.info("Chưa có feature nào được bật.")
        return

    for feature in catalog:
        if feature.key not in enabled_keys:
            continue
        glossary = FEATURE_DIM_GLOSSARY.get(feature.key)
        if not glossary:
            continue
        with st.expander(f"**{feature.name}** ({feature.group}) — dim: {feature.output_dim_display}", expanded=False):
            rows = []
            for entry in glossary:
                rows.append({'Chiều': entry['dims'], 'Ý nghĩa': entry['meaning']})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
