from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import skew

from flower_cbir_app.features.base import FeatureResult
from flower_cbir_app.utils.common import normalize_vector


def _masked_pixels(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Lấy các pixel thuộc vùng vật (mask>0) dưới dạng mảng (N,3).

    Nếu mask rỗng thì fallback dùng toàn bộ pixel ảnh để không vỡ pipeline.
    """
    pixels = image_rgb[mask > 0]
    if len(pixels) == 0:
        return image_rgb.reshape(-1, 3)
    return pixels


def extract_hsv_hist(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    """Histogram màu 3D trong không gian HSV (16 Hue x 6 Sat x 3 Val = 288 chiều).

    Chỉ tính trên vùng vật, L1-normalize. HSV tách màu (Hue) khỏi độ sáng nên
    bền với thay đổi ánh sáng hơn RGB. Là feature màu chủ lực.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], (mask > 0).astype('uint8'), [16, 6, 3], [0, 180, 0, 256, 0, 256]).flatten()
    hist = normalize_vector(hist)
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'HSV histogram': {'y': hist.tolist()}}, 'tables': {}}, {})


def extract_rgb_hist(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    """Histogram màu 3D trong không gian RGB (8 x 8 x 8 = 512 chiều).

    Chỉ tính trên vùng vật, L1-normalize. Đơn giản, nhạy với ánh sáng hơn HSV.
    """
    hist = cv2.calcHist([image_rgb], [0, 1, 2], (mask > 0).astype('uint8'), [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
    hist = normalize_vector(hist)
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'RGB histogram': {'y': hist.tolist()}}, 'tables': {}}, {})


def extract_hue_hist(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    """Histogram chỉ kênh Hue (36 chiều) — phân bố tông màu thuần.

    Bỏ qua độ bão hòa và độ sáng nên rất gọn và tập trung vào "màu gì". Hợp khi
    màu là dấu hiệu phân biệt chính giữa các loài hoa.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0], (mask > 0).astype('uint8'), [36], [0, 180]).flatten()
    hist = normalize_vector(hist)
    return FeatureResult(hist.astype('float32'), {'images': {}, 'plots': {'Hue histogram': {'y': hist.tolist()}}, 'tables': {}}, {})


def extract_dominant_colors(image_rgb: np.ndarray, mask: np.ndarray, clusters: int = 5) -> FeatureResult:
    """Palette màu chủ đạo bằng KMeans: `clusters` màu kèm tỉ lệ diện tích.

    Vector = nối [R,G,B,ratio] của từng cụm đã sắp theo tỉ lệ giảm dần
    (dim = clusters x 4). Vì so sánh palette bằng vector nối là heuristic nên
    feature này mặc định TẮT. Seed cố định để tái lập kết quả.
    """
    # Đặt seed OpenCV để kết quả tái lập khi chạy lại cùng dữ liệu.
    cv2.setRNGSeed(42)
    pixels = _masked_pixels(image_rgb, mask).astype(np.float32)
    if len(pixels) < clusters:
        if len(pixels) == 0:
            pixels = np.zeros((clusters, 3), dtype=np.float32)
        else:
            pixels = np.vstack([pixels, np.repeat(pixels[:1], clusters - len(pixels), axis=0)])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(pixels, clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    ratios = np.array([(labels == i).mean() for i in range(clusters)], dtype=np.float32)
    order = np.argsort(-ratios)
    centers = centers[order] / 255.0
    ratios = ratios[order]
    vector = np.hstack([np.hstack([centers[i], ratios[i]]) for i in range(clusters)]).astype('float32')
    table = {}
    for i in range(clusters):
        table[f'color_{i+1}'] = centers[i].tolist()
        table[f'ratio_{i+1}'] = float(ratios[i])
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'dominant_colors': table}}, {})


def extract_color_moments(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    """Color moments: mean + std + skewness theo từng kênh RGB (9 chiều).

    Mô tả màu cực gọn bằng 3 mô-men thống kê/kênh. Dùng RGB thay HSV để tránh
    Hue là đại lượng vòng (tính trung bình sai). Vector = [means(3), stds(3),
    skews(3)].
    """
    rgb = image_rgb.astype(np.float32)
    pixels = rgb[mask > 0]
    if len(pixels) == 0:
        pixels = rgb.reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    skews = np.array([float(skew(pixels[:, i])) if len(np.unique(pixels[:, i])) > 1 else 0.0 for i in range(3)], dtype=np.float32)
    vector = np.hstack([means, stds, skews]).astype('float32')
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'color_moments': {'mean_r': float(means[0]), 'mean_g': float(means[1]), 'mean_b': float(means[2])}}}, {})


def extract_lab_moments(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    """Mean + std theo từng kênh trong không gian màu LAB (6 chiều).

    LAB tách độ sáng (L) khỏi màu (a,b) và gần với cảm nhận màu của mắt người,
    nên khoảng cách màu hợp lý hơn RGB. Vector = [means(3), stds(3)].
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[mask > 0]
    if len(pixels) == 0:
        pixels = lab.reshape(-1, 3)
    means = pixels.mean(axis=0)
    stds = pixels.std(axis=0)
    vector = np.hstack([means, stds]).astype('float32')
    return FeatureResult(vector, {'images': {}, 'plots': {}, 'tables': {'lab_moments': {'mean_l': float(means[0]), 'std_l': float(stds[0])}}}, {})


# ── Feature mới: Radial Color Histogram ─────────────────────────────────────
def extract_radial_color_hist(image_rgb: np.ndarray, mask: np.ndarray, rings: int = 3, hue_bins: int = 12) -> FeatureResult:
    """Histogram màu Hue theo vành đồng tâm (radial zones).

    Ý nghĩa từng chiều vector:
      - Chia bông hoa thành `rings` vành đồng tâm từ tâm ra rìa theo bán kính.
      - Mỗi vành tính histogram Hue `hue_bins` bins → L1-normalize.
      - Vector cuối = nối `rings` histogram lại → dim = rings × hue_bins.

    Tại sao hữu ích cho hoa:
      - Nhiều loài hoa có màu tâm (nhụy) khác màu cánh ngoài.
      - Histogram màu global không phân biệt được điều này.
      - Ví dụ: hoa hướng dương có tâm nâu-đen, cánh vàng → vành trong ≠ vành ngoài.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.float32)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return FeatureResult(
            np.zeros(rings * hue_bins, dtype=np.float32),
            {'images': {}, 'plots': {}, 'tables': {}}, {}
        )

    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    radii = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_r = float(radii.max()) + 1e-12

    ring_hists = []
    for r in range(rings):
        r_lo = max_r * r / rings
        r_hi = max_r * (r + 1) / rings
        in_ring = (radii >= r_lo) & (radii < r_hi)
        if not np.any(in_ring):
            ring_hists.append(np.zeros(hue_bins, dtype=np.float32))
            continue
        hue_vals = hue[ys[in_ring], xs[in_ring]]
        hist, _ = np.histogram(hue_vals, bins=hue_bins, range=(0, 180))
        ring_hists.append(normalize_vector(hist.astype(np.float32)))

    vector = np.concatenate(ring_hists).astype(np.float32)

    # Debug: tạo ảnh trực quan vành đồng tâm
    ring_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors = [(220, 50, 50), (50, 200, 50), (50, 50, 220)]
    for r in range(rings):
        r_lo = max_r * r / rings
        r_hi = max_r * (r + 1) / rings
        all_ys, all_xs = np.where(mask > 0)
        all_r = np.sqrt((all_xs - cx) ** 2 + (all_ys - cy) ** 2)
        in_ring = (all_r >= r_lo) & (all_r < r_hi)
        ring_vis[all_ys[in_ring], all_xs[in_ring]] = colors[r % len(colors)]

    plots = {f'Ring {i+1} Hue hist': {'y': ring_hists[i].tolist()} for i in range(rings)}
    return FeatureResult(
        vector,
        {'images': {'Radial zones': ring_vis}, 'plots': plots, 'tables': {'radial_color': {'rings': rings, 'hue_bins': hue_bins, 'dim': int(vector.size)}}},
        {}
    )


# ── Feature mới: Color Coherence Vector (CCV) ───────────────────────────────
def extract_ccv(image_rgb: np.ndarray, mask: np.ndarray, hue_bins: int = 12, min_coherent_size: int = 25) -> FeatureResult:
    """Color Coherence Vector (Pass & Zabih, 1996).

    Ý nghĩa từng chiều vector:
      - Với mỗi bin màu Hue, đếm pixel thuộc vùng màu liền khối lớn (coherent α)
        và pixel màu đó nhưng rải rác/nhỏ lẻ (incoherent β).
      - Vector = [α₁, β₁, α₂, β₂, ..., αₙ, βₙ] → dim = 2 × hue_bins, L1-normalize.

    Tại sao hữu ích:
      - Phân biệt "mảng màu lớn" (cánh hoa đồng màu) với "đốm màu lốm đốm" (nhụy, vân).
      - Hai ảnh cùng histogram màu nhưng khác phân bố không gian sẽ có CCV khác nhau.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    bin_map = (hue.astype(np.float32) * hue_bins / 180.0).astype(np.int32)
    bin_map = np.clip(bin_map, 0, hue_bins - 1)

    alpha = np.zeros(hue_bins, dtype=np.float32)  # coherent
    beta  = np.zeros(hue_bins, dtype=np.float32)  # incoherent

    for b in range(hue_bins):
        bin_mask = ((bin_map == b) & (mask > 0)).astype(np.uint8)
        if not np.any(bin_mask):
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area >= min_coherent_size:
                alpha[b] += area
            else:
                beta[b] += area

    vector = np.empty(2 * hue_bins, dtype=np.float32)
    vector[0::2] = alpha
    vector[1::2] = beta
    vector = normalize_vector(vector)

    table = {f'bin_{b}_coherent': float(alpha[b]) for b in range(hue_bins)}
    table.update({f'bin_{b}_incoherent': float(beta[b]) for b in range(hue_bins)})
    return FeatureResult(
        vector,
        {'images': {}, 'plots': {'CCV coherent': {'y': alpha.tolist()}, 'CCV incoherent': {'y': beta.tolist()}}, 'tables': {'ccv': table}},
        {}
    )


# ── Feature mới: Circular Hue Statistics ────────────────────────────────────
def extract_circular_hue_stats(image_rgb: np.ndarray, mask: np.ndarray) -> FeatureResult:
    """Thống kê Hue theo thống kê hướng vòng tròn (circular statistics).

    Ý nghĩa từng chiều vector (6 chiều):
      [0] circular_mean_cos  — cos của góc trung bình Hue (thành phần x)
      [1] circular_mean_sin  — sin của góc trung bình Hue (thành phần y)
      [2] resultant_length   — độ tập trung màu [0,1]; gần 1 = màu rất thuần
      [3] circular_std       — độ lệch chuẩn vòng tròn; gần 0 = màu đồng đều
      [4] mean_saturation    — độ bão hòa trung bình foreground [0,1]
      [5] mean_value         — độ sáng trung bình foreground [0,1]

    Tại sao dùng circular statistics:
      - Hue là đại lượng vòng (0° và 360° đều là đỏ). Trung bình số học thông thường
        sẽ cho kết quả sai (mean(5°, 355°) = 180° thay vì 0°).
      - Circular mean = arctan2(mean(sin θ), mean(cos θ)) cho kết quả đúng.
      - resultant_length = sqrt(mean_cos² + mean_sin²) đo độ "tập trung" màu.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        pixels = hsv.reshape(-1, 3)

    # Hue trong OpenCV: [0, 180) → chuyển sang radian [0, 2π)
    hue_rad = pixels[:, 0] * (2.0 * np.pi / 180.0)
    cos_h = np.cos(hue_rad)
    sin_h = np.sin(hue_rad)

    mean_cos = float(np.mean(cos_h))
    mean_sin = float(np.mean(sin_h))
    resultant = float(np.sqrt(mean_cos ** 2 + mean_sin ** 2))
    # Circular std: sqrt(-2 * ln(R)), R = resultant length
    circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(resultant + 1e-12))))

    mean_sat = float(np.mean(pixels[:, 1]) / 255.0)
    mean_val = float(np.mean(pixels[:, 2]) / 255.0)

    vector = np.array([mean_cos, mean_sin, resultant, circ_std, mean_sat, mean_val], dtype=np.float32)
    return FeatureResult(
        vector,
        {'images': {}, 'plots': {}, 'tables': {'circular_hue': {
            'circular_mean_cos': mean_cos,
            'circular_mean_sin': mean_sin,
            'resultant_length': resultant,
            'circular_std': circ_std,
            'mean_saturation': mean_sat,
            'mean_value': mean_val,
        }}},
        {}
    )
