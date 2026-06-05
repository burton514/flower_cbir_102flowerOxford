from __future__ import annotations

from typing import Callable, Dict, List

from flower_cbir_app.features.base import FeatureSpec
from flower_cbir_app.features.color_features import (
    extract_ccv,
    extract_circular_hue_stats,
    extract_color_moments,
    extract_dominant_colors,
    extract_hsv_hist,
    extract_hue_hist,
    extract_lab_moments,
    extract_radial_color_hist,
    extract_rgb_hist,
)
from flower_cbir_app.features.edge_features import (
    extract_canny_derived,
    extract_edge_orientation_hist,
    extract_sobel_hist,
)
from flower_cbir_app.features.meta_features import (
    extract_centroid_offset,
    extract_foreground_occupancy,
    extract_mask_quality,
)
from flower_cbir_app.features.shape_features import (
    extract_contour_basic,
    extract_fourier_shape,
    extract_geometric_shape,
    extract_hu_moments,
    extract_radial_signature,
    extract_rotational_symmetry,
    extract_symmetry_score,
)
from flower_cbir_app.features.texture_features import extract_glcm, extract_hog, extract_lbp


def get_feature_catalog() -> List[FeatureSpec]:
    return [
        # ── COLOR ───────────────────────────────────────────────────────────────
        # Màu là tín hiệu phân biệt loài hoa mạnh nhất → bật nhóm color làm chủ lực.
        FeatureSpec('hsv_hist',     'HSV Histogram',           'color',   'Histogram màu HSV 16x6x3 = 288 bins.',       'chi_square', True,  '288', is_histogram=True, supports_chi_square=True, default_weight=1.5),
        FeatureSpec('rgb_hist',     'RGB Histogram',           'color',   'Histogram màu RGB 8x8x8.',                   'cosine', False, '512', is_histogram=True, supports_chi_square=True),
        FeatureSpec('hue_hist',     'Hue Histogram',           'color',   'Histogram riêng kênh Hue.',                  'chi_square', True, '36',  is_histogram=True, supports_chi_square=True, default_weight=1.0),
        FeatureSpec('dominant_colors', 'Dominant Colors',      'color',   '5 màu trội + tỉ lệ xuất hiện; feature heuristic, mặc định tắt.', 'cosine', False, '20'),
        FeatureSpec('color_moments','Color Moments',           'color',   'Mean, std, skew trên từng kênh RGB foreground.', 'l2',   True,  '9', default_weight=1.0),
        FeatureSpec('lab_moments',  'Lab Moments',             'color',   'Mean/std trên Lab foreground.',               'l2',     False, '6'),
        # ── COLOR MỚI ────────────────────────────────────────────────────────────
        FeatureSpec('radial_color_hist', 'Radial Color Histogram', 'color', 'Histogram Hue theo 3 vành đồng tâm (tâm→rìa). Dim = rings×hue_bins = 36.', 'chi_square', True, '36', is_histogram=True, supports_chi_square=True, default_weight=1.5),
        FeatureSpec('ccv',           'Color Coherence Vector',   'color',   'Tách pixel màu thành liền khối (coherent) và rải rác (incoherent). Dim = 2×12 = 24.', 'chi_square', False, '24', is_histogram=True, supports_chi_square=True),
        FeatureSpec('circular_hue_stats', 'Circular Hue Stats',  'color',   'Thống kê Hue vòng tròn: circular mean, resultant length, circular std, saturation, value. Dim = 6.', 'l2', True, '6', default_weight=1.0),
        # ── SHAPE ───────────────────────────────────────────────────────────────
        FeatureSpec('hu_moments',   'Hu Moments',              'shape',   '7 Hu moments trên mask nhị phân, sau log transform.', 'l2', True, '7', default_weight=0.8),
        FeatureSpec('geometric_shape','Geometric Shape',       'shape',   'Thuộc tính hình học cơ bản của vùng hoa.',   'l2',     False,  '13'),
        FeatureSpec('contour_basic','Contour Basic',           'shape',   'Độ phức tạp contour và độ lồi cơ bản.',      'l2',     False, '5'),
        FeatureSpec('radial_signature','Radial Signature',     'shape',   'Khoảng cách tâm-biên theo 36 hướng cố định.', 'cosine', False, '36'),
        FeatureSpec('fourier_shape','Fourier Shape Descriptor','shape',   'Magnitude FFT của contour đã resample/center/normalize.', 'cosine', True, '32', default_weight=0.8),
        FeatureSpec('symmetry_score','Symmetry Score',         'shape',   'Độ đối xứng theo overlap foreground với ảnh lật.', 'l2', False, '2'),
        # ── SHAPE MỚI ────────────────────────────────────────────────────────────
        FeatureSpec('rotational_symmetry', 'Rotational Symmetry', 'shape', 'Overlap khi xoay mask theo 8 góc đều nhau → ước lượng số cánh hoa. Dim = 8.', 'cosine', True, '8', default_weight=1.0),
        # ── TEXTURE / GRADIENT ──────────────────────────────────────────────────
        FeatureSpec('lbp',          'LBP Histogram',           'texture', 'LBP uniform P=24, R=3 trên foreground.',      'chi_square', True,  '26', is_histogram=True, supports_chi_square=True, default_weight=1.0),
        FeatureSpec('glcm',         'GLCM Features',           'texture', 'Contrast, homogeneity, energy... từ GLCM foreground.', 'l2', True, '6', default_weight=0.8),
        FeatureSpec('hog',          'HOG',                     'texture', 'Histogram of Oriented Gradients, block norm L2-Hys.', 'cosine', False, '324', is_histogram=True, supports_chi_square=False),
        # ── EDGE ────────────────────────────────────────────────────────────────
        FeatureSpec('edge_orientation_hist','Edge Orientation Histogram','edge','Histogram hướng tại pixel Canny edge.','chi_square',True,'36', is_histogram=True, supports_chi_square=True, default_weight=0.8),
        FeatureSpec('canny_derived','Canny-derived Features',  'edge',    'Thống kê cấu trúc từ edge map Canny.',       'l2',     False, '6'),
        FeatureSpec('sobel_hist',   'Sobel Gradient Histogram','edge',    'Histogram hướng gradient Sobel trên foreground.', 'cosine', False, '36', is_histogram=True, supports_chi_square=True),
        # ── LOCAL (BoVW) ─────────────────────────────────────────────────────────
        # Tắt mặc định: chậm (phải fit vocabulary) và với hoa thường yếu hơn color/shape.
        FeatureSpec('sift_bovw',    'SIFT BoVW',               'local',   'Bag of Visual Words từ SIFT.',               'cosine', False, 'V', is_histogram=True, supports_chi_square=True),
        FeatureSpec('orb_bovw',     'ORB BoVW',                'local',   'BoVW từ ORB; dùng như baseline thực nghiệm cho descriptor nhị phân.', 'cosine', False, 'V', is_histogram=True, supports_chi_square=True),
        FeatureSpec('akaze_bovw',   'AKAZE BoVW',              'local',   'BoVW từ AKAZE; dùng như baseline thực nghiệm cho descriptor nhị phân.', 'cosine', False, 'V', is_histogram=True, supports_chi_square=True),
        FeatureSpec('brisk_bovw',   'BRISK BoVW',              'local',   'BoVW từ BRISK; dùng như baseline thực nghiệm cho descriptor nhị phân.', 'cosine', False, 'V', is_histogram=True, supports_chi_square=True),
        # ── META ─────────────────────────────────────────────────────────────────
        # Feature kiểm tra pipeline, tự động loại khỏi retrieval. Giữ bật để xem trong DB.
        FeatureSpec('foreground_occupancy','Foreground Occupancy','meta', 'Tỉ lệ foreground.',                          'l2',     False, '1', is_meta=True),
        FeatureSpec('centroid_offset','Centroid Offset',       'meta',    'Độ lệch tâm đã chuẩn hóa theo đường chéo ảnh.', 'l2',   False, '1', is_meta=True),
        FeatureSpec('mask_quality', 'Mask Quality Indicators', 'meta',    'Số thành phần, chạm biên, tỉ lệ lỗ trong mask.', 'l2', False, '3', is_meta=True),
    ]


EXTRACTORS: Dict[str, Callable] = {
    'hsv_hist': lambda d: extract_hsv_hist(d['image_rgb'], d['mask']),
    'rgb_hist': lambda d: extract_rgb_hist(d['image_rgb'], d['mask']),
    'hue_hist': lambda d: extract_hue_hist(d['image_rgb'], d['mask']),
    'dominant_colors': lambda d: extract_dominant_colors(d['image_rgb'], d['mask']),
    'color_moments': lambda d: extract_color_moments(d['image_rgb'], d['mask']),
    'lab_moments': lambda d: extract_lab_moments(d['image_rgb'], d['mask']),
    'radial_color_hist': lambda d: extract_radial_color_hist(d['image_rgb'], d['mask']),
    'ccv': lambda d: extract_ccv(d['image_rgb'], d['mask']),
    'circular_hue_stats': lambda d: extract_circular_hue_stats(d['image_rgb'], d['mask']),
    'hu_moments': lambda d: extract_hu_moments(d['mask']),
    'geometric_shape': lambda d: extract_geometric_shape(d['mask']),
    'contour_basic': lambda d: extract_contour_basic(d['mask']),
    'radial_signature': lambda d: extract_radial_signature(d['mask']),
    'fourier_shape': lambda d: extract_fourier_shape(d['mask']),
    'symmetry_score': lambda d: extract_symmetry_score(d['mask']),
    'rotational_symmetry': lambda d: extract_rotational_symmetry(d['mask']),
    'lbp': lambda d: extract_lbp(d['gray'], d['mask']),
    'glcm': lambda d: extract_glcm(d['gray'], d['mask']),
    'hog': lambda d: extract_hog(d['gray'], d['mask']),
    'edge_orientation_hist': lambda d: extract_edge_orientation_hist(d['gray'], d['mask']),
    'canny_derived': lambda d: extract_canny_derived(d['gray'], d['mask']),
    'sobel_hist': lambda d: extract_sobel_hist(d['gray'], d['mask']),
    'foreground_occupancy': lambda d: extract_foreground_occupancy(d['mask']),
    'centroid_offset': lambda d: extract_centroid_offset(d['mask']),
    'mask_quality': lambda d: extract_mask_quality(d['mask']),
}


LOCAL_FEATURE_MAP = {
    'sift_bovw': 'sift',
    'orb_bovw': 'orb',
    'akaze_bovw': 'akaze',
    'brisk_bovw': 'brisk',
}


def get_default_feature_state() -> dict:
    state = {}
    for spec in get_feature_catalog():
        state[spec.key] = {
            'enabled': spec.enabled_by_default,
            'distance': spec.default_distance,
            'weight': float(spec.default_weight),
            'is_meta': spec.is_meta,
            'group': spec.group,
        }
    return state
