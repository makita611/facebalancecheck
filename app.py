import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math

if hasattr(mp, 'solutions'):
    mp_face_mesh = mp.solutions.face_mesh
else:
    import mediapipe.solutions.face_mesh as mp_face_mesh

st.set_page_config(page_title="アートメイク顔面バランス分析", layout="wide")

# ── 定数 ───────────────────────────────────

PHASES = ["術前", "デザイン", "術後"]
PHASE_ICONS = {"術前": "🔵", "デザイン": "✏️", "術後": "✅"}

IRIS_PRESETS = {
    "男性/裸眼 (11.7mm)": 11.7,
    "女性/裸眼 (12.0mm)": 12.0,
    "コンタクト/小 (13.2mm)": 13.2,
    "コンタクト/大 (14.2mm)": 14.2,
    "カスタム入力": None,
}

WEIGHTS = {
    "head_h":    1.0,
    "peak_h":    1.2,
    "tail_h":    0.8,
    "length":    0.8,
    "angle_d":   1.5,
    "head_dist": 0.8,
    "area_pct":  0.8,
}
WEIGHT_LABELS = {
    "head_h": "眉頭の高さ", "peak_h": "眉山の高さ", "tail_h": "眉尻の高さ",
    "length": "眉の長さ",   "angle_d": "傾き角度",  "head_dist": "眉頭距離",
    "area_pct": "面積差",
}

FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
             397,365,379,378,400,377,152,148,176,149,150,136,
             172,58,132,93,234,127,162,21,54,103,67,109]

# ── 唇専用定数 ─────────────────────────────

WEIGHTS_LIP = {
    "corner_h":   1.5,   # 口角の高さ差
    "cupid_h":    1.2,   # キューピッドボウの高さ差
    "lower_h":    0.8,   # 下唇の高さ差
    "width_diff": 0.8,   # 唇幅の左右差
    "angle_d":    2.0,   # 唇の傾き角
    "area_pct":   0.8,   # 上唇面積差
}
LIP_WEIGHT_LABELS = {
    "corner_h":   "口角の高さ",
    "cupid_h":    "キューピッドボウ",
    "lower_h":    "下唇の高さ",
    "width_diff": "唇幅の左右差",
    "angle_d":    "唇の傾き",
    "area_pct":   "上唇面積差",
}
# 唇輪郭（上唇の左右ハーフ面積計算用）
LIP_UPPER_R = [0, 37, 39, 40, 185, 61, 62, 78, 82, 81, 80, 191, 13]
LIP_UPPER_L = [0, 267, 269, 270, 409, 291, 292, 308, 312, 311, 310, 415, 13]

FACE_SHAPE_ADVICE = {
    "卵型":    "理想的なバランスの顔型です。多くのデザインが似合います。眉山を虹彩外縁に揃えるだけでさらに洗練されます。",
    "丸顔":    "眉にアーチをつけ縦のラインを強調することで輪郭が引き締まります。眉山を高めに設定し、眉尻をわずかに上げるデザインを推奨します。",
    "面長":    "水平ラインを強調する平行眉が効果的です。眉の傾きを抑えたストレート眉で顔の縦長感が緩和されます。",
    "ベース型": "緩やかなアーチ眉で視線を上方に誘導することでエラの印象が和らぎます。眉山を外側に設定すると重心が上がります。",
    "四角顔":  "適度なアーチと細めのデザインで輪郭を柔らかく見せます。眉山をなだらかにして角張りを緩和してください。",
    "逆三角形": "眉を強調しすぎないことが重要です。自然なアーチで程よいボリュームを持たせ、顔全体のバランスを整えます。",
}

FACE_SHAPE_COLORS = {
    "卵型":    "#4CAF50",
    "丸顔":    "#2196F3",
    "面長":    "#9C27B0",
    "ベース型": "#FF9800",
    "四角顔":  "#F44336",
    "逆三角形": "#00BCD4",
}

# (bg色, 文字色) のタプル。ブラック以外は文字を黒に統一
BG_OPTIONS = {
    "デフォルト":    ("", ""),
    "ホワイト":      ("#ffffff", "#111111"),
    "ブラック":      ("#111111", "#f0f0f0"),
    "グレー":        ("#c4c4c4", "#111111"),
    "ソフトベージュ": ("#c8a87a", "#111111"),
}


def get_theme_css(bg, text):
    """罫線・入力欄・セレクトボックスを含む包括的テーマCSS"""
    if not bg:
        return ""
    is_dark   = (bg == "#111111")
    input_bg  = "#2d2d2d" if is_dark else "#ffffff"
    border_c  = "rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.20)"
    return f"""<style>
    /* ── 背景 ── */
    .stApp, .main .block-container,
    section[data-testid="stSidebar"] {{
        background-color: {bg} !important;
    }}
    /* ── テキスト全般 ── */
    .stApp p, .stApp span, .stApp li,
    .stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
    .stApp label, .stMarkdown, .stMarkdown *,
    .stCaption, [data-testid="stText"],
    [data-testid="stMetricValue"],[data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"],
    .stRadio label, .stSelectbox label, .stCheckbox label,
    section[data-testid="stSidebar"] * {{
        color: {text} !important;
    }}
    /* ── 入力フィールド ── */
    input, textarea,
    div[data-baseweb="input"] input,
    .stTextInput input, .stNumberInput input {{
        background-color: {input_bg} !important;
        color: {text} !important;
        border-color: {border_c} !important;
    }}
    /* ── セレクトボックス ── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] [class*="ValueContainer"],
    div[data-baseweb="select"] [class*="SingleValue"],
    div[data-baseweb="select"] [class*="placeholder"] {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}
    /* ドロップダウンメニュー */
    div[data-baseweb="popover"], div[data-baseweb="popover"] *,
    ul[role="listbox"], ul[role="listbox"] li {{
        background-color: {input_bg} !important;
        color: {text} !important;
    }}
    /* ── 罫線 ── */
    hr {{ border-color: {border_c} !important; }}
    [data-testid="stDivider"] {{
        background: {border_c} !important;
    }}
    /* ── Expander ── */
    [data-testid="stExpander"] {{
        border-color: {border_c} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {text} !important;
    }}
    /* ── スライダー ── */
    [data-testid="stSlider"] * {{ color: {text} !important; }}
    </style>"""

R_EYE_CONTOUR = [33,160,159,158,157,173,133,155,154,153,145,144]
L_EYE_CONTOUR = [263,387,386,385,384,398,362,382,381,380,374,373]

# ── ランドマーク再利用モック ──────────────

class _LM:
    __slots__ = ("x", "y")
    def __init__(self, x, y): self.x, self.y = x, y

class MockLandmarks:
    """lm_raw dict から MediaPipe landmark オブジェクトを再構築"""
    def __init__(self, raw):
        self.landmark = {k: _LM(v["x"], v["y"]) for k, v in raw.items()}

# ── FaceMesh (キャッシュ) ──────────────────

@st.cache_resource
def get_face_mesh():
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

# ── ユーティリティ ─────────────────────────

def _detect(img, fm):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    return res.multi_face_landmarks[0] if res.multi_face_landmarks else None


def _center_shift(img, lm, idx_list, w, h, fm):
    avg_x = sum(lm.landmark[i].x for i in idx_list) / len(idx_list)
    sx = w // 2 - int(avg_x * w)
    if abs(sx) > 1:
        img = cv2.warpAffine(img, np.float32([[1,0,sx],[0,1,0]]), (w, h))
        new = _detect(img, fm)
        if new:
            lm = new
    return img, lm

# ── 顔クロップ ─────────────────────────────

def crop_face_region(img, fm, pad=0.30, target=640):
    h, w = img.shape[:2]
    lm = _detect(img, fm)
    if lm is None:
        return img
    xs = [lm.landmark[i].x * w for i in FACE_OVAL]
    ys = [lm.landmark[i].y * h for i in FACE_OVAL]
    fw, fh = max(xs) - min(xs), max(ys) - min(ys)
    x1 = max(0, int(min(xs) - fw * pad))
    y1 = max(0, int(min(ys) - fh * pad * 0.5))
    x2 = min(w, int(max(xs) + fw * pad))
    y2 = min(h, int(max(ys) + fh * pad))
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    if cw > 0 and ch > 0:
        r = target / max(cw, ch)
        crop = cv2.resize(crop, (int(cw*r), int(ch*r)), interpolation=cv2.INTER_AREA)
    return crop

# ── 傾き補正 (3モード) ────────────────────

def process_pupil(img, fm):
    h, w = img.shape[:2]
    lm = _detect(img, fm)
    if lm is None:
        return None, None, 0.0
    rp, lp = lm.landmark[468], lm.landmark[473]
    angle = math.degrees(math.atan2((lp.y-rp.y)*h, (lp.x-rp.x)*w))
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0),
                         (w, h), flags=cv2.INTER_CUBIC)
    lm = _detect(img, fm)
    if lm is None:
        return None, None, 0.0
    img, lm = _center_shift(img, lm, [8], w, h, fm)
    return img, lm, angle


def process_skeleton(img, fm):
    h, w = img.shape[:2]
    lm = _detect(img, fm)
    if lm is None:
        return None, None, 0.0
    nose, chin = lm.landmark[8], lm.landmark[152]
    skew = math.degrees(math.atan2((chin.x-nose.x)*w, (chin.y-nose.y)*h))
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2,h//2), -skew, 1.0),
                         (w, h), flags=cv2.INTER_CUBIC)
    lm = _detect(img, fm)
    if lm is None:
        return None, None, 0.0
    img, lm = _center_shift(img, lm, [234, 454], w, h, fm)
    return img, lm, skew


def process_midface(img, fm):
    """中顔面基準: 瞳孔で回転補正 → 眉間(#168)・鼻先(#1)・鼻下(#164)の平均を中央に"""
    h, w = img.shape[:2]
    lm = _detect(img, fm)
    if lm is None:
        return None, None, 0.0
    rp, lp = lm.landmark[468], lm.landmark[473]
    angle = math.degrees(math.atan2((lp.y-rp.y)*h, (lp.x-rp.x)*w))
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0),
                         (w, h), flags=cv2.INTER_CUBIC)
    lm = _detect(img, fm)
    if lm is None:
        return None, None, 0.0
    img, lm = _center_shift(img, lm, [168, 1, 164], w, h, fm)
    return img, lm, angle

# ── 顔型判定 ──────────────────────────────

def classify_face_shape(lm, w, h):
    def d(a, b):
        return math.hypot((lm.landmark[a].x-lm.landmark[b].x)*w,
                          (lm.landmark[a].y-lm.landmark[b].y)*h)
    face_h  = d(10, 152)
    cheek_w = d(234, 454)
    jaw_w   = d(172, 397)
    fore_w  = d(103, 332)
    if cheek_w == 0:
        return "卵型"
    h_r    = face_h  / cheek_w
    jaw_r  = jaw_w   / cheek_w
    fore_r = fore_w  / cheek_w
    if   h_r > 1.55:                          return "面長"
    elif h_r < 1.15 and jaw_r > 0.80:         return "丸顔"
    elif jaw_r > 0.88 and fore_r > 0.80:      return "四角顔"
    elif jaw_r > 0.85 and fore_r < 0.78:      return "ベース型"
    elif fore_r > jaw_r + 0.08:               return "逆三角形"
    else:                                     return "卵型"

# ── 眉の計測 ──────────────────────────────

def analyze_eyebrows(lm, w, h, axis_mode, override_cx=None):
    def lx(i): return lm.landmark[i].x * w
    def ly(i): return lm.landmark[i].y * h
    ipd = abs((lm.landmark[473].x - lm.landmark[468].x) * w)
    def nu(px): return (px / ipd * 100) if ipd > 0 else 0.0

    if override_cx is not None:
        cx = override_cx
    elif axis_mode == "瞳孔基準":
        cx = lx(8)
    elif axis_mode == "中顔面基準（鼻筋）":
        cx = (lx(168) + lx(1) + lx(164)) / 3
    else:
        cx = (lx(234) + lx(454)) / 2

    r_hx, r_tx = lx(285), lx(300)
    l_hx, l_tx = lx(55),  lx(70)
    r_hy, r_py, r_ty = ly(285), ly(296), ly(300)
    l_hy, l_py, l_ty = ly(55),  ly(66),  ly(70)
    r_len = abs(r_tx - r_hx)
    l_len = abs(l_tx - l_hx)

    def ang(hx, tx, hy, ty):
        dx = abs(tx - hx)
        return math.degrees(math.atan2(hy-ty, dx)) if dx > 0 else 0.0

    def brow_area(pts):
        return float(cv2.contourArea(np.array([[(int(lx(p)),int(ly(p)))] for p in pts])))

    r_area = brow_area([285,296,334,293,300,276,283,282,295])
    l_area = brow_area([55,66,105,63,70,46,53,52,65])
    ap = abs(r_area-l_area)/max(r_area,l_area)*100 if r_area>0 and l_area>0 else 0.0

    return {
        "ipd_px": ipd, "center_x": cx,
        "r_head_y": r_hy, "l_head_y": l_hy,
        "r_peak_y": r_py, "l_peak_y": l_py,
        "r_tail_y": r_ty, "l_tail_y": l_ty,
        "r_length": r_len, "l_length": l_len,
        "r_angle": ang(r_hx,r_tx,r_hy,r_ty),
        "l_angle": ang(l_hx,l_tx,l_hy,l_ty),
        "r_area": r_area, "l_area": l_area,
        "head_h":    nu(r_hy - l_hy),
        "peak_h":    nu(r_py - l_py),
        "tail_h":    nu(r_ty - l_ty),
        "length":    nu(r_len - l_len),
        "angle_d":   ang(r_hx,r_tx,r_hy,r_ty) - ang(l_hx,l_tx,l_hy,l_ty),
        "head_dist": nu(abs(r_hx-cx) - abs(l_hx-cx)),
        "area_pct":  ap,
    }

# ── 虹彩スケール ──────────────────────────

def calc_iris_scale(lm, w, h, iris_mm):
    def diam(a, b, c, d):
        return ((abs(lm.landmark[a].x-lm.landmark[b].x)*w
                +abs(lm.landmark[c].y-lm.landmark[d].y)*h) / 2)
    avg = (diam(469,471,470,472) + diam(474,476,475,477)) / 2
    return avg / iris_mm if avg > 0 else None

# ── スコア計算 ────────────────────────────

def calc_score(m):
    bd = {k: (m[k] if k == "area_pct" else abs(m[k])) * wt
          for k, wt in WEIGHTS.items()}
    score = max(0, 100 - int(sum(bd.values())))
    grade = ("Excellent" if score>=90 else "Good" if score>=80 else
             "Normal" if score>=70 else "Bad" if score>=50 else "Very Bad")
    return score, grade, bd

# ── 目元分析 ──────────────────────────────

def analyze_eyes(lm, w, h, ppm):
    def lx(i): return lm.landmark[i].x * w
    def ly(i): return lm.landmark[i].y * h
    def d(a, b): return math.hypot(lx(a)-lx(b), ly(a)-ly(b))

    r_width  = d(33, 133)
    l_width  = d(263, 362)
    r_height = d(159, 145)
    l_height = d(386, 374)
    r_area = float(cv2.contourArea(np.array([[(int(lx(i)),int(ly(i)))] for i in R_EYE_CONTOUR])))
    l_area = float(cv2.contourArea(np.array([[(int(lx(i)),int(ly(i)))] for i in L_EYE_CONTOUR])))

    def pct(a, b): return abs(a-b)/max(a,b)*100 if max(a,b) > 0 else 0.0
    def fmt_mm(px): return f"{px/ppm:.2f}mm" if ppm else f"{px:.1f}px"
    def fmt_mm2(px2): return f"{px2/(ppm**2):.2f}mm²" if ppm else f"{px2:.0f}px²"

    return {
        "r_width": r_width,   "l_width": l_width,
        "r_height": r_height, "l_height": l_height,
        "r_area": r_area,     "l_area": l_area,
        "width_pct":  pct(r_width, l_width),
        "height_pct": pct(r_height, l_height),
        "area_pct":   pct(r_area, l_area),
        "r_width_mm":  fmt_mm(r_width),  "l_width_mm":  fmt_mm(l_width),
        "r_height_mm": fmt_mm(r_height), "l_height_mm": fmt_mm(l_height),
        "r_area_mm":   fmt_mm2(r_area),  "l_area_mm":   fmt_mm2(l_area),
    }

# ── 定型文アドバイス ──────────────────────

def _sev(v, mild=2.0, strong=5.0):
    return "ok" if abs(v) < mild else ("warn" if abs(v) < strong else "error")

def generate_advice(m, ppm=None):
    items = []
    def disp(nu):
        return f"{abs(nu)*m['ipd_px']/100/ppm:.2f}mm" if ppm else f"{abs(nu):.1f}NU"
    def hi(v): return "右" if v > 0 else "左"
    def lo(v): return "左" if v > 0 else "右"
    def add(k, s, t, b): items.append({"key":k,"sev":s,"title":t,"body":b})

    hh = m["head_h"]
    if _sev(hh) == "ok":
        add("head_h","ok","眉頭の高さ","左右の眉頭はほぼ均一な高さです。")
    else:
        adj = "0.3〜0.5" if _sev(hh) == "warn" else "0.5〜1.0"
        add("head_h",_sev(hh),"眉頭の高さ",
            f"{hi(hh)}の眉頭が{lo(hh)}より{disp(hh)}低位にあります。"
            f"{hi(hh)}側の眉頭スタートを{adj}mm挙上することを推奨します。")

    ph = m["peak_h"]
    if _sev(ph) == "ok":
        add("peak_h","ok","眉山の高さ","左右の眉山は均一な高さです。")
    else:
        add("peak_h",_sev(ph),"眉山の高さ",
            f"{hi(ph)}の眉山が{lo(ph)}より{disp(ph)}低位にあります。"
            f"アーチのピーク位置が非対称です。{hi(ph)}側の眉山を挙上調整してください。")

    th = m["tail_h"]
    if _sev(th) == "ok":
        add("tail_h","ok","眉尻の高さ","左右の眉尻はほぼ均一な高さです。")
    else:
        add("tail_h",_sev(th),"眉尻の高さ",
            f"{hi(th)}の眉尻が{lo(th)}より{disp(th)}低位にあります。"
            f"{hi(th)}側の眉尻着地点を上方修正することでシンメトリーが改善します。")

    ad = m["angle_d"]
    if abs(ad) < 1.0:
        add("angle","ok","眉の傾き","左右の眉の傾きはほぼ同一です。")
    elif abs(ad) < 3.0:
        add("angle","warn","眉の傾き",
            f"{'右' if ad>0 else '左'}眉の傾斜が{'左' if ad>0 else '右'}より{abs(ad):.1f}°大きく、"
            f"より急な上がり眉になっています。"
            f"{'右' if ad>0 else '左'}側の眉尻を0.3〜0.5mm下方調整することで角度を揃えられます。")
    else:
        add("angle","error","眉の傾き",
            f"{'右' if ad>0 else '左'}眉が{'左' if ad>0 else '右'}より{abs(ad):.1f}°と著明な傾き差があります。"
            f"骨格由来の非対称が疑われます。全体角度の再設計を強く推奨します。")

    ld = m["length"]
    sev = _sev(ld, 3.0, 7.0)
    if sev == "ok":
        add("length","ok","眉の長さ","左右の眉の長さはほぼ均等です。")
    else:
        add("length",sev,"眉の長さ",
            f"{'右' if ld>0 else '左'}眉が{'左' if ld>0 else '右'}より{disp(ld)}長い状態です。"
            f"{'右' if ld>0 else '左'}側の眉尻を短縮するか、"
            f"{'左' if ld>0 else '右'}側を延長してバランスを整えてください。")

    hd = m["head_dist"]
    sev = _sev(hd, 3.0, 6.0)
    if sev == "ok":
        add("head_dist","ok","眉頭の左右対称","両眉頭は正中線からほぼ等距離です。")
    else:
        add("head_dist",sev,"眉頭の左右対称",
            f"{'右' if hd>0 else '左'}の眉頭が正中線から{disp(hd)}遠位にあります。"
            f"{'右' if hd>0 else '左'}眉の眉頭スタートを内側に移動させると対称性が改善します。")

    ap = m["area_pct"]
    if ap < 5.0:
        add("area","ok","眉のボリューム","左右の眉のボリュームはほぼ均一です。")
    elif ap < 15.0:
        add("area","warn","眉のボリューム",
            f"左右の眉のボリュームに{ap:.1f}%の差があります。"
            f"太さ・密度の調整で視覚的な均一感を高めることを推奨します。")
    else:
        add("area","error","眉のボリューム",
            f"左右の眉のボリュームに{ap:.1f}%と顕著な差があります。"
            f"細い側への増毛(ストローク追加)か、太い側の修正を検討してください。")

    return items

# ── 唇の計測 ──────────────────────────────

def analyze_lips(lm, w, h, axis_mode, override_cx=None):
    """
    唇の左右差を計測（眉と同構造）
      corner_h  : 口角の高さ差 NU  (+右が低い)
      cupid_h   : キューピッドボウ峰の高さ差 NU
      lower_h   : 下唇の高さ差 NU
      width_diff: 唇幅の左右差 NU  (+右ハーフが広い)
      angle_d   : 唇の傾き角度 ° (commissure 線の水平からのずれ)
      area_pct  : 上唇左右面積差 %
    """
    def lx(i): return lm.landmark[i].x * w
    def ly(i): return lm.landmark[i].y * h
    ipd = abs((lm.landmark[473].x - lm.landmark[468].x) * w)
    def nu(px): return (px / ipd * 100) if ipd > 0 else 0.0

    if override_cx is not None:
        cx = override_cx
    elif axis_mode == "瞳孔基準":
        cx = lx(8)
    elif axis_mode == "中顔面基準（鼻筋）":
        cx = (lx(168) + lx(1) + lx(164)) / 3
    else:
        cx = (lx(234) + lx(454)) / 2

    # 口角 (commissure)  右=61, 左=291
    r_cx, r_cy = lx(61),  ly(61)
    l_cx, l_cy = lx(291), ly(291)
    # キューピッドボウ峰  右=37, 左=267
    r_kx, r_ky = lx(37),  ly(37)
    l_kx, l_ky = lx(267), ly(267)
    # 下唇               右=91, 左=321
    r_lx, r_ly = lx(91),  ly(91)
    l_lx, l_ly = lx(321), ly(321)

    r_half_w = abs(cx - r_cx)
    l_half_w = abs(l_cx - cx)

    lip_angle = math.degrees(math.atan2(l_cy - r_cy, l_cx - r_cx))

    def lip_area(pts):
        arr = np.array([[(int(lx(p)), int(ly(p)))] for p in pts])
        return float(cv2.contourArea(arr))

    r_area = lip_area(LIP_UPPER_R)
    l_area = lip_area(LIP_UPPER_L)
    area_pct = abs(r_area - l_area) / max(r_area, l_area) * 100 if max(r_area, l_area) > 0 else 0.0

    return {
        "ipd_px": ipd, "center_x": cx,
        # raw (表示用)
        "r_corner_y": r_cy, "l_corner_y": l_cy,
        "r_cupid_y":  r_ky, "l_cupid_y":  l_ky,
        "r_lower_y":  r_ly, "l_lower_y":  l_ly,
        "r_half_w":   r_half_w, "l_half_w": l_half_w,
        "r_upper_area": r_area, "l_upper_area": l_area,
        "lip_angle":  lip_angle,
        # diffs
        "corner_h":   nu(r_cy - l_cy),
        "cupid_h":    nu(r_ky - l_ky),
        "lower_h":    nu(r_ly - l_ly),
        "width_diff": nu(r_half_w - l_half_w),
        "angle_d":    lip_angle,
        "area_pct":   area_pct,
    }


def calc_score_lip(m):
    bd = {k: (m[k] if k == "area_pct" else abs(m[k])) * wt
          for k, wt in WEIGHTS_LIP.items()}
    score = max(0, 100 - int(sum(bd.values())))
    grade = ("Excellent" if score>=90 else "Good" if score>=80 else
             "Normal" if score>=70 else "Bad" if score>=50 else "Very Bad")
    return score, grade, bd


def generate_lip_advice(m, ppm=None):
    items = []
    def disp(nu):
        return f"{abs(nu)*m['ipd_px']/100/ppm:.2f}mm" if ppm else f"{abs(nu):.1f}NU"
    def hi(v): return "右" if v > 0 else "左"
    def lo(v): return "左" if v > 0 else "右"
    def _sev(v, mild=2.0, strong=5.0):
        return "ok" if abs(v)<mild else ("warn" if abs(v)<strong else "error")
    def add(k, s, t, b): items.append({"key":k,"sev":s,"title":t,"body":b})

    # 口角の高さ
    ch = m["corner_h"]
    if _sev(ch) == "ok":
        add("corner_h","ok","口角の高さ","左右の口角はほぼ均一な高さです。")
    else:
        adj = "0.3〜0.5" if _sev(ch)=="warn" else "0.5〜1.0"
        add("corner_h",_sev(ch),"口角の高さ",
            f"{hi(ch)}の口角が{lo(ch)}より{disp(ch)}低位にあります。"
            f"{hi(ch)}側の口角を{adj}mm挙上するデザイン調整を推奨します。")

    # キューピッドボウ
    kh = m["cupid_h"]
    if _sev(kh) == "ok":
        add("cupid_h","ok","キューピッドボウ","左右のキューピッドボウ峰はほぼ均一な高さです。")
    else:
        add("cupid_h",_sev(kh),"キューピッドボウ",
            f"{hi(kh)}のキューピッドボウ峰が{lo(kh)}より{disp(kh)}低位にあります。"
            f"{hi(kh)}側の山を上方修正し、シンメトリーなボウラインを形成してください。")

    # 下唇の高さ
    lh = m["lower_h"]
    if _sev(lh) == "ok":
        add("lower_h","ok","下唇の高さ","左右の下唇はほぼ均一な高さです。")
    else:
        add("lower_h",_sev(lh),"下唇の高さ",
            f"{hi(lh)}の下唇が{lo(lh)}より{disp(lh)}低位にあります。"
            f"{hi(lh)}側の下唇ラインを上方修正することで均一感が改善します。")

    # 唇の傾き
    ad = m["angle_d"]
    if abs(ad) < 1.0:
        add("angle_d","ok","唇の傾き","唇の水平ラインはほぼ均一です。")
    elif abs(ad) < 3.0:
        add("angle_d","warn","唇の傾き",
            f"唇が{'左' if ad>0 else '右'}下がりに{abs(ad):.1f}°傾いています。"
            f"{'右' if ad>0 else '左'}側の口角を僅かに下げるか、{'左' if ad>0 else '右'}側を上げて水平を揃えてください。")
    else:
        add("angle_d","error","唇の傾き",
            f"唇が{'左' if ad>0 else '右'}下がりに{abs(ad):.1f}°と著明な傾きがあります。"
            f"骨格由来の非対称が疑われます。全体のデザインラインを再設計することを推奨します。")

    # 唇幅の左右差
    wd = m["width_diff"]
    sev = _sev(wd, 3.0, 7.0)
    if sev == "ok":
        add("width_diff","ok","唇の左右バランス","唇の左右幅はほぼ均等です。")
    else:
        add("width_diff",sev,"唇の左右バランス",
            f"{'右' if wd>0 else '左'}側の唇幅が{disp(wd)}広い状態です。"
            f"{'右' if wd>0 else '左'}側の口角位置を内側に調整するとバランスが整います。")

    # 面積差
    ap = m["area_pct"]
    if ap < 5:
        add("area_pct","ok","上唇のボリューム","左右の上唇のボリュームはほぼ均一です。")
    elif ap < 15:
        add("area_pct","warn","上唇のボリューム",
            f"左右の上唇ボリュームに{ap:.1f}%の差があります。"
            f"色の入り・ライン幅の調整で視覚的均一感を高めてください。")
    else:
        add("area_pct","error","上唇のボリューム",
            f"左右の上唇ボリュームに{ap:.1f}%と顕著な差があります。"
            f"ライン設計を見直し、細い側に厚みを持たせるデザイン修正を推奨します。")

    return items


def draw_lip_guidelines(img, lm, metrics, score, grade, angle, axis_mode):
    """唇専用ガイドライン描画（緑=右唇 / 紫=左唇）"""
    h, w = img.shape[:2]
    draw = img.copy()
    ov   = draw.copy()
    C_W    = (255, 255, 255)
    C_LR   = (0,  200,  0)    # 緑: 右唇 (患者の右)
    C_LL   = (200,  0, 200)   # 紫: 左唇 (患者の左)
    C_G    = (130, 130, 130)

    def lxp(i): return int(lm.landmark[i].x * w)
    def lyp(i): return int(lm.landmark[i].y * h)

    cx = int(metrics["center_x"])

    # 正中線
    cv2.line(draw, (cx, 0), (cx, h), C_W, 1)

    # 口角の垂直・水平ライン
    cv2.line(draw, (lxp(61),  0), (lxp(61),  h), C_LR, 1)
    cv2.line(draw, (lxp(291), 0), (lxp(291), h), C_LL, 1)
    cv2.line(draw, (0, lyp(61)),  (w, lyp(61)),  C_LR, 1)
    cv2.line(draw, (0, lyp(291)), (w, lyp(291)), C_LL, 1)

    # キューピッドボウ峰の水平ライン
    cv2.line(draw, (0, lyp(37)),  (w, lyp(37)),  C_LR, 1)
    cv2.line(draw, (0, lyp(267)), (w, lyp(267)), C_LL, 1)

    # 下唇の水平ライン (薄く)
    cv2.line(ov, (0, lyp(91)),  (w, lyp(91)),  C_LR, 1)
    cv2.line(ov, (0, lyp(321)), (w, lyp(321)), C_LL, 1)
    cv2.addWeighted(ov, 0.3, draw, 0.7, 0, draw)

    # 情報ボックス
    cv2.rectangle(draw, (5, 5), (300, 70), (0, 0, 0), -1)
    cv2.putText(draw, f"Lip: {score}/100 [{grade}]",
                (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 128), 1)
    cv2.putText(draw, f"Tilt: {angle:.2f} deg",
                (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_W, 1)
    cv2.putText(draw, "R.Lip", (165, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_LR, 1)
    cv2.putText(draw, "L.Lip", (230, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_LL, 1)

    return cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)


# ── 描画 ─────────────────────────────────

def draw_guidelines(img, lm, metrics, score, grade, angle, axis_mode,
                    show_golden=False, face_shape=None):
    h, w = img.shape[:2]
    draw = img.copy()
    ov   = draw.copy()
    C_W  = (255,255,255)
    C_R  = (0,165,255)
    C_L  = (255,255,0)
    C_G  = (130,130,130)
    C_GR = (180,105,255)

    def lxp(i): return int(lm.landmark[i].x * w)
    def lyp(i): return int(lm.landmark[i].y * h)

    cx = int(metrics["center_x"])
    py = int((lm.landmark[468].y + lm.landmark[473].y) / 2 * h)

    if axis_mode == "骨格基準（鼻根-顎先）":
        cv2.line(draw,(lxp(8),lyp(8)),(lxp(152),lyp(152)),C_G,1)
        cv2.line(draw,(lxp(234),0),(lxp(234),h),C_G,1)
        cv2.line(draw,(lxp(454),0),(lxp(454),h),C_G,1)
    elif axis_mode == "中顔面基準（鼻筋）":
        for i in [168, 1, 164]:
            cv2.circle(draw,(lxp(i),lyp(i)),3,C_G,-1)

    cv2.line(draw,(cx,0),(cx,h),C_W,1)
    cv2.line(draw,(0,py),(w,py),C_W,1)

    for i in [285,296,300]: cv2.line(draw,(lxp(i),0),(lxp(i),h),C_R,1)
    for i in [55,66,70]:    cv2.line(draw,(lxp(i),0),(lxp(i),h),C_L,1)
    for i in [285,295,296]: cv2.line(draw,(0,lyp(i)),(w,lyp(i)),C_R,1)
    for i in [55,65,66]:    cv2.line(draw,(0,lyp(i)),(w,lyp(i)),C_L,1)

    cv2.line(ov,(0,lyp(300)),(w,lyp(300)),C_R,1)
    cv2.line(ov,(0,lyp(70)),(w,lyp(70)),C_L,1)
    cv2.addWeighted(ov,0.3,draw,0.7,0,draw)

    if show_golden:
        avg_hy = int((lyp(285)+lyp(55))/2)
        avg_py = int((lyp(296)+lyp(66))/2)
        avg_ty = int((lyp(300)+lyp(70))/2)
        S = 10
        def cross(x, y):
            cv2.line(draw,(x,y-S),(x,y+S),C_GR,2)
            cv2.line(draw,(x-S,y),(x+S,y),C_GR,2)
        cross(lxp(102), avg_hy); cross(lxp(331), avg_hy)
        cross(lxp(471), avg_py); cross(lxp(476), avg_py)
        for alar, canthus in [(102,133),(331,362)]:
            ax,ay = lxp(alar),lyp(alar)
            bx,by = lxp(canthus),lyp(canthus)
            dy = by - ay
            if dy != 0:
                t = (avg_ty - ay) / dy
                cross(int(ax + t*(bx-ax)), avg_ty)
        cv2.putText(draw,"+ Golden Ratio",(5,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,C_GR,1)

    box_h = 100 if face_shape else 85
    cv2.rectangle(draw,(5,5),(300,box_h),(0,0,0),-1)
    cv2.putText(draw,f"Score: {score}/100 [{grade}]",
                (10,26),cv2.FONT_HERSHEY_DUPLEX,0.55,(0,255,0),1)
    ax_lbl = ("PUPIL" if axis_mode=="瞳孔基準" else
              "MIDFACE" if axis_mode=="中顔面基準（鼻筋）" else "SKELETON")
    cv2.putText(draw,f"Axis:{ax_lbl} Tilt:{angle:.1f}d",
                (10,47),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_W,1)
    cv2.putText(draw,"Right Brow",(165,47),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_R,1)
    cv2.putText(draw,"Left Brow", (165,68),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_L,1)
    cv2.putText(draw,f"Tilt: {angle:.2f} deg",(10,68),cv2.FONT_HERSHEY_SIMPLEX,0.38,C_W,1)
    if face_shape:
        cv2.putText(draw,f"Face: {face_shape}",
                    (10,88),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,255),1)

    return cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# ── 分析パイプライン ──────────────────────

def run_analysis(img_bgr, fm, axis_mode, iris_mm, show_golden, do_eye=False):
    img = crop_face_region(img_bgr, fm)
    # クロップ済み画像をバイト保存（傾き調整の再利用用）
    _, _buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    crop_bytes = _buf.tobytes()

    if axis_mode == "瞳孔基準":
        proc, lm, angle = process_pupil(img, fm)
    elif axis_mode == "中顔面基準（鼻筋）":
        proc, lm, angle = process_midface(img, fm)
    else:
        proc, lm, angle = process_skeleton(img, fm)
    if lm is None:
        return None
    h2, w2 = proc.shape[:2]

    # ガイドライン描画前の処理済み画像をバイト保存（中心線調整用）
    _, _buf2 = cv2.imencode('.jpg', proc, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    proc_bytes = _buf2.tobytes()

    m            = analyze_eyebrows(lm, w2, h2, axis_mode)
    score, grade, bd = calc_score(m)
    ppm          = calc_iris_scale(lm, w2, h2, iris_mm) if iris_mm else None
    face_shape   = classify_face_shape(lm, w2, h2)
    eye_metrics  = analyze_eyes(lm, w2, h2, ppm) if do_eye else None

    # 唇分析（常に計算）
    lm_lip        = analyze_lips(lm, w2, h2, axis_mode)
    lip_score, lip_grade, lip_bd = calc_score_lip(lm_lip)

    return {
        # ── 眉 ──
        "draw_rgb":    draw_guidelines(proc,lm,m,score,grade,angle,axis_mode,show_golden,face_shape),
        "metrics":     m,
        "score":       score,
        "grade":       grade,
        "breakdown":   bd,
        "advice":      generate_advice(m, ppm),
        # ── 唇 ──
        "lip_draw_rgb":  draw_lip_guidelines(proc,lm,lm_lip,lip_score,lip_grade,angle,axis_mode),
        "lip_metrics":   lm_lip,
        "lip_score":     lip_score,
        "lip_grade":     lip_grade,
        "lip_breakdown": lip_bd,
        "lip_advice":    generate_lip_advice(lm_lip, ppm),
        # ── 共通 ──
        "ppm":         ppm,
        "angle":       angle,
        "face_shape":  face_shape,
        "eye_metrics": eye_metrics,
        "lm_raw":      {i:{"x":lm.landmark[i].x,"y":lm.landmark[i].y}
                        for i in range(len(lm.landmark))},
        "img_shape":   (h2, w2),
        "crop_bytes":  crop_bytes,
        "proc_bytes":  proc_bytes,
    }

# ── 結果表示 ──────────────────────────────

def show_result(res, phase_label, adj_params=None):
    ppm = res["ppm"]
    fs  = res.get("face_shape", "")

    # ── パーツ切り替えトグル ──────────────────
    part = st.radio(
        "分析パーツ",
        ["👁 眉（Eyebrow）", "💋 唇（Lip）"],
        horizontal=True,
        key=f"part_{phase_label}",
    )
    is_lip = "唇" in part

    # 表示するデータを切り替え
    if is_lip:
        m      = res["lip_metrics"]
        score  = res["lip_score"]
        grade  = res["lip_grade"]
        bd     = res["lip_breakdown"]
        advice = res["lip_advice"]
        img    = res["lip_draw_rgb"]
        wl     = WEIGHTS_LIP
        wll    = LIP_WEIGHT_LABELS
    else:
        m      = res["metrics"]
        score  = res["score"]
        grade  = res["grade"]
        bd     = res["breakdown"]
        advice = res["advice"]
        img    = res["draw_rgb"]
        wl     = WEIGHTS
        wll    = WEIGHT_LABELS

    def nu_to_mm(nu):
        return f"{abs(nu)*m['ipd_px']/100/ppm:.2f}mm" if ppm else "—"

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(img, use_container_width=True)
        if adj_params:
            show_landmark_adjustment(res, **adj_params)
    with col2:
        g = {"Excellent":"🟢","Good":"🟡","Normal":"🟠",
             "Bad":"🔴","Very Bad":"⚫"}.get(grade,"⚪")
        label = "唇スコア" if is_lip else "眉スコア"
        st.metric(label, f"{score} / 100", delta=f"{g} {grade}")
        if ppm:
            st.caption(f"スケール: {ppm:.1f} px/mm（虹彩基準）")

        # 顔型バッジは眉タブのみ表示
        if not is_lip and fs:
            color = FACE_SHAPE_COLORS.get(fs, "#888888")
            st.markdown(
                f'<span style="background-color:{color};color:white;padding:3px 12px;'
                f'border-radius:12px;font-weight:bold;font-size:14px;">顔型: {fs}</span>',
                unsafe_allow_html=True,
            )
            st.caption(FACE_SHAPE_ADVICE.get(fs, ""))

        # 計測値テーブル
        if is_lip:
            lm = m
            with st.expander("💋 唇の計測値（数値テーブル）", expanded=False):
                st.dataframe({
                    "項目":   ["口角の高さ差","キューピッドボウ差","下唇の高さ差","唇幅左右差","傾き角度","上唇面積差"],
                    "差分":   [f"{lm['corner_h']:+.1f}NU", f"{lm['cupid_h']:+.1f}NU",
                               f"{lm['lower_h']:+.1f}NU",  f"{lm['width_diff']:+.1f}NU",
                               f"{lm['angle_d']:+.1f}°",   f"{lm['area_pct']:.1f}%"],
                    "mm換算": [nu_to_mm(lm['corner_h']), nu_to_mm(lm['cupid_h']),
                               nu_to_mm(lm['lower_h']),  nu_to_mm(lm['width_diff']), "—","—"],
                    "右唇":   [f"{lm['r_corner_y']:.0f}px", f"{lm['r_cupid_y']:.0f}px",
                               f"{lm['r_lower_y']:.0f}px",  f"{lm['r_half_w']:.0f}px",
                               f"{lm['lip_angle']:.1f}°",   f"{lm['r_upper_area']:.0f}px²"],
                    "左唇":   [f"{lm['l_corner_y']:.0f}px", f"{lm['l_cupid_y']:.0f}px",
                               f"{lm['l_lower_y']:.0f}px",  f"{lm['l_half_w']:.0f}px",
                               "—",                         f"{lm['l_upper_area']:.0f}px²"],
                }, use_container_width=True, hide_index=True)
        else:
            with st.expander("👁 眉の計測値（数値テーブル）", expanded=False):
                st.dataframe({
                    "項目":   ["眉頭高さ差","眉山高さ差","眉尻高さ差","眉長さ差","傾き角差","面積差"],
                    "差分":   [f"{m['head_h']:+.1f}NU",f"{m['peak_h']:+.1f}NU",
                               f"{m['tail_h']:+.1f}NU",f"{m['length']:+.1f}NU",
                               f"{m['angle_d']:+.1f}°",f"{m['area_pct']:.1f}%"],
                    "mm換算": [nu_to_mm(m['head_h']),nu_to_mm(m['peak_h']),
                               nu_to_mm(m['tail_h']),nu_to_mm(m['length']),"—","—"],
                    "右眉":   [f"{m['r_head_y']:.0f}px",f"{m['r_peak_y']:.0f}px",
                               f"{m['r_tail_y']:.0f}px",f"{m['r_length']:.0f}px",
                               f"{m['r_angle']:.1f}°",f"{m['r_area']:.0f}px²"],
                    "左眉":   [f"{m['l_head_y']:.0f}px",f"{m['l_peak_y']:.0f}px",
                               f"{m['l_tail_y']:.0f}px",f"{m['l_length']:.0f}px",
                               f"{m['l_angle']:.1f}°",f"{m['l_area']:.0f}px²"],
                }, use_container_width=True, hide_index=True)

    # 目元分析 (眉タブ + 術前+骨格基準のみ)
    if not is_lip:
        eye_m = res.get("eye_metrics")
        if eye_m:
            st.divider()
            with st.expander("👁 目元の左右差分析", expanded=True):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**幅（横径）**")
                    st.dataframe({"":["右目","左目","差分(%)"],
                                  "値":[eye_m["r_width_mm"],eye_m["l_width_mm"],
                                        f"{eye_m['width_pct']:.1f}%"]},
                                 use_container_width=True, hide_index=True)
                    st.markdown("**高さ（縦径）**")
                    st.dataframe({"":["右目","左目","差分(%)"],
                                  "値":[eye_m["r_height_mm"],eye_m["l_height_mm"],
                                        f"{eye_m['height_pct']:.1f}%"]},
                                 use_container_width=True, hide_index=True)
                with ec2:
                    st.markdown("**面積**")
                    st.dataframe({"":["右目","左目","差分(%)"],
                                  "値":[eye_m["r_area_mm"],eye_m["l_area_mm"],
                                        f"{eye_m['area_pct']:.1f}%"]},
                                 use_container_width=True, hide_index=True)
                for sev_t, label_e, val in [
                    ("ok" if eye_m["width_pct"]<5 else ("warn" if eye_m["width_pct"]<10 else "error"),
                     "幅", eye_m["width_pct"]),
                    ("ok" if eye_m["height_pct"]<5 else ("warn" if eye_m["height_pct"]<10 else "error"),
                     "高さ", eye_m["height_pct"]),
                    ("ok" if eye_m["area_pct"]<5 else ("warn" if eye_m["area_pct"]<15 else "error"),
                     "面積", eye_m["area_pct"]),
                ]:
                    fn = st.success if sev_t=="ok" else (st.warning if sev_t=="warn" else st.error)
                    fn(f"目の{label_e}差: {val:.1f}%")

    # アドバイス
    st.divider()
    part_lbl = "唇" if is_lip else "眉"
    st.subheader(f"デザインアドバイス（{part_lbl}） — {phase_label}")
    for item in advice:
        fn = st.success if item["sev"]=="ok" else (st.warning if item["sev"]=="warn" else st.error)
        fn(f"**{item['title']}**  \n{item['body']}")

    # 減点内訳
    st.divider()
    with st.expander(f"減点内訳（{part_lbl}スコア詳細）", expanded=True):
        total = sum(bd.values())
        st.caption(f"合計減点: {total:.1f}点 → スコア {score}点")
        st.dataframe({
            "項目":   [wll[k] for k in bd],
            "減点":   [f"−{v:.1f}点" for v in bd.values()],
            "重み":   [f"×{wl[k]:.1f}" for k in bd],
            "差分値": [f"{abs(m[k]):.1f}"+("°" if k=="angle_d" else "%" if k=="area_pct" else "NU")
                      for k in bd],
        }, use_container_width=True, hide_index=True)

# ── メイン ────────────────────────────────

# ── ランドマーク手動調整 ─────────────────

def show_landmark_adjustment(res, phase, axis_mode, show_golden, iris_mm, do_eye, fm):
    """基準点の手動調整 — 画像の直下に表示"""
    h2, w2 = res["img_shape"]
    current_cx    = int(res["metrics"]["center_x"])
    current_cx_pct = current_cx / w2 * 100
    current_angle = float(res["angle"])

    with st.expander("🔧 基準点の手動調整", expanded=False):
        st.caption(
            "① 中心線（白い縦線）の左右位置、または ② 傾き角度を変えて再計算できます。\n"
            "**左端=0% / 画像中央=50% / 右端=100%**（患者の向かって右が画像の左になります）"
        )

        # ① 中心線X（パーセント表示で直感的に）
        st.markdown("**① 中心線の位置** — 白い縦線をどこに引くか")
        new_cx_pct = st.slider(
            "← 患者の右側（画像の左）　　　　　患者の左側（画像の右） →",
            25.0, 75.0, current_cx_pct, 0.5,
            format="%.1f%%", key=f"cx_slider_{phase}",
        )
        new_cx = new_cx_pct / 100 * w2
        st.caption(f"現在: {current_cx_pct:.1f}% → 変更後: {new_cx_pct:.1f}%　（{int(new_cx)}px / {w2}px）")

        fast_btn = st.button("⚡ 中心線のみ再計算（再検出なし・高速）",
                             key=f"fast_btn_{phase}", use_container_width=True)

        st.divider()

        # ② 傾き角度
        st.markdown("**② 傾き補正角度** — 正=反時計回り / 負=時計回り")
        new_angle = st.number_input(
            "傾き (°)",
            value=current_angle,
            min_value=current_angle - 15.0,
            max_value=current_angle + 15.0,
            step=0.25, format="%.2f",
            key=f"angle_input_{phase}",
        )
        st.caption(f"自動検出値: {current_angle:.2f}°　→　変更後: {new_angle:.2f}°")

        full_btn = st.button("🔄 傾き込みで完全再分析（再検出あり）",
                             key=f"full_btn_{phase}", use_container_width=True)

        # ── 中心線のみ高速再計算 ──────────────────
        if fast_btn:
            mock = MockLandmarks(res["lm_raw"])
            new_m = analyze_eyebrows(mock, w2, h2, axis_mode, override_cx=float(new_cx))
            ns, ng, nb = calc_score(new_m)
            ppm = res["ppm"]
            proc_bgr = cv2.imdecode(np.frombuffer(res["proc_bytes"], np.uint8), cv2.IMREAD_COLOR)
            new_draw = draw_guidelines(proc_bgr, mock, new_m, ns, ng,
                                       current_angle, axis_mode, show_golden,
                                       face_shape=res.get("face_shape"))
            st.session_state.phase_data[phase].update({
                "draw_rgb": new_draw, "metrics": new_m,
                "score": ns, "grade": ng, "breakdown": nb,
                "advice": generate_advice(new_m, ppm),
            })
            st.rerun()

        # ── 傾き込み完全再分析 ────────────────────
        if full_btn:
            crop_bgr = cv2.imdecode(np.frombuffer(res["crop_bytes"], np.uint8), cv2.IMREAD_COLOR)
            ch, cw = crop_bgr.shape[:2]
            M = cv2.getRotationMatrix2D((cw//2, ch//2), new_angle, 1.0)
            rotated = cv2.warpAffine(crop_bgr, M, (cw, ch), flags=cv2.INTER_CUBIC)
            new_lm = _detect(rotated, fm)
            if new_lm is None:
                st.error("再検出に失敗しました。角度を少し変えてお試しください。")
            else:
                cidx = ([8] if axis_mode == "瞳孔基準"
                        else [234,454] if axis_mode == "骨格基準（鼻根-顎先）"
                        else [168,1,164])
                rotated, new_lm = _center_shift(rotated, new_lm, cidx, cw, ch, fm)
                nh, nw = rotated.shape[:2]
                new_m = analyze_eyebrows(new_lm, nw, nh, axis_mode)
                ns, ng, nb = calc_score(new_m)
                ppm = calc_iris_scale(new_lm, nw, nh, iris_mm) if iris_mm else None
                nfs = classify_face_shape(new_lm, nw, nh)
                nem = analyze_eyes(new_lm, nw, nh, ppm) if do_eye else None
                _, _b = cv2.imencode('.jpg', rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                st.session_state.phase_data[phase].update({
                    "draw_rgb":   draw_guidelines(rotated, new_lm, new_m, ns, ng,
                                                  new_angle, axis_mode, show_golden, nfs),
                    "metrics": new_m, "score": ns, "grade": ng, "breakdown": nb,
                    "advice": generate_advice(new_m, ppm), "ppm": ppm,
                    "angle": new_angle, "face_shape": nfs, "eye_metrics": nem,
                    "proc_bytes": _b.tobytes(),
                    "lm_raw": {i:{"x":new_lm.landmark[i].x,"y":new_lm.landmark[i].y}
                               for i in range(len(new_lm.landmark))},
                    "img_shape": (nh, nw),
                })
                st.rerun()


def main():
    st.title("アートメイク 顔面バランス分析")
    st.caption("MediaPipe Face Mesh | LLMフリー・トークンゼロ版 Ver 3.0")

    fm = get_face_mesh()

    for key, default in [
        ("phase_data",        {p: None for p in PHASES}),
        ("phase_raw",         {p: None for p in PHASES}),
        ("phase_file_id",     {p: None for p in PHASES}),
        ("phase_settings_sig",{p: None for p in PHASES}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── サイドバー ──────────────────────────
    with st.sidebar:
        st.header("設定")

        phase = st.radio(
            "分析フェーズ", PHASES,
            format_func=lambda x: f"{PHASE_ICONS[x]} {x}"
                                  + (" ✓" if st.session_state.phase_data[x] else ""),
        )

        st.divider()
        axis_mode = st.radio(
            "基準軸",
            ["瞳孔基準", "骨格基準（鼻根-顎先）", "中顔面基準（鼻筋）"],
            help=("瞳孔基準: 両瞳孔を水平補正\n"
                  "骨格基準: 鼻根〜顎先ラインを垂直補正\n"
                  "中顔面基準: 眉間(#168)・鼻先(#1)・鼻下(#164)の平均を正中線に"),
        )

        st.divider()
        st.subheader("虹彩キャリブレーション")
        iris_preset = st.selectbox("基準サイズ", list(IRIS_PRESETS.keys()))
        if IRIS_PRESETS[iris_preset] is None:
            iris_mm = st.number_input("虹彩直径 (mm)", 8.0, 20.0, 12.0, 0.1)
        else:
            iris_mm = IRIS_PRESETS[iris_preset]
        st.caption(f"設定値: {iris_mm:.1f}mm")

        st.divider()
        show_golden = st.checkbox(
            "黄金比ガイドを表示",
            help="眉頭(小鼻キワ 102/331)・眉山(虹彩外縁 471/476)・眉尻(小鼻→目尻延長)をピンク+印で表示",
        )

        st.divider()
        bg_choice = st.selectbox("背景カラー", list(BG_OPTIONS.keys()))

        st.divider()
        run_btn = st.button("✅ 設定を反映して分析を更新",
                            type="primary", use_container_width=True)

        st.divider()
        st.caption("NU = 瞳孔間距離100の相対値\n+は右劣位、−は左劣位")

    # 背景色CSS（罫線・入力欄も含む包括適用）
    bg, text = BG_OPTIONS[bg_choice]
    css = get_theme_css(bg, text)
    if css:
        st.markdown(css, unsafe_allow_html=True)

    # 目元分析フラグ (術前 + 骨格基準のみ)
    do_eye = (phase == "術前" and axis_mode == "骨格基準（鼻根-顎先）")

    # 設定シグネチャ
    settings_sig = f"{axis_mode}|{iris_mm}|{show_golden}"

    # ── フェーズ ────────────────────────────
    st.subheader(f"{PHASE_ICONS[phase]} {phase}フェーズ")

    # 設定変更の警告
    if (st.session_state.phase_data[phase] and
            st.session_state.phase_settings_sig[phase] != settings_sig):
        st.warning("設定が変更されています。「設定を反映して分析を更新」ボタンを押してください。")

    uploaded = st.file_uploader(
        f"{phase}の顔写真", type=["jpg","jpeg","png"],
        key=f"up_{phase}",
    )

    need_run = False
    img_bgr  = None

    if uploaded:
        file_id   = f"{uploaded.name}_{uploaded.size}"
        raw_bytes = uploaded.read()
        if st.session_state.phase_file_id[phase] != file_id:
            # 新ファイル → 自動実行
            st.session_state.phase_file_id[phase] = file_id
            st.session_state.phase_raw[phase]     = raw_bytes
            need_run = True
        img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    elif st.session_state.phase_raw[phase] is not None:
        img_bgr = cv2.imdecode(
            np.frombuffer(st.session_state.phase_raw[phase], np.uint8), cv2.IMREAD_COLOR
        )

    if run_btn and img_bgr is not None:
        need_run = True

    # 分析実行
    if need_run and img_bgr is not None:
        h0, w0 = img_bgr.shape[:2]
        if w0 > 1200:
            img_bgr = cv2.resize(img_bgr, (1200, int(h0*1200/w0)), interpolation=cv2.INTER_AREA)
        with st.spinner("顔検出・分析中..."):
            result = run_analysis(img_bgr, fm, axis_mode, iris_mm, show_golden, do_eye)
        if result is None:
            st.error("顔を検出できませんでした。正面向きの写真をお使いください。")
        else:
            st.session_state.phase_data[phase]         = result
            st.session_state.phase_settings_sig[phase] = settings_sig
            st.session_state.landmarks_raw             = result["lm_raw"]
            st.session_state.img_shape                 = result["img_shape"]

    if st.session_state.phase_data[phase]:
        show_result(
            st.session_state.phase_data[phase],
            phase,
            adj_params={
                "phase": phase, "axis_mode": axis_mode,
                "show_golden": show_golden, "iris_mm": iris_mm,
                "do_eye": do_eye, "fm": fm,
            }
        )

    # ── 比較ビュー ──────────────────────────
    available = [(p, st.session_state.phase_data[p])
                 for p in PHASES if st.session_state.phase_data[p]]

    if len(available) >= 2:
        st.divider()
        st.subheader("比較ビュー")
        opts = [p for p,_ in available]
        c1, c2 = st.columns(2)
        lp = c1.selectbox("左（比較元）", opts, key="cmp_l")
        rp = c2.selectbox("右（比較先）", opts,
                           index=min(1, len(opts)-1), key="cmp_r")
        if lp != rp:
            ld = st.session_state.phase_data[lp]
            rd = st.session_state.phase_data[rp]
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"{PHASE_ICONS.get(lp,'')} {lp}")
                st.image(ld["draw_rgb"], use_container_width=True)
                st.metric("スコア", f"{ld['score']}/100", delta=ld["grade"])
            with col2:
                st.caption(f"{PHASE_ICONS.get(rp,'')} {rp}")
                st.image(rd["draw_rgb"], use_container_width=True)
                delta = rd["score"] - ld["score"]
                st.metric("スコア", f"{rd['score']}/100",
                          delta=f"{delta:+d}点", delta_color="normal")


main()
