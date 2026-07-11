from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import numpy as np
import math
import cv2

app = FastAPI()

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


def resize_image_by_ratio(image, target_ratio=0.35):
    """
    画像をターゲットの比率にリサイズする関数
    target_ratio: 0.33 (1/3) ～ 0.4 (2/5) の範囲で指定
    """
    h, w = image.shape[:2]
    
    # 新しいサイズを計算
    new_w = int(w * target_ratio)
    new_h = int(h * target_ratio)
    
    # リサイズ実行（INTER_AREAは縮小時に高品質な結果が得られます）
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized_image

# --- 使用例 ---
# image = cv2.imread('path_to_your_image.jpg')
# processed_image = resize_image_by_ratio(image, target_ratio=0.35)


def process_face(img):
    """顔の傾き補正と座標取得を行う共通関数"""
    h, w, _ = img.shape
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None, None, None

    landmarks = results.multi_face_landmarks[0]
    right_pupil = landmarks.landmark[468]
    left_pupil = landmarks.landmark[473]
    
    # 傾き補正
    angle = math.degrees(math.atan2((left_pupil.y - right_pupil.y)*h, (left_pupil.x - right_pupil.x)*w))
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    # 再取得
    rgb_rotated = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
    new_results = face_mesh.process(rgb_rotated)
    if not new_results.multi_face_landmarks:
        return None, None, None

    return rotated_img, new_results.multi_face_landmarks[0], angle

def calculate_diffs(landmarks, w, h):
    """ズレとスコアを計算する関数（辛口プロ仕様）"""
    center_x = int(landmarks.landmark[8].x * w)
    ipd_px = abs((landmarks.landmark[473].x * w) - (landmarks.landmark[468].x * w))

    def to_nu(px_value):
        return (px_value / ipd_px) * 100

    r_head_y, r_peak_y, r_tail_y = int(landmarks.landmark[285].y * h), int(landmarks.landmark[296].y * h), int(landmarks.landmark[300].y * h)
    l_head_y, l_peak_y, l_tail_y = int(landmarks.landmark[55].y * h), int(landmarks.landmark[66].y * h), int(landmarks.landmark[70].y * h)
    r_head_x, l_head_x = int(landmarks.landmark[285].x * w), int(landmarks.landmark[55].x * w)

    # --- 面積の計算を追加 ---
    r_points = [285, 296, 334, 293, 300, 276, 283, 282, 295]
    r_contour = np.array([[(int(landmarks.landmark[p].x * w), int(landmarks.landmark[p].y * h))] for p in r_points])
    r_area = cv2.contourArea(r_contour)

    l_points = [55, 66, 105, 63, 70, 46, 53, 52, 65]
    l_contour = np.array([[(int(landmarks.landmark[p].x * w), int(landmarks.landmark[p].y * h))] for p in l_points])
    l_area = cv2.contourArea(l_contour)
    
    area_diff_percent = 0
    if r_area > 0 and l_area > 0:
        area_diff_percent = abs(r_area - l_area) / max(r_area, l_area) * 100

    # --- 評価項目 ---
    diff_dict = {
        "眉頭の距離": to_nu(abs(abs(center_x - r_head_x) - abs(l_head_x - center_x))),
        "眉頭の高さ": to_nu(abs(r_head_y - l_head_y)),
        "眉山の高さ": to_nu(abs(r_peak_y - l_peak_y)),
        "眉尻の高さ": to_nu(abs(r_tail_y - l_tail_y)),
        "眉の面積差": area_diff_percent  # 面積差を％で追加
    }
    
    # Top 5 の抽出
    top_5_diffs = sorted(diff_dict.items(), key=lambda item: item[1], reverse=True)[:5]
    top5_list = [{"part": k, "diff": round(v, 2), "unit": "%" if "面積" in k else "NU"} for k, v in top_5_diffs]

    # --- 辛口スコア計算 ---
    penalty = 0
    for name, value in diff_dict.items():
        if "面積" in name:
            # 面積は1%違うごとに2点減点
            penalty += value * 2.0
        else:
            # 距離・高さは 1 NU(約0.6mm) ズレるごとに5点減点（許容範囲なし）
            penalty += value * 2.0

    score = max(0, 100 - int(penalty))
    
    # 5段階評価の判定も少し厳しく
    if score >= 90: grade = "Excellent"
    elif score >= 80: grade = "Good"
    elif score >= 70: grade = "Normal"
    elif score >= 50: grade = "Bad" # 50点〜69点はBad
    else: grade = "Very Bad"

    return score, grade, top5_list


# ==========================================
# API 1: 計算データのみを返す（LLM用・激安）
# ==========================================
@app.post("/analyze_math")
async def analyze_math(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        img = resize_image_by_ratio(img, target_ratio=0.35)
        rotated_img, landmarks, _ = process_face(img)
        if not landmarks:
            return JSONResponse(content={"error": "Face not found"}, status_code=400)

        # analyze_draw と同じ鼻筋センタリング
        h, w = rotated_img.shape[:2]
        nose_x = int(landmarks.landmark[8].x * w)
        shift_x = w // 2 - nose_x
        if abs(shift_x) > 1:
            M_shift = np.float32([[1, 0, shift_x], [0, 1, 0]])
            rotated_img = cv2.warpAffine(rotated_img, M_shift, (w, h))
            rgb_shifted = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
            shifted_results = face_mesh.process(rgb_shifted)
            if shifted_results.multi_face_landmarks:
                landmarks = shifted_results.multi_face_landmarks[0]

        score, grade, top5_list = calculate_diffs(landmarks, w, h)
        
        # LLMに渡すための綺麗なテキストを作成して返す
        ai_text = f"【スコア】{score}点\n【判定】{grade}\n【ズレが大きい箇所】\n"
        for item in top5_list:
            ai_text += f"- {item['part']}: {item['diff']} {item['unit']}\n"

        return JSONResponse(content={"ai_text": ai_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ==========================================
# API 2: 画像ファイルのみを返す（表示用・全ガイド線＋文字入り）
# ==========================================
@app.post("/analyze_draw")
async def analyze_draw(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        img = resize_image_by_ratio(img, target_ratio=0.35)
        rotated_img, landmarks, angle = process_face(img)
        if not landmarks:
            return Response(content="Face not found", status_code=400)

        h, w, _ = rotated_img.shape

        # 鼻筋(landmark[8])を画像の水平中央に揃える
        nose_x = int(landmarks.landmark[8].x * w)
        shift_x = w // 2 - nose_x
        if abs(shift_x) > 1:
            M_shift = np.float32([[1, 0, shift_x], [0, 1, 0]])
            rotated_img = cv2.warpAffine(rotated_img, M_shift, (w, h))
            rgb_shifted = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
            shifted_results = face_mesh.process(rgb_shifted)
            if shifted_results.multi_face_landmarks:
                landmarks = shifted_results.multi_face_landmarks[0]

        draw_img = rotated_img.copy()
        overlay = draw_img.copy() # 透過線用

        # 1. カラー設定 (BGR)
        COLOR_WHITE = (255, 255, 255)
        COLOR_R = (0, 165, 255) # 右眉(オレンジ)
        COLOR_L = (255, 255, 0) # 左眉(水色)
        
        # 2. 座標取得
        center_x = int(landmarks.landmark[8].x * w)
        pupil_y = int((landmarks.landmark[468].y + landmarks.landmark[473].y) / 2 * h)

        # 眉頭・眉山・眉尻のX座標
        r_head_x, r_peak_x, r_tail_x = int(landmarks.landmark[285].x * w), int(landmarks.landmark[296].x * w), int(landmarks.landmark[300].x * w)
        l_head_x, l_peak_x, l_tail_x = int(landmarks.landmark[55].x * w), int(landmarks.landmark[66].x * w), int(landmarks.landmark[70].x * w)

        # 眉頭上下・眉山・眉尻のY座標
        r_head_top_y, r_head_bot_y = int(landmarks.landmark[285].y * h), int(landmarks.landmark[295].y * h)
        r_peak_y, r_tail_y = int(landmarks.landmark[296].y * h), int(landmarks.landmark[300].y * h)

        l_head_top_y, l_head_bot_y = int(landmarks.landmark[55].y * h), int(landmarks.landmark[65].y * h)
        l_peak_y, l_tail_y = int(landmarks.landmark[66].y * h), int(landmarks.landmark[70].y * h)

        # 3. ガイド線の描画
        # --- 垂直線 (白・細く) ---
        cv2.line(draw_img, (center_x, 0), (center_x, h), COLOR_WHITE, 1) # 正中線
        for x in [r_head_x, r_peak_x, r_tail_x, l_head_x, l_peak_x, l_tail_x]:
            cv2.line(draw_img, (x, 0), (x, h), COLOR_WHITE, 1)

        # --- 水平線 (瞳孔) ---
        cv2.line(draw_img, (0, pupil_y), (w, pupil_y), COLOR_WHITE, 1)

        # --- 水平線 (眉頭・眉山) ---
        cv2.line(draw_img, (0, r_head_top_y), (w, r_head_top_y), COLOR_R, 1)
        cv2.line(draw_img, (0, r_head_bot_y), (w, r_head_bot_y), COLOR_R, 1)
        cv2.line(draw_img, (0, r_peak_y), (w, r_peak_y), COLOR_R, 1)
        
        cv2.line(draw_img, (0, l_head_top_y), (w, l_head_top_y), COLOR_L, 1)
        cv2.line(draw_img, (0, l_head_bot_y), (w, l_head_bot_y), COLOR_L, 1)
        cv2.line(draw_img, (0, l_peak_y), (w, l_peak_y), COLOR_L, 1)

        # --- 眉尻の水平線 (極薄く・透過) ---
        cv2.line(overlay, (0, r_tail_y), (w, r_tail_y), COLOR_R, 1)
        cv2.line(overlay, (0, l_tail_y), (w, l_tail_y), COLOR_L, 1)
        cv2.addWeighted(overlay, 0.3, draw_img, 0.7, 0, draw_img)

        # 4. 文字情報の印字 (スコアと凡例)
        # スコアを再計算して取得
        score, grade, _ = calculate_diffs(landmarks, w, h)
        
        # 背景を少し暗くして文字を見やすくする
        cv2.rectangle(draw_img, (5, 5), (300, 85), (0, 0, 0), -1)

        cv2.putText(draw_img, f"Score: {score}/100 [{grade}]", (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 1)
        cv2.putText(draw_img, "Axis: EYE (Pupils)", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1)
        cv2.putText(draw_img, f"Tilt: {round(angle, 2)} deg", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1)

        cv2.putText(draw_img, "Right Brow", (165, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_R, 1)
        cv2.putText(draw_img, "Left Brow", (165, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_L, 1)

        # 5. 軽量化して出力
        max_size = 1200 # 文字が潰れないように少し大きめに保持
        if w > max_size:
            ratio = max_size / w
            draw_img = cv2.resize(draw_img, (max_size, int(h * ratio)))

        _, buffer = cv2.imencode('.jpg', draw_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return Response(content=buffer.tobytes(), media_type="image/jpeg")

    except Exception as e:
         return Response(content=str(e), status_code=500)