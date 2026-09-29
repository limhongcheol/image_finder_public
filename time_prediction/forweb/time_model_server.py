"""
time_model_server.py
====================
촬영시각 예측 추론 서버 (GPU 보유 머신에서 실행) — **Gemini 캡션 포함 버전**.

forweb/model_server.py 와 같은 역할이지만, 캡션 없이 이미지만 쓰는 기존 데모와 달리
이 서버는 `inference_one_image.py` 의 모듈(load_models / predict_datetime)을 그대로
재사용하여 "이미지 -> Gemini 캡션 생성 -> Qwen3.5-9B(layer 9) -> SequenceProbeModel
-> 촬영시각" 파이프라인을 수행한다. (캡션을 같이 넣어야 예측이 상수로 붕괴하지 않음)

Lumina 구조에서의 위치:
    앱 -> ② API서버(app.py, localhost) -> (POST /predict) -> 이 서버(GPU)

실행:
    python time_model_server.py
환경변수:
    MODEL_PORT    기본 5000        (이 추론 서버 포트)
    MODEL_PATH    기본 ../saved_models/best_layer_9.pth (없으면 ../best_layer_9.pth 폴백)
    USE_CAPTION   기본 1           0 으로 두면 Gemini 캡션 없이 이미지만(빠르지만 품질↓)
    MOCK          기본 0           1 이면 무거운 모델을 안 띄우고 랜덤 시각 반환(흐름 테스트용)
"""
import os
import sys
import io
import calendar
import random
from datetime import datetime

from flask import Flask, request, jsonify

# forweb 의 부모(forgithub) 를 import 경로에 추가 → inference_one_image 등 재사용
FORWEB_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(FORWEB_DIR)          # forgithub
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

YEAR = 2026
MOCK = os.environ.get("MOCK", "0") == "1"
USE_CAPTION = os.environ.get("USE_CAPTION", "1") == "1"
MODEL_PORT = int(os.environ.get("MODEL_PORT", "5000"))

# 가중치 경로: saved_models/best_layer_9.pth 우선, 없으면 forgithub/best_layer_9.pth
_default_model = os.path.join(BASE_DIR, "saved_models", "best_layer_9.pth")
if not os.path.exists(_default_model):
    _alt = os.path.join(BASE_DIR, "best_layer_9.pth")
    if os.path.exists(_alt):
        _default_model = _alt
MODEL_PATH = os.environ.get("MODEL_PATH", _default_model)

app = Flask(__name__)
_READY = False          # 모델 로딩 완료 여부
_DEVICE_STR = "cpu"


def load_models():
    """inference_one_image.load_models() 로 Qwen 임베딩 + Probe(+Gemini) 를 1회 로딩."""
    global _READY, _DEVICE_STR
    import inference_one_image as one
    one.load_models(model_path=MODEL_PATH, use_caption=USE_CAPTION)
    try:
        _DEVICE_STR = str(one._DEVICE)
    except Exception:
        pass
    _READY = True


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "mock": MOCK,
        "ready": _READY,
        "use_caption": USE_CAPTION,
        "device": _DEVICE_STR,
        "model_path": MODEL_PATH,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "image 파일이 필요합니다."}), 400

    raw = request.files["image"].read()
    filename = request.files["image"].filename or "upload"

    try:
        if MOCK:
            # 무거운 모델 없이 웹 흐름만 점검: 랜덤 시각
            month = random.randint(1, 12)
            day = random.randint(1, calendar.monthrange(YEAR, month)[1])
            pred_dt = datetime(YEAR, month, day, random.randint(0, 23), random.randint(0, 59))
            actual, caption = None, "(mock caption)"
        else:
            from PIL import Image
            import inference_one_image as one
            pil = Image.open(io.BytesIO(raw))
            # caption=None → load_models(use_caption=True) 시 Gemini 로 자동 생성
            pred_dt, actual, caption = one.predict_datetime(pil)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "ok": True,
        "file_name": filename,
        "predicted_time": pred_dt.strftime("%m-%d %H:%M"),          # 연도 제외
        "predicted_time_full": pred_dt.strftime("%Y-%m-%d %H:%M"),  # 연도 포함(YEAR 고정)
        "actual_time": actual,                                       # EXIF 실제시각 (없으면 None)
        "caption": caption,                                          # 생성된 캡션
        "mock": MOCK,
    })


if __name__ == "__main__":
    if not MOCK:
        print(f"[time_model_server] 모델 로딩 시작 (caption={USE_CAPTION}, weights={MODEL_PATH})")
        load_models()
        print("[time_model_server] 모델 로딩 완료.")
    else:
        print("[time_model_server] MOCK 모드: 무거운 모델을 로딩하지 않습니다.")
    print(f"[time_model_server] http://0.0.0.0:{MODEL_PORT} 에서 추론 서버 시작")
    # GPU 직렬 처리(동시 추론 시 VRAM 경합 방지) → threaded=False
    app.run(host="0.0.0.0", port=MODEL_PORT, threaded=False)
