"""
model_server.py
================
"이 컴퓨터"(GPU 보유) 에서 실행하는 추론 서버.

업로드된 이미지 1장을 받아서
    이미지 -> Qwen 임베딩(layer 9) -> SequenceProbeModel -> 촬영 시각 예측
을 수행하고 JSON 으로 결과를 돌려준다.

localhost 호스팅 서버(web_server.py) 가 이 서버의 /predict 엔드포인트로
이미지를 POST 한다.

실행:
    python model_server.py
환경변수:
    MODEL_PORT   기본 5000      (이 추론 서버 포트)
    MODEL_PATH   기본 saved_models/best_layer_9.pth (없으면 best_model_val_loss.pth 로 폴백)
    MOCK         기본 0         1 로 두면 무거운 모델을 안 띄우고 랜덤 시각을 반환(웹 흐름만 테스트용)
"""
import os
import sys
import io
import math
import calendar
import random
from datetime import datetime

from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# 경로 설정 : forweb 의 부모(forgithub) 를 import 경로에 추가해서
#            기존 모듈(inference_code, make_dataset_embedding, make_dataset) 재사용
# ---------------------------------------------------------------------------
FORWEB_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(FORWEB_DIR)          # forgithub
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
LAYER_IDX = 9
YEAR = 2026
IMAGE_SIDE = 448

MOCK = os.environ.get("MOCK", "0") == "1"
MODEL_PORT = int(os.environ.get("MODEL_PORT", "5000"))

_default_model = os.path.join(BASE_DIR, "saved_models", "best_layer_9.pth")
if not os.path.exists(_default_model):
    # forgithub 폴더에 같이 들어있는 가중치로 폴백
    _fallback = os.path.join(BASE_DIR, "best_model_val_loss.pth")
    if os.path.exists(_fallback):
        _default_model = _fallback
MODEL_PATH = os.environ.get("MODEL_PATH", _default_model)

app = Flask(__name__)

# 로딩된 무거운 객체들 (지연 로딩)
_PROCESSOR = None
_QWEN = None
_PROBE = None
_DEVICE = None
_inverse_angular_time = None
_extract_metadata = None


# ---------------------------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------------------------
def _build_probe():
    """학습에 쓰인 RobustFrequencyProbe 정의 (data_refine0428.ipynb 와 동일).

    best_model_val_loss.pth 는 이 구조(input_proj + query_bank + cross_attn +
    ffn + head)로 학습됐다. forward 는 4개 값[theta, phi, log_var_s, log_var_t]
    을 내며, 추론 시각 디코딩에는 앞의 두 값(mu = theta, phi)만 사용한다.
    """
    import torch
    import torch.nn as nn

    class RobustFrequencyProbe(nn.Module):
        def __init__(self, input_dim=4096, d_model=512, num_queries=10):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)

            sigma_list = [2 ** i for i in range(num_queries)]
            init_queries = torch.randn(num_queries, d_model)
            for i, sigma in enumerate(sigma_list):
                init_queries[i] *= (1.0 / (sigma + 1e-6))
            self.query_bank = nn.Parameter(init_queries)

            self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(1024, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)

            self.head = nn.Sequential(
                nn.Linear(num_queries * d_model, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, 4),
            )

        def forward(self, x, mask):
            batch_size = x.size(0)
            img_feats = self.input_proj(x)
            queries = self.query_bank.unsqueeze(0).expand(batch_size, -1, -1)

            attn_out, _ = self.cross_attn(
                query=queries, key=img_feats, value=img_feats, key_padding_mask=~mask)
            x_queries = self.norm1(queries + attn_out)
            x_queries = self.norm2(x_queries + self.ffn(x_queries))

            out = self.head(x_queries.reshape(batch_size, -1))
            mu = out[:, :2]
            log_var = torch.clamp(out[:, 2:], min=-5.0, max=3.0)
            return torch.cat([mu, log_var], dim=1)

    return RobustFrequencyProbe()


def load_models():
    """Qwen 임베딩 모델 + RobustFrequencyProbe 를 메모리에 올린다 (서버 기동 시 1회)."""
    global _PROCESSOR, _QWEN, _PROBE, _DEVICE
    global _inverse_angular_time, _extract_metadata

    import torch
    import make_dataset_embedding as emb
    import inference_code as ic
    import make_dataset as md

    _inverse_angular_time = ic.inverse_angular_time
    _extract_metadata = md.extract_metadata
    ic.YEAR = YEAR  # inverse_angular_time 이 참조하는 연도

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_server] device = {_DEVICE}")

    print("[model_server] Qwen 임베딩 모델 로딩 중...")
    _PROCESSOR, _QWEN = emb.load_model()

    print(f"[model_server] Probe 모델(RobustFrequencyProbe) 로딩 중... ({MODEL_PATH})")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없음: {MODEL_PATH}")
    _PROBE = _build_probe().to(_DEVICE)
    _PROBE.load_state_dict(torch.load(MODEL_PATH, map_location=_DEVICE))
    _PROBE.eval()
    print("[model_server] 모든 모델 로딩 완료.")


def _embed_pil(pil_image):
    """PIL 이미지 한 장에서 layer 9 hidden_states 텐서를 추출."""
    import torch
    image = pil_image.convert("RGB")
    image.thumbnail((IMAGE_SIDE, IMAGE_SIDE))

    messages = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = _PROCESSOR.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _PROCESSOR(text=[prompt], images=[image], return_tensors="pt").to(_QWEN.device)

    # 부동소수 입력(pixel_values 등)을 모델 dtype(bfloat16)으로 맞춘다.
    # (안 맞추면 비전 타워에서 "mat1 and mat2 must have the same dtype,
    #  got float and bfloat16" 런타임 에러가 날 수 있다.)
    model_dtype = getattr(_QWEN, "dtype", None)
    if model_dtype is not None:
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(model_dtype)

    with torch.no_grad():
        outputs = _QWEN(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[LAYER_IDX].squeeze(0).to(torch.float32).cpu()
    del outputs, inputs
    return hidden


def predict_datetime(pil_image):
    """이미지 -> 예측 datetime. (예측시각, EXIF실제시각|None) 반환."""
    import torch
    import pandas as pd

    # EXIF 실제 촬영시각/위도 (있으면 비교 및 남반구 보정에 사용)
    actual, lat = None, None
    try:
        dt_str, lat, _ = _extract_metadata(pil_image)
        if dt_str:
            actual = dt_str
    except Exception:
        actual, lat = None, None

    hidden = _embed_pil(pil_image)
    h = hidden.unsqueeze(0).to(_DEVICE)
    mask = torch.ones(h.shape[:2], dtype=torch.bool, device=_DEVICE)

    with torch.no_grad():
        pred = _PROBE(h, mask).cpu().numpy()[0]

    # 모델 출력 mu = (theta, phi) 만 사용 (뒤 2개는 불확실성 log_var)
    pred_dt = _inverse_angular_time(float(pred[0]), float(pred[1]))

    # 학습과 동일한 남반구 보정: 위도 < -5 이면 예측을 6개월 되돌린다
    try:
        if lat is not None and float(lat) < -5.0:
            pred_dt = (pred_dt - pd.DateOffset(months=6)).to_pydatetime()
    except (TypeError, ValueError):
        pass

    return pred_dt, actual


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "mock": MOCK, "device": str(_DEVICE)})


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
            pred_dt = datetime(YEAR, month, day,
                               random.randint(0, 23), random.randint(0, 59))
            actual = None
        else:
            from PIL import Image
            pil = Image.open(io.BytesIO(raw))
            pred_dt, actual = predict_datetime(pil)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "ok": True,
        "file_name": filename,
        "predicted_time": pred_dt.strftime("%m-%d %H:%M"),
        "predicted_time_full": pred_dt.strftime("%Y-%m-%d %H:%M"),
        "actual_time": actual,
        "mock": MOCK,
    })


if __name__ == "__main__":
    if not MOCK:
        load_models()
    else:
        print("[model_server] MOCK 모드: 무거운 모델을 로딩하지 않습니다.")
    print(f"[model_server] http://0.0.0.0:{MODEL_PORT} 에서 추론 서버 시작")
    app.run(host="0.0.0.0", port=MODEL_PORT, threaded=False)
