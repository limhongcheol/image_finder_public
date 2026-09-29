"""
web_server.py
=============
localhost 호스팅용 웹 서버.

서버용 컴퓨터에서 이 파일을 실행하면 8080 포트로 웹사이트(localhost)가 뜬다.
사용자가 사진을 업로드하면, 이 서버는 이미지를 "모델 서버"(model_server.py) 로
전달(POST /predict)하고, 돌려받은 촬영시각 예측 결과를 화면에 보여준다.

실행:
    python web_server.py
환경변수:
    WEB_PORT          기본 8080
    MODEL_SERVER_URL  기본 http://localhost:5000   (model_server.py 주소)
                      모델 서버가 다른 컴퓨터면 그 컴퓨터 주소로 지정.
                      예) http://model-server:5000
"""
import os

import requests
from flask import Flask, request, render_template_string, jsonify

WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:5000").rstrip("/")

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>localhost · 사진 촬영시각 예측</title>
  <style>
    * { box-sizing: border-box; }
    body { margin:0; font-family: -apple-system, "Segoe UI", "Malgun Gothic", sans-serif;
           background: linear-gradient(135deg,#1e1b4b,#312e81,#0f172a); color:#e5e7eb;
           min-height:100vh; display:flex; flex-direction:column; align-items:center; }
    header { margin-top:48px; text-align:center; }
    header h1 { font-size:2rem; margin:0; letter-spacing:1px; }
    header p { color:#a5b4fc; margin-top:8px; }
    .card { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12);
            border-radius:18px; padding:28px; margin-top:32px; width:min(92vw,520px);
            backdrop-filter: blur(8px); box-shadow:0 12px 40px rgba(0,0,0,0.35); }
    .drop { border:2px dashed #6366f1; border-radius:14px; padding:36px; text-align:center;
            cursor:pointer; transition:.2s; }
    .drop:hover { background:rgba(99,102,241,0.12); }
    .drop.over { background:rgba(99,102,241,0.22); border-color:#a5b4fc; }
    #preview { max-width:100%; border-radius:12px; margin-top:18px; display:none; }
    button { margin-top:18px; width:100%; padding:14px; border:none; border-radius:12px;
             background:#6366f1; color:#fff; font-size:1rem; font-weight:600; cursor:pointer; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    .result { margin-top:22px; padding:20px; border-radius:14px; background:rgba(16,185,129,0.12);
              border:1px solid rgba(16,185,129,0.4); display:none; }
    .result .big { font-size:1.8rem; font-weight:700; color:#6ee7b7; }
    .row { display:flex; justify-content:space-between; margin-top:8px; color:#cbd5e1; }
    .err { color:#fca5a5; margin-top:14px; }
    footer { margin:40px 0 24px; color:#64748b; font-size:.85rem; }
  </style>
</head>
<body>
  <header>
    <h1>📸 localhost</h1>
    <p>사진 한 장으로 촬영 시각을 예측합니다</p>
  </header>

  <div class="card">
    <div class="drop" id="drop">
      <div id="dropText">클릭하거나 사진을 여기로 끌어다 놓으세요</div>
      <input type="file" id="file" accept="image/*" hidden>
    </div>
    <img id="preview">
    <button id="submit" disabled>예측하기</button>
    <div class="err" id="err"></div>
    <div class="result" id="result">
      <div>예측 촬영 시각</div>
      <div class="big" id="pred">-</div>
      <div class="row"><span>전체 예측</span><span id="predFull">-</span></div>
      <div class="row"><span>EXIF 실제 시각</span><span id="actual">-</span></div>
    </div>
  </div>

  <footer>model: RobustFrequencyProbe + Qwen embedding · backend: {{ model_url }}</footer>

<script>
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  const preview = document.getElementById('preview');
  const submit = document.getElementById('submit');
  const errBox = document.getElementById('err');
  const result = document.getElementById('result');
  let selected = null;

  drop.addEventListener('click', () => file.click());
  ['dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
  ['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', ev => { if (ev.dataTransfer.files.length) setFile(ev.dataTransfer.files[0]); });
  file.addEventListener('change', () => { if (file.files.length) setFile(file.files[0]); });

  function setFile(f) {
    selected = f;
    submit.disabled = false;
    result.style.display = 'none';
    errBox.textContent = '';
    const r = new FileReader();
    r.onload = e => { preview.src = e.target.result; preview.style.display = 'block'; };
    r.readAsDataURL(f);
  }

  submit.addEventListener('click', async () => {
    if (!selected) return;
    submit.disabled = true; submit.textContent = '예측 중...';
    errBox.textContent = ''; result.style.display = 'none';
    const fd = new FormData(); fd.append('image', selected);
    try {
      const res = await fetch('/upload', { method:'POST', body: fd });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || '예측 실패');
      document.getElementById('pred').textContent = data.predicted_time;
      document.getElementById('predFull').textContent = data.predicted_time_full;
      document.getElementById('actual').textContent = data.actual_time || '(EXIF 없음)';
      result.style.display = 'block';
    } catch (e) {
      errBox.textContent = '⚠ ' + e.message;
    } finally {
      submit.disabled = false; submit.textContent = '예측하기';
    }
  });
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE, model_url=MODEL_SERVER_URL)


@app.route("/upload", methods=["POST"])
def upload():
    """브라우저 -> 이 서버 -> 모델 서버 로 이미지를 중계."""
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "이미지가 없습니다."}), 400

    f = request.files["image"]
    try:
        resp = requests.post(
            f"{MODEL_SERVER_URL}/predict",
            files={"image": (f.filename, f.stream, f.mimetype)},
            timeout=300,
        )
        return jsonify(resp.json()), resp.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({
            "ok": False,
            "error": f"모델 서버({MODEL_SERVER_URL}) 연결 실패: {e}",
        }), 502


if __name__ == "__main__":
    print(f"[web_server] localhost -> http://0.0.0.0:{WEB_PORT}")
    print(f"[web_server] 모델 서버: {MODEL_SERVER_URL}")
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)
