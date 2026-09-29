"""
web_app.py — Image Finder 통합 대시보드 (조회 전용)
data_info_agent.py (라이프스타일) + web2_mypicture.py (관계 분석) 통합
"""
import os
import re
import csv
import json
import gzip
import math
import time
import threading
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, send_file, abort, Response, request, session, redirect, url_for, g
from flask_cors import CORS
from functools import wraps

import requests
import secrets
from werkzeug.security import generate_password_hash, check_password_hash

import config

app = Flask(__name__)
CORS(app)  # 모바일 앱에서 호출 가능하도록
app.secret_key = os.getenv("LUMINA_SECRET_KEY") or secrets.token_hex(32)
# 업로드 최대 용량: 10GB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = None  # 폼 필드 메모리 제한 해제 (대용량 파일 대응)

# ── 멀티유저 경로 (요청의 user_id 별 폴더로 동적 해석) ──────────
# user_id 가 오고 DATA_DIR/<user_id>/ 가 있으면 그 유저 폴더를, 없으면 config 기본 폴더(데모).
def _resolve_user_dirs() -> dict:
    uid = (request.args.get("user_id") or request.form.get("user_id") or "").strip()
    if uid:
        root = os.path.join(str(config.DATA_DIR), uid)
        if os.path.isdir(root):
            info = os.path.join(root, "person0_deep_profile")
            return {
                "agent": root,
                "info": info,
                "thumbs": os.path.join(info, "thumbnails"),
                "cluster": os.path.join(root, "cropped_insight", "adaface_results"),
                "original": os.path.join(root, "Camera"),
            }
    # 폴백: config 고정 경로 (단일 데이터셋 / 데모)
    return {
        "agent": str(config.AGENT_OUTPUT_DIR),
        "info": str(config.INFO_OUTPUT_DIR),
        "thumbs": str(config.THUMBNAILS_DIR),
        "cluster": str(config.CLUSTER_DIR),
        "original": str(config.ORIGINAL_DIR),
    }


@app.before_request
def _bind_user_dirs():
    g.user_dirs = _resolve_user_dirs()


class _ReqPath(os.PathLike):
    """요청별 user_id 폴더로 해석되는 경로 (os.path.* 와 호환)."""
    def __init__(self, key):
        self._key = key
    def __fspath__(self):
        return str(g.user_dirs[self._key])
    def __str__(self):
        return self.__fspath__()


AGENT_OUTPUT_DIR = _ReqPath("agent")      # 관계분석 결과
INFO_OUTPUT_DIR  = _ReqPath("info")       # 라이프스타일 결과
THUMBNAILS_DIR   = _ReqPath("thumbs")
ORIGINAL_DIR     = _ReqPath("original")   # 원본 사진
CLUSTER_DIR      = _ReqPath("cluster")    # 얼굴 클러스터

HOST = config.HOST
PORT = config.PORT

# ── 보안 설정 ──────────────────────────────────────────────────
PASSWORD = config.PASSWORD

def login_required(f):
    """모바일용: 기본 인증 비활성화(통과). config.REQUIRE_AUTH=True 면 세션 검사."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if config.REQUIRE_AUTH and not session.get('authenticated'):
            return jsonify({"error": "인증 필요"}), 401
        return f(*args, **kwargs)
    return decorated_function

# ── 경로 설정 ──────────────────────────────────────────────────
NAMES_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person_names.json")
UPLOAD_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
FILE_META   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_metadata.json")
AI_SERVER_STATE_DIR = os.path.join(UPLOAD_DIR, "_ai_server_state")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AI_SERVER_STATE_DIR, exist_ok=True)

# 푸시 알림 토큰 저장 (user_id -> Expo push token). 동시 쓰기 보호용 락.
PUSH_TOKENS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "push_tokens.json")
_push_tokens_lock = threading.Lock()

# 회원 계정 저장 (user_id -> {pw_hash, token, created_at}). 동시 쓰기 보호용 락.
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")
_users_lock = threading.Lock()

RELATION_COLORS = {
    "가족":        "#c98282",
    "학창시절친구": "#7aa2f7",
    "직장동료":    "#8fa7b8",
    "지인":        "#9fb18d",
    "알수없음":    "#6f7480",
}


# ── 공통 유틸 ──────────────────────────────────────────────────

def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _request_user_id(default: str = "default") -> str:
    data = request.get_json(silent=True) or {}
    return (data.get("user_id") or request.args.get("user_id") or request.form.get("user_id") or default).strip()


def _safe_state_user_id(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", user_id.strip())
    return safe or "default"


def _ai_server_state_path(user_id: str) -> str:
    return os.path.join(AI_SERVER_STATE_DIR, f"{_safe_state_user_id(user_id)}.json")


def _load_ai_server_state(user_id: str) -> dict:
    return safe_load_json(_ai_server_state_path(user_id)) or {}


def _save_ai_server_state(user_id: str, state: dict) -> None:
    with open(_ai_server_state_path(user_id), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ── 푸시 알림 (Expo Push) ──────────────────────────────────────
def _load_push_tokens() -> dict:
    return safe_load_json(PUSH_TOKENS_FILE) or {}


def _save_push_tokens(tokens: dict) -> None:
    with open(PUSH_TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)


def _send_expo_push(token: str, title: str, body: str, data: dict | None = None) -> dict | None:
    """Expo Push API 로 푸시 1건 전송. 실패해도 예외 없이 None 반환."""
    try:
        r = requests.post(
            "https://exp.host/--/api/v2/push/send",
            json={
                "to": token,
                "title": title,
                "body": body,
                "data": data or {},
                "sound": "default",
                "priority": "high",
                "channelId": "lumina",   # 앱이 만든 안드로이드 채널명
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return r.json()
    except Exception as e:
        print(f"[push] 전송 실패: {e}", flush=True)
        return None


def _maybe_send_completion_push(user_id: str, state: dict) -> None:
    """분석 완료 시 해당 user_id 토큰으로 푸시 1회 전송.
    state['push_sent'] 플래그로 중복 발송을 막는다(같은 완료에 한 번만).
    토큰이 아직 없으면 플래그를 세우지 않아, 추후 토큰 등록 후 재시도가 가능하다."""
    if state.get("push_sent"):
        return
    token = _load_push_tokens().get(user_id)
    if not token:
        return
    result = _send_expo_push(
        token, "AI 분석 완료 🎉",
        "내 사진 분석이 끝났어요. 앱에서 확인하세요!",
        {"type": "analysis_done", "user_id": user_id},
    )
    # 무효 토큰(DeviceNotRegistered) 정리
    try:
        item = (result or {}).get("data")
        if isinstance(item, dict) and item.get("status") == "error" \
                and (item.get("details") or {}).get("error") == "DeviceNotRegistered":
            with _push_tokens_lock:
                tokens = _load_push_tokens()
                if tokens.pop(user_id, None) is not None:
                    _save_push_tokens(tokens)
    except Exception:
        pass
    state["push_sent"] = True
    state["push_sent_at"] = datetime.now().isoformat()
    _save_ai_server_state(user_id, state)


def _analysis_ready() -> bool:
    analyses = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
    graph_ready = os.path.isfile(os.path.join(AGENT_OUTPUT_DIR, "knowledge_graph.json"))
    return bool(analyses) and graph_ready


def _ai_server_progress(user_id: str) -> int:
    state = _load_ai_server_state(user_id)
    if _analysis_ready():
        return 100
    return max(0, min(100, safe_int(state.get("progress"), 0)))


# ── AI 처리 서버(별도 컴퓨터) 호출 클라이언트 ──────────────────
def _ai_server_call(method: str, path: str, **kwargs):
    """AI 서버로 HTTP 요청. 실패하면 None 반환(연결 불가/타임아웃 등)."""
    url = config.AI_SERVER_URL.rstrip("/") + path
    kwargs.setdefault("timeout", config.AI_SERVER_TIMEOUT)
    try:
        return requests.request(method, url, **kwargs)
    except Exception:
        return None


def _fetch_ai_server_status(user_id: str) -> dict | None:
    """AI 서버에서 해당 user_id 의 현재 작업 상태를 가져온다 (없으면 None)."""
    resp = _ai_server_call("GET", "/api/ai_server/status", params={"user_id": user_id})
    if resp is None or not resp.ok:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def _ai_status_payload(user_id: str, state: dict, *, completed: bool = False,
                       live: dict | None = None, progress_override: int | None = None) -> dict:
    """앱에 돌려줄 통일된 상태 응답을 만든다."""
    if completed:
        progress = 100
        status = "completed"
    elif progress_override is not None:
        progress = progress_override
        status = state.get("status", "idle")
    else:
        progress = max(0, min(100, safe_int(state.get("progress"), 0)))
        status = state.get("status", "idle")
    return {
        "user_id":        user_id,
        "status":         status,
        "progress":       progress,
        "completed":      progress >= 100,
        "queue_position": state.get("queue_position"),
        "current_step":   state.get("current_step"),
        "step_label":     state.get("step_label"),
        "total_steps":    (live or {}).get("total_steps", 4),
        "message":        state.get("message"),
        "job_id":         state.get("job_id"),
    }


def get_person_images(person_id: str, limit: int = 12) -> list:
    person_dir = os.path.join(CLUSTER_DIR, person_id)
    if not os.path.isdir(person_dir):
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([f for f in os.listdir(person_dir)
                    if Path(f).suffix.lower() in exts])
    lifestyle_names = _get_lifestyle_original_names()
    files.sort(key=lambda f: (
        _face_crop_to_original_name(f) not in lifestyle_names,
        f.lower(),
    ))
    return files[:limit]


_lifestyle_original_names_cache = {}   # user별(info_dir별) 캐시


def _get_lifestyle_original_names() -> set:
    key = str(INFO_OUTPUT_DIR)   # 유저마다 다른 경로 → 유저별 캐시
    if key not in _lifestyle_original_names_cache:
        data = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
        _lifestyle_original_names_cache[key] = {
            os.path.basename(item.get("filename", ""))
            for item in data
            if item.get("filename")
        }
    return _lifestyle_original_names_cache[key]


def safe_int(val, default: int = 5) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def load_person_data(person_id: str) -> dict:
    base = os.path.join(AGENT_OUTPUT_DIR, person_id)
    l1 = safe_load_json(os.path.join(base, "level1_observations.json")) or []
    l2 = safe_load_json(os.path.join(base, "level2_patterns.json")) or {}
    l3 = safe_load_json(os.path.join(base, "level3_relations.json")) or {}
    profile = context_expansion = None
    if person_id == "person_0":
        profile = safe_load_json(os.path.join(base, "profile.json"))
        context_expansion = safe_load_json(os.path.join(base, "context_expansion.json"))
    return {
        "person_id": person_id,
        "images": get_person_images(person_id),
        "photo_count": len(l1),
        "level2": l2,
        "level3": l3,
        "profile": profile,
        "context_expansion": context_expansion,
    }


# ── Flask 라우트 ───────────────────────────────────────────────

@app.route("/login", methods=["POST"])
def login():
    """비밀번호 검증 및 세션 생성"""
    data = request.get_json() or {}
    password = data.get("password", "").strip()
    if PASSWORD and password == PASSWORD:
        session['authenticated'] = True
        return jsonify({"success": True, "message": "로그인 성공"})
    return jsonify({"success": False, "error": "비밀번호가 잘못되었습니다"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    """세션 제거 및 로그아웃"""
    session.clear()
    return jsonify({"success": True, "message": "로그아웃 되었습니다"})

@app.route("/")
def index():
    return jsonify({
        "ok": True,
        "service": "lumina-backend (web_app.py 기반 API)",
        "data_dir": str(config.DATA_DIR),
        "auth": "ON" if config.REQUIRE_AUTH else "OFF",
        "endpoints": [
            "POST /upload  (앱→수집)",
            "/api/search", "/api/graph", "/api/all_profiles", "/api/person/<id>",
            "/api/lifestyle/events", "/api/lifestyle/temporal", "/api/lifestyle/yearly",
            "/api/person_names", "/thumbnail/<f>", "/original/<f>", "/image/<pid>/<f>",
        ],
    })


# ── 수집(업로드) — 앱이 사진/동영상을 보내는 곳 ──────────────────
# 모델은 UPLOAD_DIR/<user_id>/ 폴더를 읽어서 분석 → 결과를 AGENT/INFO/CLUSTER 폴더에 씀.
# 즉 이 서버 하나가 [수집 /upload] + [모델(동료)] + [조회 /api/*] 를 다 담당.

@app.route("/upload", methods=["POST"])
def external_upload():
    """앱에서 사진/동영상 + 메타데이터(촬영일시·GPS)를 받아 user_id 폴더에 저장.
    이미지는 640x640 리사이즈본이 와도 메타데이터는 폼 필드로 함께 와서 보존됨."""
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400
    f = request.files["file"]
    user_id = (request.form.get("user_id") or "unknown").strip()
    filename = f.filename or ""
    if not filename:
        return jsonify({"error": "파일명이 없습니다."}), 400
    safe_name = os.path.basename(filename)
    if not safe_name or ".." in safe_name:
        return jsonify({"error": "잘못된 파일명입니다."}), 400

    user_dir = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    dest = os.path.join(user_dir, safe_name)
    if os.path.exists(dest):
        stem, ext = os.path.splitext(safe_name)
        dest = os.path.join(user_dir, f"{stem}_{int(datetime.now().timestamp())}{ext}")
    f.save(dest)

    # 메타데이터를 사이드카(.meta.json)로 보존 → 모델이 촬영일시/GPS 사용 (리사이즈로 EXIF 사라져도 OK)
    meta = {
        "original_filename": safe_name,
        "captured_at": request.form.get("captured_at"),   # ISO8601 (예: 2024-04-28T12:07:52)
        "lat": request.form.get("lat"),
        "lon": request.form.get("lon"),
        "media_type": request.form.get("media_type"),      # "image" | "video"
        # 오프라인 번들용: 폰 내부 사진 위치(기기 식별자). 앱이 보내면 보존 → /api/bundle 의 local_ref 로 환원.
        "asset_id": request.form.get("asset_id"),          # MediaLibrary asset id (1순위)
        "asset_uri": request.form.get("asset_uri"),        # content:// 또는 ph:// (보조)
        "platform": request.form.get("platform"),          # "android" | "ios"
        "uploaded_at": datetime.now().isoformat(),
    }
    try:
        with open(dest + ".meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return jsonify({"ok": True, "saved_filename": os.path.basename(dest), "user_id": user_id})


@app.route("/upload_batch", methods=["POST"])
def external_upload_batch():
    """앱이 640x640 리사이즈 사진 여러 장을 zip 1개로 묶어 보내는 곳 (단건 /upload 의 벌크 버전).
    zip 내부는 flat: 각 사진마다 '<이름>.jpg' + '<이름>.jpg.meta.json' 2개 파일.
    풀어서 uploads/<user_id>/ 에 그대로 떨구면 다운스트림 모델 파이프라인은 수정 없이 동작."""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "파일이 없습니다."}), 400
    f = request.files["file"]
    user_id = (request.form.get("user_id") or "unknown").strip()

    user_dir = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    saved = 0
    skipped = 0
    errors = []
    tmp_path = None
    try:
        # zip 임시 저장
        fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(fd)
        f.save(tmp_path)

        with zipfile.ZipFile(tmp_path) as zf:
            names = zf.namelist()
            # flat 구조: '.meta.json' 은 사이드카, 나머지는 이미지
            image_names = [
                n for n in names
                if not n.endswith("/") and not n.lower().endswith(".meta.json")
            ]
            for entry in image_names:
                safe_name = os.path.basename(entry)
                if not safe_name or ".." in safe_name:
                    skipped += 1
                    errors.append({"file": entry, "error": "잘못된 파일명"})
                    continue

                # 단건 /upload 과 동일한 충돌 처리: 동일명이면 _타임스탬프
                dest = os.path.join(user_dir, safe_name)
                if os.path.exists(dest):
                    stem, ext = os.path.splitext(safe_name)
                    dest = os.path.join(user_dir, f"{stem}_{int(datetime.now().timestamp())}{ext}")

                try:
                    with zf.open(entry) as src, open(dest, "wb") as out:
                        out.write(src.read())
                except Exception as e:
                    skipped += 1
                    errors.append({"file": entry, "error": str(e)})
                    continue

                # 짝이 되는 메타(.meta.json) 를 같은 이름으로 저장 → 이미지와 메타 이름 정렬 유지
                meta_entry = entry + ".meta.json"
                if meta_entry in names:
                    try:
                        with zf.open(meta_entry) as msrc, open(dest + ".meta.json", "wb") as mout:
                            mout.write(msrc.read())
                    except Exception:
                        pass

                saved += 1
    except zipfile.BadZipFile:
        return jsonify({"ok": False, "saved": 0, "error": "잘못된 zip 파일입니다."}), 400
    except Exception as e:
        return jsonify({"ok": False, "saved": saved, "error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return jsonify({"ok": True, "saved": saved, "skipped": skipped, "errors": errors, "user_id": user_id})


# ── 촬영시각 예측 (단일 이미지 → GPU 모델서버 중계) ────────────────
# 앱이 사진 1장을 보내면 GPU 모델서버(time_model_server.py, 기본 127.0.0.1:5000)로
# 그대로 프록시하고 예측 결과(JSON)를 돌려준다. 스테이트리스(저장 안 함).
TIME_MODEL_SERVER_URL = getattr(
    config, "TIME_MODEL_SERVER_URL",
    os.getenv("LUMINA_TIME_MODEL_URL", "http://127.0.0.1:5000"),
)


@app.route("/api/predict_time", methods=["POST"])
def api_predict_time():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "이미지가 없습니다."}), 400
    f = request.files["image"]
    try:
        resp = requests.post(
            TIME_MODEL_SERVER_URL.rstrip("/") + "/predict",
            files={"image": (f.filename or "upload", f.stream, f.mimetype or "image/jpeg")},
            timeout=300,
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"ok": False,
                        "error": f"모델 서버({TIME_MODEL_SERVER_URL}) 연결 실패: {e}"}), 502
    try:
        return jsonify(resp.json()), resp.status_code
    except Exception:
        return jsonify({"ok": False, "error": "모델 서버 응답 파싱 실패"}), 502


# ── 회원가입 / 로그인 (werkzeug 단방향 해시) ───────────────────
# user_id 를 회원 아이디로 사용. 비밀번호는 pbkdf2-sha256+salt 해시로만 저장(평문/복원 불가).
def _load_users() -> dict:
    return safe_load_json(USERS_FILE) or {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


_USER_ID_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "").strip()
    password = data.get("password") or ""
    if not _USER_ID_RE.match(user_id):
        return jsonify({"ok": False, "error": "아이디는 영문/숫자/_ 3~32자여야 합니다."}), 400
    if len(password) < 4:
        return jsonify({"ok": False, "error": "비밀번호는 4자 이상이어야 합니다."}), 400
    with _users_lock:
        users = _load_users()
        if user_id in users:
            return jsonify({"ok": False, "error": "이미 존재하는 아이디입니다."}), 409
        users[user_id] = {
            "pw_hash": generate_password_hash(password),
            "token": None,
            "created_at": datetime.now().isoformat(),
        }
        _save_users(users)
    return jsonify({"ok": True, "user_id": user_id}), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "").strip()
    password = data.get("password") or ""
    with _users_lock:
        users = _load_users()
        u = users.get(user_id)
        if not u or not check_password_hash(u.get("pw_hash", ""), password):
            return jsonify({"ok": False, "error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401
        token = secrets.token_hex(16)
        u["token"] = token
        u["last_login"] = datetime.now().isoformat()
        _save_users(users)
    return jsonify({"ok": True, "user_id": user_id, "token": token})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "").strip()
    with _users_lock:
        users = _load_users()
        if user_id in users:
            users[user_id]["token"] = None
            _save_users(users)
    return jsonify({"ok": True})


# ── 업로드된 사진 목록 / 삭제 (분석 전 관리) ───────────────────
@app.route("/api/uploads", methods=["GET"])
def api_uploads_list():
    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id 가 필요합니다."}), 400
    up_dir = os.path.join(UPLOAD_DIR, user_id)
    files = []
    if os.path.isdir(up_dir):
        # 파일명만 반환 (메타 파싱/stat 생략 → 대량에서도 빠름).
        # 썸네일과 asset_id 는 앱이 폰 갤러리에서 파일명으로 해결한다.
        files = [fn for fn in os.listdir(up_dir) if not fn.endswith(".meta.json")]
        files.sort()
    return jsonify({"count": len(files), "files": files})


@app.route("/api/uploads", methods=["DELETE"])
def api_uploads_delete():
    user_id = (request.args.get("user_id") or "").strip()
    filename = (request.args.get("filename") or "").strip()
    delete_all = (request.args.get("all") or "").strip() in ("1", "true", "True")
    if not user_id:
        return jsonify({"ok": False, "error": "user_id 가 필요합니다."}), 400

    up_dir = os.path.join(UPLOAD_DIR, user_id)

    # 전체 삭제: uploads/<user_id>/ 폴더째 제거 (폰 원본은 유지)
    if delete_all:
        n = 0
        if os.path.isdir(up_dir):
            n = sum(1 for f in os.listdir(up_dir)
                    if os.path.isfile(os.path.join(up_dir, f)) and not f.endswith(".meta.json"))
            import shutil
            shutil.rmtree(up_dir, ignore_errors=True)
        return jsonify({"ok": True, "deleted_count": n})

    # 개별 삭제
    if not filename:
        return jsonify({"ok": False, "error": "filename 이 필요합니다."}), 400
    safe = os.path.basename(filename)
    if not safe or ".." in safe:
        return jsonify({"ok": False, "error": "잘못된 파일명입니다."}), 400
    up_dir = os.path.join(UPLOAD_DIR, user_id)
    target = os.path.join(up_dir, safe)
    deleted = []
    for p in (target, target + ".meta.json"):
        if os.path.isfile(p):
            try:
                os.remove(p)
                deleted.append(os.path.basename(p))
            except Exception as e:
                return jsonify({"ok": False, "error": str(e)}), 500
    if not deleted:
        return jsonify({"ok": False, "error": "파일을 찾을 수 없습니다."}), 404
    return jsonify({"ok": True, "deleted": deleted})


@app.route("/api/status")
def api_status():
    """분석 진행 상태. 앱이 폴링해서 '분석중/완료' 를 판단."""
    user_id = (request.args.get("user_id") or "default").strip()

    # 업로드된 수 (uploads/<user_id>/ 의 .meta.json 제외)
    uploaded = 0
    user_dir = os.path.join(UPLOAD_DIR, user_id)
    if os.path.isdir(user_dir):
        uploaded = len([f for f in os.listdir(user_dir)
                        if os.path.isfile(os.path.join(user_dir, f)) and not f.endswith(".meta.json")])

    # 모델이 만든 결과 (deep_analyses.json 항목 수 + 그래프 존재 여부)
    analyses = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
    analyzed = len(analyses)
    graph_ready = os.path.isfile(os.path.join(AGENT_OUTPUT_DIR, "knowledge_graph.json"))
    ready = analyzed > 0 and graph_ready

    return jsonify({
        "user_id": user_id,
        "uploaded": uploaded,
        "analyzed": analyzed,
        "graph_ready": graph_ready,
        "ready": ready,
        "progress": round(min(1.0, analyzed / uploaded), 3) if uploaded else (1.0 if ready else 0.0),
    })


@app.route("/api/ai_server/start", methods=["POST"])
@app.route("/api/ai_server/signal", methods=["POST"])
def api_ai_server_start():
    """앱 요청을 받아 AI 서버(별도 컴퓨터)에 작업 시작을 전달하고 큐에 등록시킨다."""
    user_id = _request_user_id()
    now = datetime.now().isoformat()
    state = _load_ai_server_state(user_id)

    resp = _ai_server_call("POST", "/api/ai_server/start", json={"user_id": user_id})
    if resp is None or not resp.ok:
        # AI 서버 연결 실패/오류 → 상태에 기록하고 502 반환
        state.update({
            "user_id": user_id,
            "requested": True,
            "status": "ai_server_unreachable",
            "requested_at": state.get("requested_at") or now,
            "updated_at": now,
        })
        _save_ai_server_state(user_id, state)
        detail = ""
        if resp is not None:
            try:
                detail = (resp.json() or {}).get("error", "")
            except Exception:
                detail = (resp.text or "")[:200]
        return jsonify({
            "ok": False,
            "user_id": user_id,
            "status": state["status"],
            "error": "AI 서버에 연결할 수 없습니다." if resp is None else f"AI 서버 오류: {detail}",
        }), 502

    data = resp.json() if resp.content else {}
    job = data.get("job") or {}
    state.update({
        "user_id": user_id,
        "requested": True,
        "job_id": job.get("job_id"),
        "status": job.get("status", "requested"),
        "progress": max(0, min(100, safe_int(job.get("progress"), 0))),
        "queue_position": job.get("queue_position"),
        "current_step": job.get("current_step"),
        "step_label": job.get("step_label"),
        "message": job.get("message"),
        "requested_at": state.get("requested_at") or now,
        "updated_at": now,
    })
    _save_ai_server_state(user_id, state)
    return jsonify({
        "ok": True,
        "user_id": user_id,
        "duplicated": data.get("duplicated", False),
        "job": job,
        "status": state["status"],
        "progress": state["progress"],
        "queue_position": state.get("queue_position"),
    }), (200 if data.get("duplicated") else 202)


@app.route("/api/push_token", methods=["POST"])
def api_push_token():
    """앱이 발급한 Expo 푸시 토큰을 user_id 별로 저장(같은 user_id 면 덮어쓰기)."""
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "").strip()
    token = (data.get("token") or "").strip()
    if not user_id or not token:
        return jsonify({"error": "user_id, token required"}), 400
    with _push_tokens_lock:
        tokens = _load_push_tokens()
        tokens[user_id] = token
        _save_push_tokens(tokens)
    return jsonify({"ok": True})


@app.route("/api/ai_server/progress", methods=["GET"])
@app.route("/api/ai_server/status", methods=["GET"])
def api_ai_server_progress():
    """AI 서버의 현재 단계/진행률(0~100)을 조회해 앱에 돌려준다.
    결과 파일이 이미 도착했으면 완료(100)로 본다."""
    user_id = _request_user_id()
    state = _load_ai_server_state(user_id)
    now = datetime.now().isoformat()

    # 1) 결과(deep_analyses.json + knowledge_graph.json)가 실제로 도착 → 완료 확정
    if _analysis_ready():
        if state.get("progress") != 100 or state.get("status") != "completed":
            state.update({
                "user_id": user_id, "status": "completed", "progress": 100,
                "completed_at": state.get("completed_at") or now, "updated_at": now,
            })
            _save_ai_server_state(user_id, state)
        _maybe_send_completion_push(user_id, state)  # 폴링 경로 폴백(앱이 한 번이라도 조회 시)
        return jsonify(_ai_status_payload(user_id, state, completed=True))

    # 2) AI 서버에 라이브 상태 질의 → 상태 캐시에 반영
    live = _fetch_ai_server_status(user_id)
    if live:
        state.update({
            "user_id":        user_id,
            "job_id":         live.get("job_id", state.get("job_id")),
            "status":         live.get("status", state.get("status", "idle")),
            "progress":       max(0, min(100, safe_int(live.get("progress"), 0))),
            "queue_position": live.get("queue_position"),
            "current_step":   live.get("current_step"),
            "step_label":     live.get("step_label"),
            "message":        live.get("message"),
            "updated_at":     now,
        })
        _save_ai_server_state(user_id, state)
        return jsonify(_ai_status_payload(user_id, state, live=live))

    # 3) AI 서버 불통 → 마지막으로 보고/저장된 값으로 폴백
    progress = _ai_server_progress(user_id)
    return jsonify(_ai_status_payload(user_id, state, progress_override=progress))


@app.route("/api/ai_server/progress", methods=["POST"])
def api_ai_server_update_progress():
    """Allow the future AI server to publish progress, clamped to 0..100."""
    data = request.get_json(silent=True) or {}
    user_id = _request_user_id()
    progress = max(0, min(100, safe_int(data.get("progress"), 0)))
    state = _load_ai_server_state(user_id)
    state.update({
        "user_id": user_id,
        "requested": state.get("requested", False),
        "status": "completed" if progress >= 100 else "running",
        "progress": progress,
        "updated_at": datetime.now().isoformat(),
    })
    if progress >= 100:
        state["completed_at"] = state.get("completed_at") or state["updated_at"]
    _save_ai_server_state(user_id, state)
    if progress >= 100:
        # AI 서버가 완료(100)를 통보하는 주 경로 → 앱이 닫혀 있어도 여기서 푸시 발송
        _maybe_send_completion_push(user_id, state)
    return jsonify({
        "ok": True,
        "user_id": user_id,
        "status": state["status"],
        "progress": progress,
    })


# ── 관계 분석 API ──────────────────────────────────────────────

@app.route("/api/graph")
@login_required
def api_graph():
    kg = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, "knowledge_graph.json"))
    if not kg:
        return jsonify({"error": "knowledge_graph.json 파일을 찾을 수 없습니다."}), 404

    node_map: dict = {}
    for node in kg["nodes"]:
        nid = node["id"]
        if nid in node_map:
            node_map[nid].update({k: v for k, v in node.items() if v not in (None, "", [], {})})
        else:
            node_map[nid] = dict(node)

    vis_nodes, vis_edges = [], []
    for node in node_map.values():
        nid = node["id"]
        is_me = (nid == "person_0")
        rel = node.get("relationship_type", "알수없음")
        score = safe_int(node.get("intimacy_score", 5))
        age = node.get("estimated_age", "")
        color = "#FFD700" if is_me else RELATION_COLORS.get(rel, "#666677")
        age_line = f"\n{age}" if age and age not in ("알수없음", "") else ""
        vis_nodes.append({
            "id": nid,
            "label": "나" if is_me else f"{nid}\n{rel}{age_line}",
            "title": ("나 (사진 소유자)" if is_me
                      else f"{rel} · 친밀도 {score} · {age} {node.get('gender','')} · {node.get('photo_count',0)}장"),
            "color": {"background": color, "border": "#ffffff55",
                      "highlight": {"background": "#ffffff", "border": color},
                      "hover": {"background": color, "border": "#ffffff"}},
            "size": 48 if is_me else max(18, score * 3),
            "font": {"color": "#ffffff", "size": 14 if is_me else 11,
                     "bold": is_me, "face": "Segoe UI"},
            "borderWidth": 3 if is_me else 1,
            "shape": "dot",
            "shadow": {"enabled": True, "size": 20, "color": color + "55"},
            "meta": node,
        })

    for edge in kg["edges"]:
        weight = safe_int(edge.get("weight", 5))
        rel = edge.get("relation", "")
        tags = ", ".join(edge.get("tags", []))
        rel_color = RELATION_COLORS.get(rel, "#aaaaaa")
        vis_edges.append({
            "from": edge.get("from", "person_0"),
            "to": edge.get("to"),
            "label": rel,
            "title": f"관계: {rel}\n친밀도: {weight}\n태그: {tags}",
            "width": max(1, weight // 3),
            "color": {"color": rel_color + "66", "highlight": rel_color, "hover": rel_color},
            "font": {"color": "#cccccc", "size": 10, "align": "top", "face": "Segoe UI"},
            "smooth": {"type": "curvedCW", "roundness": 0.2},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
        })

    return jsonify({"nodes": vis_nodes, "edges": vis_edges, "total": len(vis_nodes) - 1})


@app.route("/api/person/<person_id>")
@login_required
def api_person(person_id: str):
    if ".." in person_id:
        abort(400)
    return jsonify(load_person_data(person_id))


@app.route("/api/person_zero_combined")
@login_required
def api_person_zero_combined():
    base = os.path.join(AGENT_OUTPUT_DIR, "person_0")
    l1 = safe_load_json(os.path.join(base, "level1_observations.json")) or []
    return jsonify({
        "profile":           safe_load_json(os.path.join(base, "profile.json")) or {},
        "level2":            safe_load_json(os.path.join(base, "level2_patterns.json")) or {},
        "context_expansion": safe_load_json(os.path.join(base, "context_expansion.json")) or {},
        "images":            get_person_images("person_0", limit=8),
        "photo_count":       len(l1),
        "style_summary":     safe_load_json(os.path.join(INFO_OUTPUT_DIR, "style_summary.json")) or {},
        "style_stats":       safe_load_json(os.path.join(INFO_OUTPUT_DIR, "style_stats.json")) or {},
    })


@app.route("/api/merges")
@login_required
def api_merges():
    data = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, "merge_candidates.json")) or []
    enriched = []
    for item in data:
        entry = dict(item)
        for key, suffix in [("person_a", "a"), ("person_b", "b")]:
            pid = item.get(key, "")
            entry[f"images_{suffix}"] = get_person_images(pid, limit=3)
            l2 = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, pid, "level2_patterns.json")) or {}
            l3 = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, pid, "level3_relations.json")) or {}
            entry[f"age_{suffix}"] = l2.get("estimated_age", "알수없음")
            entry[f"gender_{suffix}"] = l2.get("gender", "불명확")
            entry[f"rel_{suffix}"] = l3.get("relationship_type", "알수없음")
        enriched.append(entry)
    return jsonify(enriched)


@app.route("/api/people_mapping")
@login_required
def api_people_mapping():
    import csv as csv_mod
    csv_path = os.path.join(CLUSTER_DIR, "image_people_mapping.csv")
    mapping: dict = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                name = row.get("image_name", "").strip()
                people_str = row.get("people", "").strip()
                if name and people_str:
                    unique = list(dict.fromkeys(p.strip() for p in people_str.split(",") if p.strip()))
                    mapping[name] = unique
    except Exception:
        pass
    return jsonify(mapping)


@app.route("/api/filtered")
@login_required
def api_filtered():
    data = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, "artwork_filtered.json")) or []
    return jsonify([{**item, "images": get_person_images(item.get("person_id", ""), limit=3)}
                    for item in data])


@app.route("/api/all_profiles")
@login_required
def api_all_profiles():
    kg = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, "knowledge_graph.json")) or {"nodes": [], "edges": []}
    edge_map = {e.get("to"): e for e in kg.get("edges", [])}
    seen: set = set()
    profiles = []
    for node in kg.get("nodes", []):
        nid = node.get("id")
        if nid in seen:
            continue
        seen.add(nid)
        edge = edge_map.get(nid, {})
        profiles.append({
            "person_id":             nid,
            "estimated_age":         node.get("estimated_age", ""),
            "gender":                node.get("gender", ""),
            "relationship_type":     node.get("relationship_type", "알수없음"),
            "intimacy_score":        safe_int(node.get("intimacy_score", 0)),
            "lifestyle_summary":     node.get("lifestyle_summary", ""),
            "relationship_narrative":node.get("relationship_narrative", ""),
            "age_based_reasoning":   node.get("age_based_reasoning", ""),
            "photo_count":           node.get("photo_count", 0),
            "edge_tags":             edge.get("tags", []),
            "images":                get_person_images(nid, limit=4),
        })
    profiles.sort(key=lambda x: x["intimacy_score"], reverse=True)
    return jsonify(profiles)


# ── 나(person_0) 중심 동행 랭킹 ─────────────────────────────────

def _person_zero_relation_map() -> dict:
    """knowledge_graph.json 의 nodes 를 person_id → 관계정보 로 정리."""
    kg = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, "knowledge_graph.json")) or {}
    rel = {}
    for node in kg.get("nodes", []):
        nid = node.get("id")
        if not nid:
            continue
        rel[nid] = {
            "relationship_type": node.get("relationship_type", "알수없음"),
            "intimacy_score": safe_int(node.get("intimacy_score", 0)),
            "estimated_age": node.get("estimated_age", ""),
            "gender": node.get("gender", ""),
            "relationship_narrative": node.get("relationship_narrative", ""),
        }
    return rel


@app.route("/api/person_zero_ranking")
@login_required
def api_person_zero_ranking():
    """image_people_mapping.csv 공동출현 집계 → 나와 함께 찍은 인물 랭킹/분포/연도별 등."""
    import csv as csv_mod
    from collections import defaultdict

    rel_map = _person_zero_relation_map()

    # deep_analyses: image basename → datetime (날짜 결합용)
    analyses = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
    date_of = {}
    for p in analyses:
        bn = os.path.basename(p.get("filename", ""))
        if bn and p.get("datetime"):
            date_of[bn] = p["datetime"]

    csv_path = os.path.join(CLUSTER_DIR, "image_people_mapping.csv")
    cooccur = defaultdict(lambda: {"count": 0, "first": None, "last": None})
    yearly = defaultdict(lambda: defaultdict(int))   # year → person → count
    my_photos = solo_photos = 0
    all_persons = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                img = (row.get("image_name") or "").strip()
                people = {x.strip() for x in (row.get("people") or "").split(",") if x.strip()}
                all_persons |= people
                if "person_0" not in people:
                    continue
                my_photos += 1
                others = people - {"person_0"}
                if not others:
                    solo_photos += 1
                    continue
                dt = date_of.get(os.path.basename(img))
                year = dt[:4] if dt and len(dt) >= 4 else None
                for o in others:
                    c = cooccur[o]
                    c["count"] += 1
                    if dt:
                        if c["first"] is None or dt < c["first"]: c["first"] = dt
                        if c["last"] is None or dt > c["last"]: c["last"] = dt
                    if year:
                        yearly[year][o] += 1
    except FileNotFoundError:
        return jsonify({"error": "image_people_mapping.csv 를 찾을 수 없습니다."}), 404
    except Exception as e:
        return jsonify({"error": f"csv 읽기 실패: {e}"}), 500

    # ranking
    ranking = []
    for pid, c in sorted(cooccur.items(), key=lambda kv: kv[1]["count"], reverse=True):
        info = rel_map.get(pid, {})
        imgs = get_person_images(pid, limit=1)
        ranking.append({
            "person_id": pid,
            "count": c["count"],
            "relationship_type": info.get("relationship_type", "알수없음"),
            "intimacy_score": info.get("intimacy_score", 0),
            "estimated_age": info.get("estimated_age", ""),
            "gender": info.get("gender", ""),
            "first_date": (c["first"] or "")[:10],
            "last_date": (c["last"] or "")[:10],
            "thumbnail": f"/image/{pid}/{imgs[0]}" if imgs else None,
        })

    # distribution: 관계유형별 인원/사진 수
    dist = defaultdict(lambda: {"persons": 0, "photos": 0})
    for r in ranking:
        d = dist[r["relationship_type"]]
        d["persons"] += 1
        d["photos"] += r["count"]
    distribution = sorted(
        [{"relationship_type": k, **v} for k, v in dist.items()],
        key=lambda x: x["photos"], reverse=True,
    )

    # 친밀도 TOP 5
    intimacy_top = sorted(ranking, key=lambda r: r["intimacy_score"], reverse=True)[:5]

    # 연도별 최다 동행
    yearly_best = []
    for year in sorted(yearly):
        best_pid, best_cnt = max(yearly[year].items(), key=lambda kv: kv[1])
        info = rel_map.get(best_pid, {})
        imgs = get_person_images(best_pid, limit=1)
        yearly_best.append({
            "year": year, "person_id": best_pid, "count": best_cnt,
            "relationship_type": info.get("relationship_type", "알수없음"),
            "thumbnail": f"/image/{best_pid}/{imgs[0]}" if imgs else None,
        })

    summary = {
        "total_persons": len(all_persons - {"person_0"}),
        "photographed_with": len(cooccur),
        "my_photos": my_photos,
        "solo_photos": solo_photos,
    }

    return jsonify({
        "ranking": ranking,
        "distribution": distribution,
        "intimacy_top": intimacy_top,
        "yearly_best": yearly_best,
        "summary": summary,
    })


# ── 라이프스타일 API ────────────────────────────────────────────

@app.route("/api/lifestyle/events")
@login_required
def api_lifestyle_events():
    data = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
    events = []
    for i, p in enumerate(data):
        dt_raw = p.get("datetime", "")
        ts_ms = 0
        try:
            if isinstance(dt_raw, str) and dt_raw:
                d = datetime.fromisoformat(dt_raw)
                ts_ms = int(d.timestamp() * 1000)
        except Exception:
            pass
        stem = Path(p.get("filename", "unknown")).stem
        events.append({
            "id": i,
            "filename":     p.get("filename", ""),
            "datetime":     dt_raw,
            "timestamp_ms": ts_ms,
            "lat":          p.get("lat"),
            "lon":          p.get("lon"),
            "analysis":     p.get("analysis", {}),
            "thumbnail_path": f"/thumbnail/{i:05d}_{stem}.jpg",
        })
    return jsonify({"events": events})


def _face_crop_to_original_name(filename: str) -> str:
    """person_x 얼굴 crop 파일명에서 원본 사진 파일명을 추정한다."""
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    stem = re.sub(r'_\d+$', '', stem)
    return stem + ext


@app.route("/api/photo_event/<path:filename>")
@login_required
def api_photo_event(filename: str):
    if ".." in filename:
        abort(400)
    target = _face_crop_to_original_name(filename)
    data = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
    for i, p in enumerate(data):
        base = os.path.basename(p.get("filename", ""))
        if base != target:
            continue
        ts_ms = 0
        dt_raw = p.get("datetime", "")
        try:
            if dt_raw:
                ts_ms = int(datetime.fromisoformat(dt_raw).timestamp() * 1000)
        except Exception:
            pass
        stem = Path(base or "unknown").stem
        return jsonify({
            "id": i,
            "filename": base,
            "timestamp_ms": ts_ms,
            "thumbnail_path": f"/thumbnail/{i:05d}_{stem}.jpg",
        })
    return jsonify({"error": f"{target}에 해당하는 라이프스타일 사진을 찾지 못했습니다."}), 404


@app.route("/api/lifestyle/summary")
@login_required
def api_lifestyle_summary():
    return jsonify(safe_load_json(os.path.join(INFO_OUTPUT_DIR, "style_summary.json")) or {})


@app.route("/api/lifestyle/stats")
@login_required
def api_lifestyle_stats():
    """음식/풍경/사회성/활동 통계 (style_stats.json) — 라이프스타일 막대그래프용."""
    return jsonify(safe_load_json(os.path.join(INFO_OUTPUT_DIR, "style_stats.json")) or {})


@app.route("/api/lifestyle/temporal")
@login_required
def api_lifestyle_temporal():
    return jsonify(safe_load_json(os.path.join(INFO_OUTPUT_DIR, "temporal_stats.json")) or {})


@app.route("/api/lifestyle/yearly")
@login_required
def api_lifestyle_yearly():
    return jsonify(safe_load_json(os.path.join(INFO_OUTPUT_DIR, "yearly_summaries.json")) or {})


# ── 오프라인 번들 (모든 결과 JSON 1개 + 사진은 폰 내부 위치 참조) ──
# 설계: OFFLINE_BUNDLE_GUIDE.md
# 원칙: sections.* 는 기존 /api/* 응답과 동일 구조(뷰 함수를 그대로 호출해 담음) →
#       앱은 데이터 출처만 '네트워크 → 로컬 번들'로 스위치. 사진만 local_ref(기기 위치) 추가.
BUNDLE_SCHEMA_VERSION = 1


def _rv_json(rv):
    """뷰 함수 반환값(Response 또는 (Response, status))에서 JSON(dict/list)을 추출."""
    if isinstance(rv, tuple):
        rv = rv[0]
    try:
        return rv.get_json(silent=True)
    except Exception:
        return None


def _photo_entry(filename: str, analysis: dict | None, meta: dict | None) -> dict:
    """사진 1장의 번들 항목. local_ref(폰 위치) 우선, online(서버 URL)은 폴백용."""
    analysis = analysis or {}
    meta = meta or {}
    local_ref = None
    if meta.get("asset_id") or meta.get("asset_uri"):
        local_ref = {
            "asset_id":  meta.get("asset_id"),
            "asset_uri": meta.get("asset_uri"),
            "platform":  meta.get("platform"),
        }
    return {
        "filename":       filename,
        "saved_filename": meta.get("saved_filename", filename),
        "captured_at":    analysis.get("datetime") or meta.get("captured_at"),
        "lat":            analysis.get("lat", meta.get("lat")),
        "lon":            analysis.get("lon", meta.get("lon")),
        "local_ref":      local_ref,   # photo_policy=device_local: 폰 갤러리에서 직접 로드
        "online": {                    # 네트워크 있을 때 폴백용
            "original":  f"/original/{filename}",
            "thumbnail": f"/thumbnail/{filename}",
        },
    }


def _build_photo_index(user_id: str) -> list:
    """분석된 사진(deep_analyses)을 기준으로, 업로드 메타(uploads/<user>/*.meta.json)의
    asset_id/asset_uri/platform 과 filename 으로 조인해 photos[] 를 만든다."""
    # 1) 업로드 메타 인덱싱 (saved_filename / original_filename 두 키로 룩업 가능하게)
    meta_by_name: dict = {}
    up_dir = os.path.join(UPLOAD_DIR, user_id)
    if os.path.isdir(up_dir):
        suffix = ".meta.json"
        for fn in os.listdir(up_dir):
            if not fn.endswith(suffix):
                continue
            m = safe_load_json(os.path.join(up_dir, fn)) or {}
            saved = fn[:-len(suffix)]               # 사이드카가 가리키는 이미지 저장명
            orig = m.get("original_filename") or saved
            entry = {
                "asset_id":       m.get("asset_id"),
                "asset_uri":      m.get("asset_uri"),
                "platform":       m.get("platform"),
                "captured_at":    m.get("captured_at"),
                "lat":            m.get("lat"),
                "lon":            m.get("lon"),
                "saved_filename": saved,
            }
            meta_by_name[saved] = entry
            meta_by_name.setdefault(orig, entry)

    # 2) 분석된 사진 기준으로 photos[] 구성
    analyses = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
    photos = []
    used: set = set()
    for a in analyses:
        fn = os.path.basename(a.get("filename") or "")
        if not fn or fn in used:
            continue
        used.add(fn)
        photos.append(_photo_entry(fn, a, meta_by_name.get(fn)))
    return photos


def build_user_bundle(user_id: str) -> dict:
    """흩어진 결과 JSON/CSV 를 기존 API 응답과 동일 구조로 묶은 단일 번들 dict.
    (현재 요청의 g.user_dirs 가 user_id 로 바인딩된 상태에서 호출되어야 한다.)"""
    # 인물 목록: knowledge_graph nodes (중복 제거, 등장 순서 유지)
    kg = safe_load_json(os.path.join(AGENT_OUTPUT_DIR, "knowledge_graph.json")) or {}
    person_ids: list = []
    seen: set = set()
    for node in kg.get("nodes", []):
        nid = node.get("id")
        if nid and nid not in seen:
            seen.add(nid)
            person_ids.append(nid)
    persons = {pid: load_person_data(pid) for pid in person_ids}

    sections = {
        "graph":                _rv_json(api_graph()),
        "all_profiles":         _rv_json(api_all_profiles()),
        "person_zero_combined": _rv_json(api_person_zero_combined()),
        "person_zero_ranking":  _rv_json(api_person_zero_ranking()),
        "people_mapping":       _rv_json(api_people_mapping()),
        "merges":               _rv_json(api_merges()),
        "filtered":             _rv_json(api_filtered()),
        "person_names":         _rv_json(api_get_person_names()),
        "lifestyle": {
            "events":   _rv_json(api_lifestyle_events()),
            "summary":  _rv_json(api_lifestyle_summary()),
            "stats":    _rv_json(api_lifestyle_stats()),
            "temporal": _rv_json(api_lifestyle_temporal()),
            "yearly":   _rv_json(api_lifestyle_yearly()),
        },
        "persons": persons,   # = GET /api/person/<id> 들을 미리 펼쳐 담음
    }

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "user_id":        user_id,
        "photo_policy":   "device_local",
        "sections":       sections,
        "photos":         _build_photo_index(user_id),
    }


@app.route("/api/bundle", methods=["GET"])
@login_required
def api_bundle():
    """모든 분석 결과를 JSON 1개로 내려준다(오프라인 번들). gzip 지원."""
    user_id = _request_user_id()
    if not _analysis_ready():
        return jsonify({"error": "아직 분석 결과가 없습니다.", "user_id": user_id}), 404

    bundle = build_user_bundle(user_id)
    raw = json.dumps(bundle, ensure_ascii=False).encode("utf-8")

    if "gzip" in (request.headers.get("Accept-Encoding") or "").lower():
        resp = Response(gzip.compress(raw), mimetype="application/json")
        resp.headers["Content-Encoding"] = "gzip"
        resp.headers["Vary"] = "Accept-Encoding"
    else:
        resp = Response(raw, mimetype="application/json")
    resp.headers["X-Bundle-Schema"] = str(BUNDLE_SCHEMA_VERSION)
    return resp


# ── 파일 서빙 ───────────────────────────────────────────────────

@app.route("/thumbnail/<path:filename>")
@login_required
def serve_thumbnail(filename: str):
    if ".." in filename:
        abort(400)
    path = os.path.join(THUMBNAILS_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


@app.route("/original/<path:filename>")
@login_required
def serve_original(filename: str):
    if ".." in filename:
        abort(400)
    basename = os.path.basename(filename)
    # 원본 폴더에서 검색 (서브폴더 포함)
    for root, _, files in os.walk(ORIGINAL_DIR):
        if basename in files:
            return send_file(os.path.join(root, basename))
    abort(404)


def _load_names() -> dict:
    try:
        with open(NAMES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_names(names: dict) -> None:
    with open(NAMES_FILE, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)


@app.route("/api/person_names", methods=["GET"])
@login_required
def api_get_person_names():
    return jsonify(_load_names())


@app.route("/api/person_names", methods=["POST"])
@login_required
def api_set_person_name():
    data = request.get_json(force=True) or {}
    person_id = (data.get("person_id") or "").strip()
    name      = (data.get("name") or "").strip()
    if not person_id:
        return jsonify({"error": "person_id required"}), 400
    names = _load_names()
    if name:
        names[person_id] = name
    else:
        names.pop(person_id, None)
    _save_names(names)
    return jsonify({"ok": True})


# ── 파일 관리 API ──────────────────────────────────────────────

def _load_file_meta() -> dict:
    try:
        with open(FILE_META, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_file_meta(meta: dict) -> None:
    with open(FILE_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


@app.route("/api/files", methods=["GET"])
@login_required
def api_list_files():
    meta = _load_file_meta()
    files = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        fp = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(fp):
            continue
        stat = os.stat(fp)
        files.append({
            "name":        name,
            "size":        stat.st_size,
            "modified":    datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "description": meta.get(name, ""),
        })
    return jsonify(files)


@app.route("/api/files/upload", methods=["POST"])
@login_required
def api_upload_file():
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400
    f = request.files["file"]
    filename = f.filename or ""
    if not filename:
        return jsonify({"error": "파일명이 없습니다."}), 400
    # 안전한 파일명 생성 (경로 순회 방지)
    safe_name = os.path.basename(filename)
    if not safe_name or ".." in safe_name:
        return jsonify({"error": "잘못된 파일명입니다."}), 400
    # 중복 파일명 처리
    dest = os.path.join(UPLOAD_DIR, safe_name)
    if os.path.exists(dest):
        stem, ext = os.path.splitext(safe_name)
        ts = int(datetime.now().timestamp())
        safe_name = f"{stem}_{ts}{ext}"
        dest = os.path.join(UPLOAD_DIR, safe_name)
    f.save(dest)
    description = (request.form.get("description") or "").strip()
    if description:
        meta = _load_file_meta()
        meta[safe_name] = description
        _save_file_meta(meta)
    return jsonify({"ok": True, "name": safe_name})


@app.route("/api/files/<path:filename>", methods=["GET"])
@login_required
def api_download_file(filename: str):
    if ".." in filename:
        abort(400)
    safe_name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=safe_name)


@app.route("/api/files/<path:filename>/description", methods=["POST"])
@login_required
def api_update_file_description(filename: str):
    if ".." in filename:
        abort(400)
    safe_name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.isfile(path):
        abort(404)
    data = request.get_json(force=True) or {}
    description = (data.get("description") or "").strip()
    meta = _load_file_meta()
    if description:
        meta[safe_name] = description
    else:
        meta.pop(safe_name, None)
    _save_file_meta(meta)
    return jsonify({"ok": True})


@app.route("/api/files/<path:filename>", methods=["DELETE"])
@login_required
def api_delete_file(filename: str):
    if ".." in filename:
        abort(400)
    safe_name = os.path.basename(filename)
    path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.isfile(path):
        abort(404)
    os.remove(path)
    meta = _load_file_meta()
    meta.pop(safe_name, None)
    _save_file_meta(meta)
    return jsonify({"ok": True})


# ── 검색 기능 ──────────────────────────────────────────────────

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        import vertexai
        from google.oauth2 import service_account
        from langchain_google_vertexai import ChatVertexAI
        # 서버 컴퓨터의 web_app.py 와 같은 경로에 있는 gcp-key.json 서비스 계정 키 사용
        key_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        vertexai.init(
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location="us-central1",
            credentials=credentials,
        )
        _llm = ChatVertexAI(
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
            max_output_tokens=2048,
            credentials=credentials,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location="us-central1",
        )
    return _llm


def _invoke_llm(prompt: str) -> str:
    from langchain_core.messages import HumanMessage
    for attempt in range(3):
        try:
            return _get_llm().invoke([HumanMessage(content=prompt)]).content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise e


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _load_people_mapping() -> dict:
    csv_path = os.path.join(CLUSTER_DIR, "image_people_mapping.csv")
    mapping: dict = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("image_name", "").strip()
                people_str = row.get("people", "").strip()
                if name and people_str:
                    mapping[name] = [p.strip() for p in people_str.split(",") if p.strip()]
    except Exception:
        pass
    return mapping


def _parse_query(query: str, person_names: dict) -> dict:
    """LLM으로 자연어 질문 → 구조화된 검색 파라미터 변환"""
    name_map_str = json.dumps(person_names, ensure_ascii=False) if person_names else "{}"
    current_year = datetime.now().year
    prompt = f"""당신은 사진 검색 도우미입니다. 사용자의 자연어 검색 질문을 분석해서 반드시 아래 형식의 JSON만 반환하세요.

사용자 질문: "{query}"

현재 연도: {current_year}
등록된 인물 이름 (person_id → 이름): {name_map_str}

반환 형식 (JSON만, 설명 없이):
{{
  "date_range": {{
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD"
  }},
  "location": {{
    "name": "장소명",
    "lat": 위도숫자,
    "lon": 경도숫자,
    "radius_km": 반경숫자
  }},
  "people": ["person_id_또는_이름"],
  "date_months": [월숫자],
  "object_terms": ["사용자가 찾는 물건/객체/의류/소품/음식"],
  "activity_terms": ["사용자가 찾는 행동/활동/상황"]
}}

규칙:
- 날짜가 없으면 date_range는 null
- "12월", "5월달"처럼 연도 없이 월만 말하면 date_range는 null, date_months는 [12] 또는 [5]
- "2018년 12월"처럼 연도가 있으면 date_range를 2018-12-01~2018-12-31로 반환하고 date_months는 []
- 장소가 없으면 location은 null
- 사람이 없으면 people는 []
- 월 단독 조건이 없으면 date_months는 []
- 물건/객체/의류/소품/음식이 없으면 object_terms는 []
- 행동/활동/상황이 없으면 activity_terms는 []
- 날짜: "작년"→전년도, "여름"→6~8월, "2018년"→2018-01-01~2018-12-31 등으로 변환
- 위치: 도시면 radius_km=15, 관광지/특정장소면 radius_km=1, 나라면 radius_km=300
- 사람: 등록된 이름 목록에서 매칭되면 해당 person_id 반환, 없으면 이름 그대로 반환
- object_terms 예: 모자, 안경, 마스크, 케이크, 스키 장비, 자동차, 바다, 눈, 음식, 배낭, 간판
- "모자 쓰고 있는", "안경 낀", "마스크 착용한" 같은 의류/소품 착용 표현은 activity_terms가 아니라 object_terms에만 넣기
- activity_terms 예: 업무, 여행, 식사, 스키, 산책, 생일파티, 쉬는 모습, 운전, 공연 관람, 학술활동, 문화 활동
- 위도/경도는 스마트폰 GPS 기준 실제 좌표 (십진도)"""

    try:
        text = _invoke_llm(prompt)
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            parsed = json.loads(match.group())
            # 정규화
            if not isinstance(parsed.get("date_range"), dict):
                parsed["date_range"] = None
            if not isinstance(parsed.get("location"), dict):
                parsed["location"] = None
            if not isinstance(parsed.get("people"), list):
                parsed["people"] = []
            if not isinstance(parsed.get("date_months"), list):
                parsed["date_months"] = []
            if not isinstance(parsed.get("object_terms"), list):
                parsed["object_terms"] = []
            if not isinstance(parsed.get("activity_terms"), list):
                parsed["activity_terms"] = []
            parsed = _augment_object_terms_from_query(query, parsed)
            parsed = _normalize_month_only_query(query, parsed)
            parsed = _normalize_object_activity_terms(parsed)
            return parsed
    except Exception:
        pass
    parsed = _augment_object_terms_from_query(
        query,
        {"date_range": None, "location": None, "people": [], "date_months": [], "object_terms": [], "activity_terms": []},
    )
    return _normalize_object_activity_terms(_normalize_month_only_query(
        query,
        parsed,
    ))


def _augment_object_terms_from_query(query: str, parsed: dict) -> dict:
    q = query or ""
    terms = [str(t).strip() for t in (parsed.get("object_terms") or []) if str(t).strip()]
    known_objects = ["모자", "안경", "마스크", "선글라스", "가방", "배낭", "자동차", "차", "케이크", "음식", "스키", "스키장비"]
    for obj in known_objects:
        if obj in q and obj not in terms:
            terms.append(obj)
    for match in re.findall(r'([가-힣A-Za-z0-9]+)\s*(?:을|를|이|가|은|는)?\s*(?:쓰고|쓴|착용|입고|신고|낀|걸친|두른)', q):
        obj = re.sub(r'(을|를|이|가|은|는)$', '', match.strip())
        if len(obj) >= 2 and obj not in terms:
            terms.append(obj)
    parsed["object_terms"] = terms
    return parsed


def _normalize_object_activity_terms(parsed: dict) -> dict:
    """착용 표현은 활동이 아니라 객체 조건으로 취급한다."""
    object_terms = [str(t).strip() for t in (parsed.get("object_terms") or []) if str(t).strip()]
    activity_terms = [str(t).strip() for t in (parsed.get("activity_terms") or []) if str(t).strip()]
    if not object_terms:
        parsed["activity_terms"] = activity_terms
        return parsed

    wear_re = re.compile(r'(쓰고|쓴|착용|입고|신고|낀|걸친|두른)')
    filtered = []
    for term in activity_terms:
        if wear_re.search(term):
            continue
        filtered.append(term)
    parsed["object_terms"] = object_terms
    parsed["activity_terms"] = filtered
    return parsed


def _normalize_month_only_query(query: str, parsed: dict) -> dict:
    """연도 없이 '12월'처럼 말한 검색은 모든 연도의 해당 월로 보정한다."""
    q = query or ""
    has_year_hint = bool(re.search(r'(\d{4}\s*년|\d{4}[-./]\d{1,2}|올해|금년|작년|재작년|내년|지난해|이번\s*해)', q))
    month_matches = re.findall(r'(?<!\d)(1[0-2]|[1-9])\s*월(?:달)?(?!\s*\d)', q)
    if month_matches and not has_year_hint:
        months = sorted({int(m) for m in month_matches if 1 <= int(m) <= 12})
        parsed["date_range"] = None
        parsed["date_months"] = months
    else:
        parsed["date_months"] = [
            int(m) for m in (parsed.get("date_months") or [])
            if str(m).isdigit() and 1 <= int(m) <= 12
        ]
    return parsed


def _clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_clean_text(v) for v in value)
    return str(value).strip()


def _infer_activity_terms_from_events(query: str, raw_events: list) -> list:
    """현재 분석 데이터에 태그된 activity.label 기반으로 활동 검색어를 보강한다."""
    q = query or ""
    if not q:
        return []

    labels = set()
    for item in raw_events:
        activity = ((item.get("analysis") or {}).get("activity") or {})
        for key in ("label", "category"):
            val = _clean_text(activity.get(key))
            if val and val.lower() not in ("none", "unknown"):
                labels.add(val)

    aliases = {
        "일": "업무",
        "회사": "업무",
        "근무": "업무",
        "출장": "업무",
        "놀러": "여행",
        "관광": "여행",
        "여행지": "여행",
        "공부": "학술활동",
        "학습": "학술활동",
        "강의": "학술활동",
        "세미나": "학술활동",
        "산책": "산책",
        "식사": "식사",
        "문화": "문화 활동",
    }

    terms = []
    for label in sorted(labels, key=len, reverse=True):
        if label in q and label not in terms:
            terms.append(label)
    for word, label in aliases.items():
        if word in q and label not in terms:
            terms.append(label)
    return terms


def _compact_event_search_doc(ev: dict, mode: str) -> dict:
    if mode == "object":
        return {
            "id": ev["id"],
            "filename": ev.get("filename", ""),
            "specific_items": ev.get("_specific_items", []),
            "description": ev.get("_description", "")[:360],
            "activity": ev.get("_activity_text", "")[:120],
        }
    return {
        "id": ev["id"],
        "filename": ev.get("filename", ""),
        "description": ev.get("_description", "")[:360],
        "activity": ev.get("_activity_text", "")[:180],
    }


def _term_tokens(term: str) -> list:
    text = re.sub(r'[^\w가-힣]+', ' ', str(term).lower())
    tokens = []
    for token in text.split():
        token = re.sub(r'(으로|에서|에게|하고|하게|있는|사진|것들|것|을|를|이|가|은|는|에|의)$', '', token)
        if len(token) >= 2 and token not in {"사진", "있는", "보여", "모습"}:
            tokens.append(token)
    return tokens


def _lexical_filter_event_ids(terms: list, events: list, mode: str) -> set:
    matched = set()
    for ev in events:
        searchable = " ".join([
            _clean_text(ev.get("filename", "")),
            _clean_text(ev.get("_specific_items", [])),
            _clean_text(ev.get("_description", "")),
            _clean_text(ev.get("_activity_text", "")),
        ]).lower()
        for term in terms:
            term_s = str(term).strip().lower()
            if not term_s:
                continue
            if term_s in searchable:
                matched.add(ev["id"])
                break
            tokens = _term_tokens(term_s)
            if tokens and all(token in searchable for token in tokens):
                matched.add(ev["id"])
                break
    return matched


def _semantic_filter_event_ids(query: str, terms: list, events: list, mode: str) -> set | None:
    terms = [str(t).strip() for t in (terms or []) if str(t).strip()]
    if not terms:
        return None
    if not events:
        return set()

    lexical_ids = _lexical_filter_event_ids(terms, events, mode)
    if mode == "activity" and lexical_ids:
        return lexical_ids
    candidate_events = [ev for ev in events if ev["id"] in lexical_ids] if lexical_ids else events

    docs = [_compact_event_search_doc(ev, mode) for ev in candidate_events]
    docs_json = json.dumps(docs, ensure_ascii=False, separators=(",", ":"))
    term_json = json.dumps(terms, ensure_ascii=False)
    if mode == "object":
        target = "specific_items, description, filename을 함께 보고 사용자가 찾는 물건/객체/의류/소품/음식과 관련된 사진 선택"
        key_name = "object_match_ids"
    else:
        target = "description과 activity를 함께 보고 사용자가 말한 행동/활동/상황과 의미상 관련된 사진 선택"
        key_name = "activity_match_ids"

    prompt = f"""당신은 사진 검색 결과를 의미 기반으로 좁히는 도우미입니다.
사용자 질문: "{query}"
검색어: {term_json}

아래 후보 사진 목록에서 {target}하세요.
관련성이 충분한 사진 id만 반환하고, 관련 없는 사진은 제외하세요.
반드시 JSON만 반환하세요.

반환 형식:
{{"{key_name}":[0,1,2]}}

후보 사진:
{docs_json}"""

    try:
        text = _invoke_llm(prompt)
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return set()
        parsed = json.loads(match.group())
        ids = parsed.get(key_name) or parsed.get("matched_ids") or []
        llm_ids = {int(v) for v in ids if isinstance(v, (int, float, str)) and str(v).strip().isdigit()}
        return lexical_ids | llm_ids
    except Exception:
        return lexical_ids


def _strip_internal_event_fields(events: list) -> list:
    cleaned = []
    for ev in events:
        item = dict(ev)
        item.pop("_specific_items", None)
        item.pop("_description", None)
        item.pop("_activity_text", None)
        cleaned.append(item)
    return cleaned


def _search_events(parsed: dict, person_names: dict, query: str = "", include_filter_groups: bool = False):
    """파싱된 파라미터로 이벤트 순차 검색: 날짜 → 위치 → 사람 → 활동 → 객체"""
    raw = safe_load_json(os.path.join(INFO_OUTPUT_DIR, "deep_analyses.json")) or []
    date_range = parsed.get("date_range")
    date_months = [
        int(m) for m in (parsed.get("date_months") or [])
        if str(m).isdigit() and 1 <= int(m) <= 12
    ]
    location   = parsed.get("location")
    people     = parsed.get("people") or []
    object_terms = parsed.get("object_terms") or []
    activity_terms = parsed.get("activity_terms") or []
    inferred_activity_terms = _infer_activity_terms_from_events(query, raw)
    for term in inferred_activity_terms:
        if term not in activity_terms:
            activity_terms.append(term)
    parsed["activity_terms"] = activity_terms

    # people: 이름 → person_id 역매핑
    name_to_id = {v: k for k, v in person_names.items()}
    person_ids = []
    for p in people:
        if p in name_to_id:
            person_ids.append(name_to_id[p])
        else:
            person_ids.append(p)

    # 전체 이벤트 빌드
    events = []
    for i, p in enumerate(raw):
        analysis = p.get("analysis") or {}
        scene = analysis.get("scene") or {}
        food = analysis.get("food") or {}
        landscape = analysis.get("landscape") or {}
        activity = analysis.get("activity") or {}
        specific_items = food.get("specific_items") or []
        if not isinstance(specific_items, list):
            specific_items = [specific_items]
        description = " ".join(
            part for part in [
                _clean_text(scene.get("description")),
                _clean_text(landscape.get("description")),
            ] if part and part.upper() != "N/A"
        )
        activity_text = " ".join(
            part for part in [
                _clean_text(activity.get("label")),
                _clean_text(activity.get("category")),
                _clean_text(activity.get("indoor_outdoor")),
                _clean_text(activity.get("intensity")),
            ] if part and part.lower() not in ("none", "unknown")
        )
        dt_raw = p.get("datetime", "")
        ts_ms = 0
        try:
            if dt_raw:
                ts_ms = int(datetime.fromisoformat(dt_raw).timestamp() * 1000)
        except Exception:
            pass
        stem = Path(p.get("filename", "unknown")).stem
        events.append({
            "id": i,
            "filename":       p.get("filename", ""),
            "datetime":       dt_raw,
            "timestamp_ms":   ts_ms,
            "lat":            p.get("lat"),
            "lon":            p.get("lon"),
            "thumbnail_path": f"/thumbnail/{i:05d}_{stem}.jpg",
            "match_reasons":  [],
            "_specific_items": [_clean_text(item) for item in specific_items if _clean_text(item)],
            "_description":    description,
            "_activity_text":  activity_text,
        })

    filter_groups = []
    results = events[:]

    # 1단계: 날짜 필터
    if date_range or date_months:
        start_s = date_range.get("start") if date_range else ""
        end_s   = date_range.get("end") if date_range else ""
        date_matches = []
        for ev in events:
            dt = ev.get("datetime", "")[:10]
            month_ok = True
            if date_months:
                month_ok = len(dt) >= 7 and int(dt[5:7]) in date_months
            range_ok = (not date_range) or ((not start_s or dt >= start_s) and (not end_s or dt <= end_s))
            if month_ok and range_ok:
                date_matches.append({**ev, "match_reasons": ["날짜"]})
        filter_groups.append({
            "key": "date",
            "label": "날짜",
            "count": len(date_matches),
            "results": _strip_internal_event_fields(date_matches[:60]),
        })

        filtered = []
        for ev in results:
            if ev["id"] in {item["id"] for item in date_matches}:
                ev["match_reasons"].append("날짜")
                filtered.append(ev)
        results = filtered

    # 2단계: 위치 필터
    if location:
        clat = location.get("lat")
        clon = location.get("lon")
        radius = location.get("radius_km", 15)
        if clat is not None and clon is not None:
            location_matches = []
            for ev in events:
                elat, elon = ev.get("lat"), ev.get("lon")
                if elat is not None and elon is not None:
                    if _haversine_km(clat, clon, elat, elon) <= radius:
                        location_matches.append({**ev, "match_reasons": ["위치"]})
            location_ids = {item["id"] for item in location_matches}
            filtered = []
            for ev in results:
                if ev["id"] in location_ids:
                    ev["match_reasons"].append("위치")
                    filtered.append(ev)
            results = filtered

    # 3단계: 사람 필터
    people_map = None
    if person_ids:
        people_map = _load_people_mapping()
        people_matches = []
        for ev in events:
            base = os.path.basename(ev.get("filename", ""))
            photo_people = people_map.get(base, [])
            if any(pid in photo_people for pid in person_ids):
                people_matches.append({**ev, "match_reasons": ["사람"]})
        filter_groups.append({
            "key": "people",
            "label": "사람",
            "count": len(people_matches),
            "results": _strip_internal_event_fields(people_matches[:60]),
        })

        people_ids = {item["id"] for item in people_matches}
        filtered = []
        for ev in results:
            if ev["id"] in people_ids:
                ev["match_reasons"].append("사람")
                filtered.append(ev)
        results = filtered

    # 4단계: 활동/상황 필터 (현재 activity.label 태그 + description 기반 의미 매칭)
    activity_group_ids = _semantic_filter_event_ids(query, activity_terms, events, "activity")
    if activity_group_ids is not None:
        activity_matches = [{**ev, "match_reasons": ["활동"]} for ev in events if ev["id"] in activity_group_ids]
        filter_groups.append({
            "key": "activity",
            "label": "활동",
            "count": len(activity_matches),
            "results": _strip_internal_event_fields(activity_matches[:60]),
        })

    activity_ids = activity_group_ids if activity_group_ids is not None else None
    if activity_ids is not None:
        filtered = []
        for ev in results:
            if ev["id"] in activity_ids:
                ev["match_reasons"].append("활동")
                filtered.append(ev)
        results = filtered

    # 5단계: 객체/물건/음식 필터 (specific_items 중심 LLM 의미 매칭)
    object_group_ids = _semantic_filter_event_ids(query, object_terms, events, "object")
    if object_group_ids is not None:
        object_matches = [{**ev, "match_reasons": ["객체"]} for ev in events if ev["id"] in object_group_ids]
        filter_groups.append({
            "key": "object",
            "label": "객체",
            "count": len(object_matches),
            "results": _strip_internal_event_fields(object_matches[:60]),
        })

    object_ids = object_group_ids if object_group_ids is not None else None
    if object_ids is not None:
        filtered = []
        for ev in results:
            if ev["id"] in object_ids:
                ev["match_reasons"].append("객체")
                filtered.append(ev)
        results = filtered

    # match_reasons가 비어 있으면(검색 조건이 없는 경우) 모두 반환
    results = _strip_internal_event_fields(results)
    if include_filter_groups:
        return results, filter_groups
    return results
# 사진 1장에 등장한 인물 프로필 (이름·얼굴·관계) — 라이프스타일 사진뷰어용
_people_mapping_cache = {}   # user별(cluster_dir별) 캐시


def _people_mapping_cached() -> dict:
    key = str(CLUSTER_DIR)
    if key not in _people_mapping_cache:
        _people_mapping_cache[key] = _load_people_mapping()
    return _people_mapping_cache[key]


@app.route("/api/photo_people")
@login_required
def api_photo_people():
    filename = request.args.get("filename", "")
    base = os.path.basename(filename)
    mapping = _people_mapping_cached()
    pids = mapping.get(base) or mapping.get(filename) or []

    seen, uniq = set(), []
    for p in pids:
        if p not in seen:
            seen.add(p); uniq.append(p)

    names = _load_names()
    rel = _person_zero_relation_map()
    people = []
    for pid in uniq:
        imgs = get_person_images(pid, limit=1)
        info = rel.get(pid, {})
        people.append({
            "person_id": pid,
            "name": names.get(pid, pid),
            "thumbnail": f"/image/{pid}/{imgs[0]}" if imgs else None,
            "relationship_type": info.get("relationship_type", ""),
            "intimacy_score": info.get("intimacy_score", 0),
        })
    return jsonify({"filename": base, "count": len(people), "people": people})


@app.route("/api/search", methods=["POST"])
@login_required
def api_search():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400
    try:
        person_names = _load_names()
        parsed  = _parse_query(query, person_names)
        results, filter_groups = _search_events(parsed, person_names, query, include_filter_groups=True)
        return jsonify({
            "parsed":  parsed,
            "results": results[:60],
            "total":   len(results),
            "filter_groups": filter_groups,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_gps_cache: dict = {}

def _get_gps(filepath: str):
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        img = Image.open(filepath)
        exif_raw = img._getexif()
        if not exif_raw:
            return None
        for tag, val in exif_raw.items():
            if TAGS.get(tag) == "GPSInfo":
                gps = {GPSTAGS.get(k, k): v for k, v in val.items()}
                lat_dms = gps.get("GPSLatitude")
                lon_dms = gps.get("GPSLongitude")
                if not lat_dms or not lon_dms:
                    return None
                def dms(d):
                    return float(d[0]) + float(d[1]) / 60 + float(d[2]) / 3600
                lat = dms(lat_dms)
                lon = dms(lon_dms)
                if gps.get("GPSLatitudeRef") == "S":
                    lat = -lat
                if gps.get("GPSLongitudeRef") == "W":
                    lon = -lon
                return {"lat": round(lat, 6), "lon": round(lon, 6)}
    except Exception:
        pass
    return None


@app.route("/api/photo_gps/<path:filename>")
@login_required
def api_photo_gps(filename: str):
    if ".." in filename:
        abort(400)
    basename = os.path.basename(filename)
    if basename in _gps_cache:
        result = _gps_cache[basename]
        return jsonify(result) if result else ("", 204)
    for root, _, files in os.walk(ORIGINAL_DIR):
        if basename in files:
            coords = _get_gps(os.path.join(root, basename))
            _gps_cache[basename] = coords
            if coords:
                return jsonify(coords)
            return ("", 204)
    _gps_cache[basename] = None
    return ("", 204)


@app.route("/image/<person_id>/<path:filename>")
@login_required
def serve_image(person_id: str, filename: str):
    if ".." in person_id or ".." in filename:
        abort(400)
    path = os.path.join(CLUSTER_DIR, person_id, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


# ══════════════════════════════════════════════════════════════
# 실행 (web_app.py 의 HTML 대시보드 제거 → 순수 API 서버)
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 56)
    print("  lumina-backend (web_app.py 기반 모바일 API 서버)")
    print(f"  URL        : http://localhost:{PORT}")
    print(f"  데이터 폴더 : {config.DATA_DIR}")
    print(f"  인증        : {'ON' if config.REQUIRE_AUTH else 'OFF'}")
    print(f"  DATA_DIR   : {config.DATA_DIR}")
    print(f"  기본 AGENT : {config.AGENT_OUTPUT_DIR}")
    print(f"  멀티유저   : DATA_DIR/<user_id>/ 있으면 그 폴더, 없으면 기본(데모)")
    print("  (Android 에뮬레이터: http://10.0.2.2:%d)" % PORT)
    print("  서버        : Flask(Werkzeug) + threaded — 동시 업로드 처리")
    print("=" * 56)
    # threaded=True 로 요청마다 스레드를 분리해 여러 업로드를 동시에 받는다.
    # (기본 app.run() 은 단일 스레드라 동시 업로드를 직렬/블로킹 처리)
    # waitress 는 쓰지 않는다 — 요청/응답 처리 방식이 Werkzeug 와 달라
    # 앱의 실시간 업로드 진행 표시가 깨졌기 때문. threaded 만으로 동시성은 충분.
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
