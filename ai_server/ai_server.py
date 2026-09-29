"""
ai_server.py — AI 처리 관제 서버 (AI 서버 컴퓨터에서 실행)
================================================================
역할
  1. API 서버(app.py, 127.0.0.1)가 보내는 "작업 시작" / "상태확인" API를 받는다.
  2. 들어온 요청을 FIFO 큐에 넣고, 먼저 온 것부터 한 번에 하나씩 처리한다.
     (1건당 약 8시간이라 동시 처리는 하지 않는다.)
  3. 작업 처리 순서:
       (a) API 서버의  uploads/<user_id>/  이미지(약 8000장)를 SFTP로 끌어와
           로컬 작업폴더(photo_root) 바로 아래에 둔다.
       (b) integrated_server.py --photo-root <photo_root> 를 실행한다.
           stdout 의 [START]/[DONE]/[FAIL] 마커를 파싱해 현재 단계를 추적한다.
       (c) 생성된 폴더/파일 구조를 "그대로 보존"하여 API 서버의
           data/<user_id>/ 밑으로 SFTP push 한다 (app.py 가 읽는 구조와 일치).
           원본 입력 이미지는 app.py 의 /original 서빙 경로인 Camera/ 로 보낸다.
  4. 상태확인 API로 "지금 어느 단계인지"를 언제든 조회할 수 있다.
     (선택) API 서버의 /api/ai_server/progress 로도 진행률을 best-effort 로 보고한다.

integrated_server.py 가 photo_root 에 만드는 구조:
  photo_root/                        ← 입력 이미지(루트, 비재귀)
  photo_root/cropped_insight/        ← 얼굴 크롭 + 임베딩 DB
  photo_root/cropped_insight/adaface_results/  ← person_N 클러스터, image_people_mapping.csv
  photo_root/person_N/                 ← 관계분석 결과(level1~3.json, profile.json ...)
  photo_root/knowledge_graph.json 등   ← 관계분석 산출물(루트)
  photo_root/person0_deep_profile/     ← deep_analyses.json, *_stats.json, thumbnails/

이것을 API 서버 data/<user_id>/ 에 그대로 매핑한다 (app.py _resolve_user_dirs 기준):
  data/<user_id>/                      = agent  (knowledge_graph.json, person_N/levelX ...)
  data/<user_id>/cropped_insight/adaface_results/ = cluster
  data/<user_id>/person0_deep_profile/  = info (+ thumbnails)
  data/<user_id>/Camera/                = original (입력 이미지)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import queue
import re
import shutil
import stat as stat_mod
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import paramiko
import requests
# [추가됨] render_template_string 임포트 추가
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

try:
    import piexif  # 업로드 메타데이터(.meta.json)를 이미지 EXIF 로 주입할 때 사용
except Exception:  # 미설치 시 메타데이터 주입만 비활성 (작업은 계속)
    piexif = None

# Windows 콘솔(cp949)에서 한글/특수문자 print 가 UnicodeEncodeError 로 워커 스레드를
# 죽이지 않도록 표준출력을 UTF-8 로 강제한다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ════════════════════════════════════════════════════════════════
# 설정 (환경에 맞게 이 블록만 고치면 된다)
# ════════════════════════════════════════════════════════════════
SSH_HOST = os.getenv("AI_SSH_HOST", "127.0.0.1")
SSH_PORT = int(os.getenv("AI_SSH_PORT", "22"))
SSH_USER = os.environ.get("AI_SSH_USER", "")
SSH_PASS = os.environ.get("AI_SSH_PASS", "")

# API 서버(app.py)의 베이스 경로. uploads/ 와 data/ 가 이 아래에 있다.
REMOTE_BASE        = os.getenv("AI_REMOTE_BASE", "")
REMOTE_UPLOADS_DIR = os.getenv("AI_REMOTE_UPLOADS", REMOTE_BASE + "/uploads")
REMOTE_DATA_DIR    = os.getenv("AI_REMOTE_DATA",    REMOTE_BASE + "/data")

# app.py 진행률 보고 (best-effort, 실패해도 작업은 계속). 끄려면 빈 문자열.
API_SERVER_URL = os.getenv("AI_API_SERVER_URL", "http://127.0.0.1:8080")

THIS_DIR          = Path(__file__).resolve().parent
LOCAL_WORK_ROOT   = Path(os.getenv("AI_WORK_ROOT", str(THIS_DIR / "work")))
JOBS_STATE_FILE   = THIS_DIR / "jobs_state.json"
INTEGRATED_SERVER = Path(os.getenv(
    "AI_INTEGRATED_SERVER",
    str(THIS_DIR.parent / "ai_pipeline" / "integrated_server.py"),
)).resolve()
PYTHON_EXE = os.getenv("AI_PYTHON_EXE", sys.executable)

HOST = os.getenv("AI_HOST", "127.0.0.1")
PORT = int(os.getenv("AI_PORT", "8090"))

# ── 촬영시각 예측 서버 설정 (time_model_server.py 통합) ─────────────
# 이 프로세스 안에서 예측 전용 Flask 앱을 PREDICT_PORT(기본 5000)로 함께 띄운다.
# 원격 app.py 의 /api/predict_time 이 http://127.0.0.1:5000/predict 로 프록시하므로
# 포트 5000 을 유지하면 app.py 는 수정할 필요가 없다.
# THIS_DIR = image_finder/data_refine/20260610/ai_server → image_finder 까지 3단계 위
FORGITHUB_DIR = (THIS_DIR.parent / "time_prediction").resolve()
PREDICT_PORT        = int(os.getenv("AI_PREDICT_PORT", "5000"))
PREDICT_USE_CAPTION = os.getenv("AI_PREDICT_USE_CAPTION", "1") == "1"
PREDICT_MOCK        = os.getenv("AI_PREDICT_MOCK", "0") == "1"
PREDICT_YEAR        = 2026
# 가중치: forgithub/saved_models/best_layer_9.pth 우선, 없으면 forgithub/best_layer_9.pth
_predict_model_default = FORGITHUB_DIR / "saved_models" / "best_layer_9.pth"
if not _predict_model_default.exists():
    _alt = FORGITHUB_DIR / "best_layer_9.pth"
    if _alt.exists():
        _predict_model_default = _alt
PREDICT_MODEL_PATH = os.getenv("AI_PREDICT_MODEL_PATH", str(_predict_model_default))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}
META_SUFFIX = ".meta.json"            # app.py /upload 이 남기는 사이드카 (<이미지>.meta.json)
EXIF_WRITABLE_EXTS = {".jpg", ".jpeg", ".webp"}  # piexif 로 EXIF 주입 가능한 형식

# integrated_server.py 의 4단계 (stdout 마커와 동일한 이름)
PIPELINE_STEPS = ["face_crop_embedding", "make_listofpeople", "data_agent", "data_info_agent"]
STEP_LABELS = {
    "face_crop_embedding": "얼굴 크롭 + 임베딩 클러스터링",
    "make_listofpeople":   "인물-사진 매핑 CSV 생성",
    "data_agent":          "인물 관계/프로필 분석",
    "data_info_agent":     "person_0 라이프스타일/타임라인 분석",
}
# 진행률 구간: 0~5 pull, 5~90 파이프라인, 90~99 push, 100 완료
PULL_PCT, RUN_START_PCT, RUN_END_PCT, PUSH_PCT = 5, 5, 90, 90

# 파이프라인 4단계의 진행률 가중치(실제 소요시간 비중 근사).
# 균등 분배(각 21%)하면 가장 긴 face_crop_embedding 동안 바가 오래 멈춘 것처럼 보이므로,
# 실제로 오래 걸리는 단계에 더 넓은 구간을 배정한다.
STEP_WEIGHTS = {
    "face_crop_embedding": 0.55,   # 얼굴 크롭+임베딩: 수천 장 처리 → 가장 김
    "make_listofpeople":   0.05,   # CSV 매핑: 짧음
    "data_agent":          0.25,   # 관계/프로필 분석(LLM)
    "data_info_agent":     0.15,   # person_0 딥 프로필
}
# 단계 내부 시간기반 스무딩용 예상 소요시간(초) 근사 (시상수 tau 계산에만 사용).
# face_crop 은 이미지 수에 비례해 런타임 계산, 나머지는 고정 근사.
STEP_EST_SEC = {
    "face_crop_embedding": None,   # runtime: input_count 기반
    "make_listofpeople":   60.0,
    "data_agent":          1800.0,
    "data_info_agent":     900.0,
}
FACE_CROP_SEC_PER_IMAGE = 2.5      # face_crop_embedding 장당 예상 초


def _step_bounds() -> dict:
    """가중치로 각 단계의 (시작%, 끝%) 경계를 [RUN_START_PCT, RUN_END_PCT] 안에서 계산."""
    total_w = sum(STEP_WEIGHTS.get(s, 0.0) for s in PIPELINE_STEPS) or 1.0
    span = RUN_END_PCT - RUN_START_PCT
    bounds = {}
    lo = float(RUN_START_PCT)
    acc = 0.0
    for s in PIPELINE_STEPS:
        acc += STEP_WEIGHTS.get(s, 0.0) / total_w
        hi = RUN_START_PCT + span * acc
        bounds[s] = (lo, hi)
        lo = hi
    return bounds

LOCAL_WORK_ROOT.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# 작업 상태 관리 (스레드 안전 + 파일 영속화)
# ════════════════════════════════════════════════════════════════
_lock = threading.Lock()
_jobs: dict[str, dict] = {}          # job_id -> job dict
_pending_order: list[str] = []       # 큐 대기 중인 job_id 순서 (FIFO)
_work_queue: "queue.Queue[str]" = queue.Queue()
_running_procs: dict[str, "subprocess.Popen"] = {}   # job_id -> 실행 중 파이프라인 프로세스(강제종료용)
_procs_lock = threading.Lock()

TERMINAL = {"completed", "failed", "interrupted"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _persist_locked() -> None:
    """_lock 을 잡은 상태에서 호출. 전체 작업 상태를 파일에 저장."""
    try:
        with open(JOBS_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"jobs": _jobs, "pending_order": _pending_order}, f,
                      ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 상태 저장 실패: {e}", flush=True)


def _update_job(job_id: str, **changes) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return {}
        job.update(changes)
        job["updated_at"] = _now()
        # 큐 대기순번 재계산 (대기 중인 것에만 의미)
        for idx, jid in enumerate(_pending_order):
            if jid in _jobs:
                _jobs[jid]["queue_position"] = idx + 1
        _persist_locked()
        return dict(job)


def _set_progress(job_id: str, pct: int, message: str | None = None) -> None:
    pct = max(0, min(100, int(pct)))
    changes = {"progress": pct}
    if message is not None:
        changes["message"] = message
    _update_job(job_id, **changes)
    _report_progress_to_api(job_id, pct)


def _load_state_on_startup() -> None:
    if not JOBS_STATE_FILE.is_file():
        return
    try:
        with open(JOBS_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 상태 로드 실패: {e}", flush=True)
        return
    jobs = data.get("jobs", {})
    # 이전 실행에서 처리 중이던 작업은 프로세스가 죽었으므로 interrupted 처리
    for jid, job in jobs.items():
        if job.get("status") not in TERMINAL:
            job["status"] = "interrupted"
            job["message"] = "서버 재시작으로 중단됨 (재요청 필요)"
            job["queue_position"] = None
        _jobs[jid] = job
    print(f"[INIT] 이전 작업 {len(_jobs)}건 로드 (중단된 건은 interrupted 처리)", flush=True)


# ════════════════════════════════════════════════════════════════
# app.py 로 진행률 보고 (best-effort)
# ════════════════════════════════════════════════════════════════
def _report_progress_to_api(job_id: str, pct: int) -> None:
    if not API_SERVER_URL:
        return
    job = _jobs.get(job_id)
    if not job:
        return
    try:
        requests.post(
            f"{API_SERVER_URL.rstrip('/')}/api/ai_server/progress",
            json={"user_id": job["user_id"], "progress": pct},
            timeout=4,
        )
    except Exception:
        pass  # 보고 실패는 무시 (작업 진행에 영향 없음)


# ════════════════════════════════════════════════════════════════
# 업로드 메타데이터(.meta.json) → 이미지 EXIF 주입
# ════════════════════════════════════════════════════════════════
def _to_exif_datetime(captured_at) -> datetime | None:
    if not captured_at:
        return None
    s = str(captured_at).strip().replace("Z", "")
    for parse in (
        lambda x: datetime.fromisoformat(x),
        lambda x: datetime.strptime(x[:19], "%Y-%m-%dT%H:%M:%S"),
        lambda x: datetime.strptime(x[:19], "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            return parse(s)
        except Exception:
            continue
    return None


def _deg_to_dms_rational(deg_float: float) -> tuple:
    deg_abs = abs(float(deg_float))
    d = int(deg_abs)
    m_float = (deg_abs - d) * 60
    m = int(m_float)
    s = round((m_float - m) * 60, 4)
    return ((d, 1), (m, 1), (int(s * 10000), 10000))


def _apply_meta_to_image(image_path: Path, meta: dict) -> bool:
    """meta(captured_at/lat/lon)를 image 의 EXIF 로 기록. 변경 시 True."""
    if piexif is None:
        return False
    if image_path.suffix.lower() not in EXIF_WRITABLE_EXTS:
        return False
    try:
        exif_dict = piexif.load(str(image_path))
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict.setdefault("0th", {}); exif_dict.setdefault("Exif", {}); exif_dict.setdefault("GPS", {})

    changed = False
    dt = _to_exif_datetime(meta.get("captured_at"))
    if dt:
        s = dt.strftime("%Y:%m:%d %H:%M:%S")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = s
        exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = s
        exif_dict["0th"][piexif.ImageIFD.DateTime] = s
        changed = True
    lat, lon = meta.get("lat"), meta.get("lon")
    try:
        if lat is not None and lon is not None and str(lat).strip() != "" and str(lon).strip() != "":
            latf, lonf = float(lat), float(lon)
            exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N" if latf >= 0 else "S"
            exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = _deg_to_dms_rational(latf)
            exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E" if lonf >= 0 else "W"
            exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = _deg_to_dms_rational(lonf)
            changed = True
    except (TypeError, ValueError):
        pass

    if changed:
        try:
            piexif.insert(piexif.dump(exif_dict), str(image_path))
        except Exception as e:
            print(f"[WARN] EXIF 주입 실패 {image_path.name}: {e}", flush=True)
            return False
    return changed


def _apply_metadata_from_sidecars(photo_root: Path, meta_dir: Path,
                                  image_names: list[str], job_id: str) -> int:
    """<이미지>.meta.json 을 읽어 해당 이미지의 EXIF 에 주입. 반환: 적용한 이미지 수."""
    if piexif is None:
        print("[WARN] piexif 미설치 — 메타데이터 EXIF 주입 건너뜀 (pip install piexif)", flush=True)
        return 0
    applied = 0
    for name in image_names:
        sidecar = meta_dir / (name + META_SUFFIX)
        if not sidecar.is_file():
            continue
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if _apply_meta_to_image(photo_root / name, meta or {}):
            applied += 1
    _update_job(job_id, meta_applied=applied, message=f"메타데이터 EXIF 주입 {applied}장")
    return applied


# ════════════════════════════════════════════════════════════════
# SSH / SFTP 헬퍼
# ════════════════════════════════════════════════════════════════
def _ssh_connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS,
        timeout=20, look_for_keys=False, allow_agent=False,
    )
    return client


def _rpath(*parts: str) -> str:
    """원격(Windows OpenSSH SFTP) 경로를 forward-slash 로 정규화."""
    joined = "/".join(p.strip("/") for p in parts if p != "")
    return joined.replace("\\", "/")


def _sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str, _cache: set) -> None:
    """원격 디렉터리를 부모까지 재귀 생성 (이미 만든 경로는 캐시로 스킵)."""
    remote_dir = remote_dir.rstrip("/")
    if not remote_dir or remote_dir in _cache:
        return
    parts = remote_dir.split("/")
    cur = parts[0]  # 드라이브 (예: 'E:')
    for p in parts[1:]:
        cur = cur + "/" + p
        if cur in _cache:
            continue
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass  # 동시 생성 경쟁 등 무시
        _cache.add(cur)


def _sftp_pull_images(sftp: paramiko.SFTPClient, remote_user_dir: str,
                      local_dir: Path, meta_dir: Path, job_id: str) -> list[str]:
    """uploads/<user_id>/ 의 이미지를 local_dir 바로 아래로, 메타데이터 사이드카
    (<이미지>.meta.json)는 meta_dir 로 내려받는다. 반환: 내려받은 이미지 파일명 목록.
    (사이드카는 photo_root 밖(meta_dir)에 두어 파이프라인/푸시와 섞이지 않게 한다.)"""
    try:
        entries = sftp.listdir_attr(remote_user_dir)
    except IOError as e:
        raise RuntimeError(f"원격 입력 폴더 없음/접근불가: {remote_user_dir} ({e})")

    files = [e for e in entries if not stat_mod.S_ISDIR(e.st_mode)]
    images = [e for e in files if Path(e.filename).suffix.lower() in IMAGE_EXTS]
    sidecars = [e for e in files if e.filename.endswith(META_SUFFIX)]
    total = len(images)
    if total == 0:
        raise RuntimeError(f"내려받을 이미지가 없음: {remote_user_dir}")

    local_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[str] = []
    for i, e in enumerate(images, 1):
        remote_f = _rpath(remote_user_dir, e.filename)
        local_f = local_dir / e.filename
        # 이미 같은 크기로 받아져 있으면 스킵 (재시도/재실행 대비)
        if not (local_f.exists() and local_f.stat().st_size == e.st_size):
            sftp.get(remote_f, str(local_f))
        pulled.append(e.filename)
        if i % 200 == 0 or i == total:
            _update_job(job_id, pulled=i, pull_total=total)
            _set_progress(job_id, min(PULL_PCT, int(PULL_PCT * i / total)),
                          f"입력 이미지 수신 {i}/{total}")

    # 메타데이터 사이드카 수신 (있을 때만)
    if sidecars:
        meta_dir.mkdir(parents=True, exist_ok=True)
        for e in sidecars:
            try:
                sftp.get(_rpath(remote_user_dir, e.filename), str(meta_dir / e.filename))
            except IOError:
                pass
        _update_job(job_id, meta_pulled=len(sidecars))
    return pulled


def _sftp_push_outputs(sftp: paramiko.SFTPClient, local_root: Path,
                       remote_user_data: str, input_names: set[str],
                       job_id: str) -> int:
    """local_root(=photo_root) 트리를 remote_user_data 밑으로 구조 보존하며 push.
    - [개인정보 보호] 루트의 입력 이미지(input_names, 원본 사진)는 서버로 보내지
      않는다. 앱은 폰 갤러리(localUriFor)에서 원본을 직접 표시한다.
    - 그 외 모든 파일/폴더(분석 결과) → 상대경로 그대로 remote_user_data/ 밑에 만든다.
    반환: 업로드한 파일 수."""
    dir_cache: set = set()
    # 업로드 대상 파일 목록 먼저 수집 (진행률 계산용)
    files: list[tuple[Path, str]] = []  # (로컬파일, 원격대상경로)
    for dirpath, _dirnames, filenames in os.walk(local_root):
        rel_dir = os.path.relpath(dirpath, local_root)
        rel_dir = "" if rel_dir == "." else rel_dir.replace("\\", "/")
        for fn in filenames:
            local_f = Path(dirpath) / fn
            if rel_dir == "" and fn in input_names:
                continue   # 원본 사진은 push 하지 않음 (분석완료 시 서버에서 삭제 정책)
            remote_f = _rpath(remote_user_data, rel_dir, fn)
            files.append((local_f, remote_f))

    # app.py 의 완료 판정(_analysis_ready)이 보는 두 마커 파일을 "맨 마지막"에 올린다.
    # → push 도중에는 둘 중 하나라도 없으므로 app.py 가 '완료'로 오인하지 않고,
    #    두 파일이 올라간 시점엔 나머지(클러스터 이미지 등)가 이미 전부 도착해 있다.
    marker_remotes = {
        _rpath(remote_user_data, "knowledge_graph.json"),
        _rpath(remote_user_data, "person0_deep_profile", "deep_analyses.json"),
    }
    non_markers = [t for t in files if t[1] not in marker_remotes]
    markers = [t for t in files if t[1] in marker_remotes]
    files = non_markers + markers

    total = len(files)
    _sftp_mkdirs(sftp, remote_user_data, dir_cache)
    for i, (local_f, remote_f) in enumerate(files, 1):
        _sftp_mkdirs(sftp, remote_f.rsplit("/", 1)[0], dir_cache)
        sftp.put(str(local_f), remote_f)
        if i % 200 == 0 or i == total:
            _update_job(job_id, pushed=i, push_total=total)
            pct = PUSH_PCT + int((99 - PUSH_PCT) * i / total) if total else PUSH_PCT
            _set_progress(job_id, pct, f"결과 전송 {i}/{total}")
    return total


# ════════════════════════════════════════════════════════════════
# 파이프라인 실행 (stdout 마커로 단계 추적)
# ════════════════════════════════════════════════════════════════
_RE_START = re.compile(r"^\[START\]\s+(\S+)")
_RE_DONE = re.compile(r"^\[DONE\]\s+(\S+)")
_RE_FAIL = re.compile(r"^\[FAIL\]\s+(\S+)")


def _run_pipeline(photo_root: Path, job_id: str) -> None:
    """integrated_server.py 를 실행하고 stdout 을 파싱해 단계/진행률을 갱신."""
    if not INTEGRATED_SERVER.is_file():
        raise RuntimeError(f"integrated_server.py 를 찾을 수 없음: {INTEGRATED_SERVER}")

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [PYTHON_EXE, str(INTEGRATED_SERVER), "--photo-root", str(photo_root)]
    _update_job(job_id, message="파이프라인 시작", current_step=PIPELINE_STEPS[0],
                step_index=0)
    _set_progress(job_id, RUN_START_PCT, "파이프라인 시작")

    proc = subprocess.Popen(
        cmd, cwd=str(INTEGRATED_SERVER.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
    )
    with _procs_lock:
        _running_procs[job_id] = proc   # 강제종료(cancel) 시 이 프로세스를 kill

    # ── 진행률 계산 준비 ──
    bounds = _step_bounds()
    # face_crop_embedding 예상시간 = 입력 이미지 수 × 장당초 (없으면 기본 30분)
    try:
        n_img = int((_jobs.get(job_id, {}) or {}).get("input_count") or 0)
    except Exception:
        n_img = 0
    est_sec = dict(STEP_EST_SEC)
    est_sec["face_crop_embedding"] = max(120.0, n_img * FACE_CROP_SEC_PER_IMAGE) if n_img else 1800.0

    # 스무더 공유 상태 (단계 경계 사이를 경과시간 기반으로 부드럽게 전진)
    sm = {"step": PIPELINE_STEPS[0], "start_t": time.time(),
          "active": True, "last_pct": float(RUN_START_PCT)}
    sm_lock = threading.Lock()

    def _smoother():
        while True:
            time.sleep(3)
            with sm_lock:
                if not sm["active"]:
                    return
                step = sm["step"]
                lo, hi = bounds.get(step, (RUN_START_PCT, RUN_END_PCT))
                tau = max(30.0, (est_sec.get(step) or 600.0) * 0.6)
                elapsed = time.time() - sm["start_t"]
                # 단계 끝(hi)의 97% 지점까지 지수적으로 접근 → 경계를 넘지 않음
                eased = lo + (hi - lo) * 0.97 * (1.0 - math.exp(-elapsed / tau))
                pct = max(sm["last_pct"], eased)   # 단조 증가 보장
                changed = int(pct) > int(sm["last_pct"])
                sm["last_pct"] = pct
            if changed:
                _set_progress(job_id, int(pct), None)  # 메시지는 유지(단계 라벨은 START에서 세팅)

    smoother = threading.Thread(target=_smoother, name=f"prog-{job_id[:8]}", daemon=True)
    smoother.start()

    done_count = 0
    failed_step = None
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                print(f"[pipe:{job_id[:8]}] {line}", flush=True)
            m = _RE_START.match(line)
            if m:
                step = m.group(1)
                idx = PIPELINE_STEPS.index(step) if step in PIPELINE_STEPS else done_count
                lo, hi = bounds.get(step, (RUN_START_PCT, RUN_END_PCT))
                with sm_lock:
                    sm["step"] = step
                    sm["start_t"] = time.time()
                    sm["last_pct"] = max(sm["last_pct"], float(lo))
                    pct = int(sm["last_pct"])
                _update_job(job_id, current_step=step, step_index=idx)
                _set_progress(job_id, pct,
                              f"[{idx + 1}/{len(PIPELINE_STEPS)}] {STEP_LABELS.get(step, step)}")
                continue
            m = _RE_DONE.match(line)
            if m:
                step = m.group(1)
                done_count += 1
                _, hi = bounds.get(step, (RUN_START_PCT, RUN_END_PCT))
                with sm_lock:
                    sm["last_pct"] = max(sm["last_pct"], float(hi))
                    pct = int(sm["last_pct"])
                _set_progress(job_id, pct, f"{step} 완료")
                continue
            m = _RE_FAIL.match(line)
            if m:
                failed_step = m.group(1)
    finally:
        with sm_lock:
            sm["active"] = False   # 스무더 종료

    rc = proc.wait()
    with _procs_lock:
        _running_procs.pop(job_id, None)
    if rc != 0:
        raise RuntimeError(
            f"파이프라인 실패 (exit={rc}" + (f", step={failed_step}" if failed_step else "") + ")")


# ════════════════════════════════════════════════════════════════
# 작업 처리 (워커 스레드에서 실행)
# ════════════════════════════════════════════════════════════════
def _process_job(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        if job.get("status") == "cancelled":   # 시작 전 이미 강제 종료됨 → 처리하지 않음
            return
        user_id = job["user_id"]
        if job_id in _pending_order:
            _pending_order.remove(job_id)
    _update_job(job_id, status="pulling", started_at=_now(),
                queue_position=None, message="입력 이미지 수신 준비")

    photo_root = LOCAL_WORK_ROOT / _safe_dirname(user_id)
    meta_dir = LOCAL_WORK_ROOT / (_safe_dirname(user_id) + "__meta")
    ssh = None
    try:
        ssh = _ssh_connect()
        sftp = ssh.open_sftp()

        # (a) 입력 이미지 + 메타데이터 사이드카 pull
        remote_user_dir = _rpath(REMOTE_UPLOADS_DIR, user_id)
        pulled_names = _sftp_pull_images(sftp, remote_user_dir, photo_root, meta_dir, job_id)
        input_names = set(pulled_names)
        _update_job(job_id, input_count=len(input_names))
        sftp.close()

        # (a-2) app.py 가 직접 준 메타데이터(captured_at/lat/lon)를 이미지 EXIF 로 주입
        #       → integrated_server 가 EXIF 그대로 읽어 datetime/GPS 활용
        _apply_metadata_from_sidecars(photo_root, meta_dir, pulled_names, job_id)

        # (b) 파이프라인 실행 (장시간) — SSH 세션은 닫고 진행
        ssh.close(); ssh = None
        _update_job(job_id, status="running")
        _run_pipeline(photo_root, job_id)

        # (c) 결과 push
        _update_job(job_id, status="pushing", message="결과 전송 준비")
        _set_progress(job_id, PUSH_PCT, "결과 전송 준비")
        ssh = _ssh_connect()
        sftp = ssh.open_sftp()
        remote_user_data = _rpath(REMOTE_DATA_DIR, user_id)
        pushed = _sftp_push_outputs(sftp, photo_root, remote_user_data, input_names, job_id)
        sftp.close(); ssh.close(); ssh = None

        _update_job(job_id, status="completed", finished_at=_now(),
                    pushed=pushed, message="완료", current_step=None, step_index=None)
        _set_progress(job_id, 100, "완료")
        print(f"[OK] job {job_id} ({user_id}) 완료: 입력 {len(input_names)}장, 전송 {pushed}개", flush=True)

        # (d) 개인정보 보호: 분석 완료 → 로컬 작업폴더(원본 사진 사본) 일괄 삭제
        try:
            shutil.rmtree(photo_root, ignore_errors=True)
            shutil.rmtree(meta_dir, ignore_errors=True)
            print(f"[CLEAN] 로컬 작업폴더 삭제 완료: {photo_root}", flush=True)
        except Exception as e:
            print(f"[WARN] 작업폴더 삭제 실패(무시): {e}", flush=True)

    except Exception as e:
        # 사용자가 강제 종료(cancelled)한 경우엔 그 상태를 유지 (failed 로 덮지 않음)
        if _jobs.get(job_id, {}).get("status") == "cancelled":
            print(f"[CANCEL] job {job_id} ({user_id}) 강제 종료됨", flush=True)
        else:
            _update_job(job_id, status="failed", finished_at=_now(),
                        error=str(e), message=f"실패: {e}")
            print(f"[ERR] job {job_id} ({user_id}) 실패: {e}", flush=True)
    finally:
        with _procs_lock:
            _running_procs.pop(job_id, None)
        if ssh is not None:
            try:
                ssh.close()
            except Exception:
                pass


def _worker_loop() -> None:
    print("[WORKER] 시작 — 큐 대기 중", flush=True)
    while True:
        job_id = _work_queue.get()
        try:
            _process_job(job_id)
        except Exception as e:  # 방어적: 워커는 죽지 않게
            print(f"[WORKER] 예외: {e}", flush=True)
        finally:
            _work_queue.task_done()


def _safe_dirname(name: str) -> str:
    """로컬 작업폴더명으로 안전하게 (한글은 유지, 경로문자만 제거)."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip() or "unknown"


# ════════════════════════════════════════════════════════════════
# Flask API
# ════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)


def _public_job(job: dict) -> dict:
    """외부로 내보낼 작업 상태 (내부 필드 정리)."""
    return {
        "job_id":         job.get("job_id"),
        "user_id":        job.get("user_id"),
        "status":         job.get("status"),
        "progress":       job.get("progress", 0),
        "queue_position": job.get("queue_position"),
        "current_step":   job.get("current_step"),
        "step_index":     job.get("step_index"),
        "total_steps":    len(PIPELINE_STEPS),
        "step_label":     STEP_LABELS.get(job.get("current_step") or "", ""),
        "message":        job.get("message", ""),
        "input_count":    job.get("input_count"),
        "pulled":         job.get("pulled"),
        "meta_applied":   job.get("meta_applied"),
        "pushed":         job.get("pushed"),
        "error":          job.get("error"),
        "created_at":     job.get("created_at"),
        "started_at":     job.get("started_at"),
        "finished_at":    job.get("finished_at"),
        "updated_at":     job.get("updated_at"),
    }


@app.route("/")
def index():
    return jsonify({
        "service": "ai_server (관제 + 큐 + integrated_server 실행)",
        "remote_uploads": REMOTE_UPLOADS_DIR,
        "remote_data": REMOTE_DATA_DIR,
        "endpoints": [
            "POST /api/ai_server/start   {user_id}      작업 큐 등록",
            "GET  /api/ai_server/status?user_id=<id>    해당 유저 최신 작업 상태",
            "GET  /api/ai_server/jobs/<job_id>          특정 작업 상태",
            "GET  /api/ai_server/jobs                   전체 작업 목록",
            "GET  /api/ai_server/queue                  큐 현황",
            "GET  /dashboard                            [대시보드] 작업 현황 UI",
            "GET  /health",
        ],
    })


@app.route("/health")
def health():
    with _lock:
        running = [j["job_id"] for j in _jobs.values()
                   if j.get("status") in ("pulling", "running", "pushing")]
        pending = list(_pending_order)
    return jsonify({"ok": True, "running": running, "pending": pending,
                    "queue_size": len(pending)})


@app.route("/api/ai_server/start", methods=["POST"])
def api_start():
    """app.py 가 호출하는 작업 시작 API. body: {"user_id": "..."}"""
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or request.args.get("user_id") or
               request.form.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id 가 필요합니다."}), 400

    # 같은 user_id 가 이미 대기/처리 중이면 중복 등록 막고 기존 작업 반환
    with _lock:
        for jid in _pending_order:
            if _jobs[jid]["user_id"] == user_id:
                return jsonify({"ok": True, "duplicated": True,
                                "job": _public_job(_jobs[jid])}), 200
        for job in _jobs.values():
            if job["user_id"] == user_id and job["status"] in ("pulling", "running", "pushing"):
                return jsonify({"ok": True, "duplicated": True,
                                "job": _public_job(job)}), 200

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "user_id": user_id,
            "status": "queued",
            "progress": 0,
            "queue_position": len(_pending_order) + 1,
            "current_step": None,
            "step_index": None,
            "message": "큐 대기 중",
            "input_count": None,
            "pulled": None,
            "pushed": None,
            "error": None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "updated_at": _now(),
        }
        _jobs[job_id] = job
        _pending_order.append(job_id)
        _persist_locked()
        public = _public_job(job)

    _work_queue.put(job_id)
    return jsonify({"ok": True, "job": public}), 202


def _latest_job_for_user(user_id: str) -> dict | None:
    candidates = [j for j in _jobs.values() if j.get("user_id") == user_id]
    if not candidates:
        return None
    return max(candidates, key=lambda j: j.get("created_at") or "")


@app.route("/api/ai_server/status", methods=["GET"])
def api_status():
    """app.py 가 호출하는 상태확인 API. ?user_id=<id> 의 최신 작업 상태."""
    user_id = (request.args.get("user_id") or "").strip()
    if not user_id:
        return jsonify({"error": "user_id 가 필요합니다."}), 400
    with _lock:
        job = _latest_job_for_user(user_id)
        if not job:
            return jsonify({"user_id": user_id, "status": "none",
                            "message": "해당 user_id 작업 없음"}), 404
        return jsonify(_public_job(job))


@app.route("/api/ai_server/jobs/<job_id>", methods=["GET"])
def api_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error": "job 없음"}), 404
        return jsonify(_public_job(job))


@app.route("/api/ai_server/jobs", methods=["GET"])
def api_jobs():
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.get("created_at") or "", reverse=True)
        return jsonify([_public_job(j) for j in jobs])


@app.route("/api/ai_server/queue", methods=["GET"])
def api_queue():
    with _lock:
        pending = [_public_job(_jobs[jid]) for jid in _pending_order if jid in _jobs]
        running = [_public_job(j) for j in _jobs.values()
                   if j.get("status") in ("pulling", "running", "pushing")]
    return jsonify({"running": running, "pending": pending, "queue_size": len(pending)})


@app.route("/api/ai_server/cancel", methods=["POST"])
def api_cancel():
    """진행 중(또는 대기 중)인 작업을 강제 종료한다. body/query: job_id 또는 user_id."""
    data = request.get_json(silent=True) or {}
    job_id = (data.get("job_id") or request.args.get("job_id") or "").strip()
    user_id = (data.get("user_id") or request.args.get("user_id") or "").strip()

    ACTIVE = ("queued", "pulling", "running", "pushing")
    with _lock:
        if not job_id and user_id:
            for jid in sorted(_jobs, key=lambda j: _jobs[j].get("created_at") or "", reverse=True):
                if _jobs[jid].get("user_id") == user_id and _jobs[jid].get("status") in ACTIVE:
                    job_id = jid
                    break
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "취소할 작업을 찾을 수 없습니다."}), 404
        if job.get("status") not in ACTIVE:
            return jsonify({"ok": False, "error": f"이미 종료된 작업입니다({job.get('status')})."}), 409
        # 대기 중이면 큐에서 제거
        if job_id in _pending_order:
            _pending_order.remove(job_id)

    # 실행 중 파이프라인 프로세스를 트리째 종료
    with _procs_lock:
        proc = _running_procs.get(job_id)
    killed = False
    if proc is not None and proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
            killed = True
        except Exception:
            try:
                proc.terminate()
                killed = True
            except Exception:
                pass
    _update_job(job_id, status="cancelled", finished_at=_now(),
                message="사용자가 강제 종료함", current_step=None, step_index=None,
                queue_position=None)
    print(f"[CANCEL] job {job_id} 강제 종료 요청 (killed={killed})", flush=True)
    return jsonify({"ok": True, "job_id": job_id, "killed": killed})


# ════════════════════════════════════════════════════════════════
# [추가됨] 시각화 대시보드 (기존 로직 개입 없음)
# ════════════════════════════════════════════════════════════════
@app.route("/dashboard")
def dashboard():
    html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Server Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; color: #333; }
            h1, h2 { color: #2c3e50; }
            .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .job-card { border: 1px solid #e1e8ed; padding: 15px; border-radius: 6px; margin-bottom: 15px; background: #fafafa; }
            .progress-bg { background: #e0e0e0; height: 22px; border-radius: 11px; overflow: hidden; margin: 12px 0; }
            .progress-bar { background: #3498db; height: 100%; transition: width 0.5s ease-in-out; color: white; text-align: center; font-size: 13px; line-height: 22px; font-weight: bold; white-space: nowrap; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; margin-left: 10px; }
            .status-running { background: #2ecc71; color: white; }
            .status-queued { background: #f39c12; color: white; }
            .status-completed { background: #34495e; color: white; }
            .status-failed { background: #e74c3c; color: white; }
            .status-interrupted { background: #95a5a6; color: white; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            th, td { padding: 12px; border-bottom: 1px solid #eee; text-align: left; font-size: 14px; }
            th { background-color: #f8f9fa; }
        </style>
    </head>
    <body>
        <h1>🚀 AI Server 관제 대시보드</h1>

        <div class="card">
            <h2>⏳ 진행 중인 작업 (Running)</h2>
            <div id="running-container">데이터를 불러오는 중...</div>
        </div>

        <div class="card">
            <h2>📋 대기 중인 큐 (Pending Queue)</h2>
            <div id="queue-container">데이터를 불러오는 중...</div>
        </div>

        <div class="card">
            <h2>📜 전체 작업 히스토리 (History)</h2>
            <div id="history-container">데이터를 불러오는 중...</div>
        </div>

        <script>
            function getStatusBadge(status) {
                let cls = 'status-' + status;
                if (status === 'pulling' || status === 'pushing') cls = 'status-running';
                return `<span class="badge ${cls}">${status.toUpperCase()}</span>`;
            }

            async function cancelJob(jobId, userId) {
                if (!confirm(userId + ' 의 작업을 강제 종료할까요?\\n(진행 중인 분석 프로세스가 즉시 중단됩니다)')) return;
                try {
                    const r = await fetch('/api/ai_server/cancel', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ job_id: jobId }),
                    });
                    const d = await r.json();
                    alert(d.ok ? '강제 종료되었습니다.' : ('실패: ' + (d.error || '')));
                    fetchDashboardData();
                } catch (e) { alert('오류: ' + e.message); }
            }

            async function fetchDashboardData() {
                try {
                    const queueRes = await fetch('/api/ai_server/queue');
                    const queueData = await queueRes.json();

                    // 1. 진행 중 렌더링
                    let runHtml = '';
                    if (queueData.running.length > 0) {
                        runHtml = queueData.running.map(job => `
                            <div class="job-card">
                                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
                                    <div>
                                        <strong>유저:</strong> ${job.user_id} |
                                        <strong>작업 ID:</strong> <span style="color:#7f8c8d;">${job.job_id}</span>
                                        ${getStatusBadge(job.status)}
                                    </div>
                                    <button onclick="cancelJob('${job.job_id}','${job.user_id}')" style="background:#e74c3c;color:#fff;border:none;padding:6px 14px;border-radius:5px;cursor:pointer;font-weight:bold;white-space:nowrap;">■ 강제 종료</button>
                                </div>
                                <div class="progress-bg">
                                    <div class="progress-bar" style="width: ${job.progress}%">${job.progress}%</div>
                                </div>
                                <div style="font-size: 14px; color: #555;">
                                    <div><strong>현재 상태:</strong> ${job.message || '-'}</div>
                                    <div><strong>파이프라인 단계:</strong> ${job.step_label || '-'} (${job.current_step || '-'})</div>
                                </div>
                            </div>
                        `).join('');
                    } else {
                        runHtml = '<p style="color:#7f8c8d;">현재 실행 중인 작업이 없습니다.</p>';
                    }
                    document.getElementById('running-container').innerHTML = runHtml;

                    // 2. 대기 큐 렌더링
                    let pendHtml = '';
                    if (queueData.pending.length > 0) {
                        pendHtml = queueData.pending.map(job => `
                            <div class="job-card" style="padding: 10px;">
                                <strong>대기 순번 ${job.queue_position}번</strong> | 유저: ${job.user_id}
                                ${getStatusBadge(job.status)}
                                <span style="font-size: 12px; color: #999; margin-left: 10px;">(요청시간: ${job.created_at})</span>
                            </div>
                        `).join('');
                    } else {
                        pendHtml = '<p style="color:#7f8c8d;">대기 중인 작업이 없습니다.</p>';
                    }
                    document.getElementById('queue-container').innerHTML = pendHtml;

                    // 3. 히스토리 렌더링
                    const jobsRes = await fetch('/api/ai_server/jobs');
                    const jobsData = await jobsRes.json();
                    let historyHtml = `
                        <table>
                            <tr>
                                <th>유저 ID</th>
                                <th>상태</th>
                                <th>메시지</th>
                                <th>진행률</th>
                                <th>업데이트 시간</th>
                            </tr>
                    `;
                    // 최근 15개만 표시
                    jobsData.slice(0, 15).forEach(job => {
                        historyHtml += `
                            <tr>
                                <td><strong>${job.user_id}</strong></td>
                                <td>${getStatusBadge(job.status)}</td>
                                <td>${job.message || '-'}</td>
                                <td>${job.progress}%</td>
                                <td>${job.updated_at}</td>
                            </tr>
                        `;
                    });
                    historyHtml += '</table>';
                    document.getElementById('history-container').innerHTML = historyHtml;

                } catch (error) {
                    console.error('대시보드 업데이트 중 오류 발생:', error);
                }
            }

            // 2초마다 자동 갱신
            setInterval(fetchDashboardData, 2000);
            fetchDashboardData();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


# ════════════════════════════════════════════════════════════════
# 촬영시각 예측 서버 (time_model_server.py 통합, 별도 포트 5000)
# ════════════════════════════════════════════════════════════════
# inference_one_image.load_models() / predict_datetime() 를 그대로 재사용한다.
#   이미지 -> (Gemini 캡션) -> Qwen3.5-4B(layer 9) -> SequenceProbeModel -> 촬영시각
# GPU 는 파이프라인 subprocess 와 공유하므로, 예측도 직렬(_predict_lock)로 처리한다.
predict_app = Flask("ai_server_predict")
CORS(predict_app)

_predict_ready = False          # 모델 로딩 완료 여부
_predict_device_str = "cpu"
_predict_lock = threading.Lock()  # GPU 직렬 추론 보장
_one = None                       # inference_one_image 모듈 핸들


def _load_predict_models() -> None:
    """Qwen 임베딩 + Probe(+Gemini 캡션) 를 1회 로딩. 실패해도 서버는 계속."""
    global _predict_ready, _predict_device_str, _one
    if _predict_ready:
        return
    if str(FORGITHUB_DIR) not in sys.path:
        sys.path.insert(0, str(FORGITHUB_DIR))
    import inference_one_image as one  # 무거운 import 는 로딩 시점에만
    if not os.path.exists(PREDICT_MODEL_PATH):
        raise FileNotFoundError(f"예측 모델 가중치 없음: {PREDICT_MODEL_PATH}")
    one.load_models(model_path=PREDICT_MODEL_PATH, use_caption=PREDICT_USE_CAPTION)
    try:
        _predict_device_str = str(one._DEVICE)
    except Exception:
        pass
    _one = one
    _predict_ready = True
    print(f"[predict] 모델 로딩 완료 (device={_predict_device_str}, "
          f"caption={PREDICT_USE_CAPTION}, weights={PREDICT_MODEL_PATH})", flush=True)


def _preload_predict_models() -> None:
    """백그라운드에서 예측 모델을 미리 로딩 (첫 요청 지연 방지)."""
    if PREDICT_MOCK:
        print("[predict] MOCK 모드: 무거운 모델을 로딩하지 않습니다.", flush=True)
        return
    try:
        print(f"[predict] 모델 로딩 시작 (weights={PREDICT_MODEL_PATH})", flush=True)
        _load_predict_models()
    except Exception as e:
        print(f"[predict] 모델 사전 로딩 실패(요청 시 재시도): {e}", flush=True)


@predict_app.route("/health", methods=["GET"])
def predict_health():
    return jsonify({
        "ok": True,
        "mock": PREDICT_MOCK,
        "ready": _predict_ready,
        "use_caption": PREDICT_USE_CAPTION,
        "device": _predict_device_str,
        "model_path": PREDICT_MODEL_PATH,
    })


@predict_app.route("/predict", methods=["POST"])
def predict():
    import io
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "image 파일이 필요합니다."}), 400

    raw = request.files["image"].read()
    filename = request.files["image"].filename or "upload"

    try:
        if PREDICT_MOCK:
            import calendar
            import random
            month = random.randint(1, 12)
            day = random.randint(1, calendar.monthrange(PREDICT_YEAR, month)[1])
            pred_dt = datetime(PREDICT_YEAR, month, day,
                               random.randint(0, 23), random.randint(0, 59))
            actual, caption = None, "(mock caption)"
        else:
            from PIL import Image
            with _predict_lock:                # GPU 직렬 처리
                if not _predict_ready:
                    _load_predict_models()     # 사전 로딩 실패 시 요청 시점에 재시도
                pil = Image.open(io.BytesIO(raw))
                # caption=None → use_caption=True 이면 Gemini 로 자동 생성
                pred_dt, actual, caption = _one.predict_datetime(pil)
    except Exception as e:
        # 전체 트레이스백을 콘솔에 남겨 정확한 발생 위치(파일/줄)를 알 수 있게 한다.
        import traceback
        tb = traceback.format_exc()
        print(f"[predict] 예외 발생:\n{tb}", flush=True)
        # 트레이스백의 마지막 프레임(파일:줄)을 응답에도 포함해 원격에서도 위치 파악 가능
        last = tb.strip().splitlines()
        where = ""
        for ln in reversed(last):
            s = ln.strip()
            if s.startswith("File "):
                where = s
                break
        return jsonify({"ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "where": where}), 500

    return jsonify({
        "ok": True,
        "file_name": filename,
        "predicted_time": pred_dt.strftime("%m-%d %H:%M"),          # 연도 제외
        "predicted_time_full": pred_dt.strftime("%Y-%m-%d %H:%M"),  # 연도 포함(YEAR 고정)
        "actual_time": actual,                                       # EXIF 실제시각 (없으면 None)
        "caption": caption,                                          # 생성된 캡션
        "mock": PREDICT_MOCK,
    })


def _run_predict_server() -> None:
    """예측 전용 Flask 앱을 PREDICT_PORT 에서 기동 (GPU 직렬 → threaded=False)."""
    predict_app.run(host=HOST, port=PREDICT_PORT, debug=False,
                    threaded=False, use_reloader=False)


# ════════════════════════════════════════════════════════════════
# 실행
# ════════════════════════════════════════════════════════════════
def _bootstrap() -> None:
    _load_state_on_startup()
    t = threading.Thread(target=_worker_loop, name="ai-worker", daemon=True)
    t.start()
    # 예측 서버(포트 5000) + 모델 사전 로딩을 백그라운드 스레드로 기동
    threading.Thread(target=_run_predict_server, name="predict-server", daemon=True).start()
    threading.Thread(target=_preload_predict_models, name="predict-preload", daemon=True).start()


if __name__ == "__main__":
    _bootstrap()
    print("=" * 60)
    print("  ai_server — AI 처리 관제 서버 (+ 촬영시각 예측 서버 통합)")
    print(f"  관제 API      : http://{HOST}:{PORT}")
    print(f"  예측 API      : http://{HOST}:{PREDICT_PORT}/predict  (app.py 프록시 대상)")
    print(f"  예측 모델     : {PREDICT_MODEL_PATH}")
    print(f"  예측 캡션     : {'ON(Gemini)' if PREDICT_USE_CAPTION else 'OFF'}"
          f"{'  [MOCK]' if PREDICT_MOCK else ''}")
    print(f"  SSH 대상      : {SSH_USER}@{SSH_HOST}:{SSH_PORT}")
    print(f"  원격 uploads  : {REMOTE_UPLOADS_DIR}/<user_id>")
    print(f"  원격 data     : {REMOTE_DATA_DIR}/<user_id>")
    print(f"  로컬 작업폴더 : {LOCAL_WORK_ROOT}")
    print(f"  integrated    : {INTEGRATED_SERVER}")
    print(f"  python        : {PYTHON_EXE}")
    print("=" * 60)
    # 워커 스레드를 쓰므로 reloader 비활성(스레드 중복 방지)
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
