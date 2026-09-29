"""
config.py — 백엔드 설정 (web_app.py 의 하드코딩 경로를 여기로 분리)
============================================================
동료가 보내준 분석 파일들의 위치를 .env 또는 환경변수로 지정.
기본값: 이 폴더 안의 data/ 하위에 web_app.py 와 같은 구조로 둔다고 가정.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

BASE_DIR = Path(__file__).parent

# 동료가 준 분석 결과 최상위 폴더
DATA_DIR = Path(os.getenv("LUMINA_DATA_DIR", str(BASE_DIR / "data")))

# web_app.py 가 읽던 4개 위치 (동료 파일 구조에 맞춰 .env 로 조정 가능)
#   - AGENT_OUTPUT_DIR : 관계분석(knowledge_graph.json, person_X/level1~3.json ...)
#   - INFO_OUTPUT_DIR  : 라이프스타일(deep_analyses.json, *_stats.json, thumbnails/)
#   - ORIGINAL_DIR     : 원본 사진 폴더
#   - CLUSTER_DIR      : 얼굴 클러스터(person_X/, image_people_mapping.csv)
AGENT_OUTPUT_DIR = os.getenv("LUMINA_AGENT_DIR",   str(DATA_DIR / "default"))
INFO_OUTPUT_DIR  = os.getenv("LUMINA_INFO_DIR",    str(DATA_DIR / "person0_deep_profile"))
THUMBNAILS_DIR   = os.getenv("LUMINA_THUMBS_DIR",  str(Path(INFO_OUTPUT_DIR) / "thumbnails"))
ORIGINAL_DIR     = os.getenv("LUMINA_ORIGINAL_DIR", str(Path(AGENT_OUTPUT_DIR) / "Camera"))
CLUSTER_DIR      = os.getenv("LUMINA_CLUSTER_DIR",
                             str(Path(ORIGINAL_DIR) / "cropped_insight" / "insight_results"))

HOST = os.getenv("LUMINA_HOST", "127.0.0.1")
PORT = int(os.getenv("LUMINA_PORT", "8080"))

# AI 처리 서버 (integrated_server 를 돌리는 별도 컴퓨터) 주소.
# app.py(API 서버)가 작업 시작/상태확인 요청을 이 주소로 보낸다.
AI_SERVER_URL     = os.getenv("LUMINA_AI_SERVER_URL", "http://127.0.0.1:8090")
AI_SERVER_TIMEOUT = float(os.getenv("LUMINA_AI_SERVER_TIMEOUT", "10"))

# 촬영시각 예측 GPU 모델서버 (time_model_server.py). app.py 의 /api/predict_time 이 중계.
TIME_MODEL_SERVER_URL = os.getenv("LUMINA_TIME_MODEL_URL", "http://127.0.0.1:5000")

# 인증: 모바일은 기본 OFF. 켜려면 LUMINA_REQUIRE_AUTH=1
REQUIRE_AUTH = os.getenv("LUMINA_REQUIRE_AUTH", "0") == "1"
PASSWORD     = os.getenv("LUMINA_PASSWORD", "")

# 검색 LLM: GOOGLE_APPLICATION_CREDENTIALS로 지정한 외부 서비스 계정 사용 (Vertex AI)
