import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import sys
import io
import json
import base64
import re
import time
import math
from datetime import datetime

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image  # type: ignore[import-untyped]
from PIL.ExifTags import TAGS, GPSTAGS  # type: ignore[import-untyped]

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage

# ── 설정 ──────────────────────────────────────────────────────


# InsightFace 클러스터링 결과 (person_N 폴더들)
CLUSTER_DIR = os.path.join(os.getenv("LUMINA_PHOTO_ROOT", "runtime/photos"), "cropped_insight", "adaface_results")
# 에이전트 분석 결과 저장 경로
OUTPUT_DIR  = os.getenv("LUMINA_PHOTO_ROOT", "runtime/photos")
# 크롭 전 원본 사진 폴더 (EXIF 기반 컨텍스트 확장에 사용)
ORIGINAL_PHOTOS_DIR = os.getenv("LUMINA_PHOTO_ROOT", "runtime/photos")

MIN_PHOTOS_FOR_REASONING = 3
TIME_WINDOW_MINUTES  = 180    # person_0 근접 사진 탐색 시간 범위 (분)
DISTANCE_KM_THRESHOLD = 1.0  # GPS 거리 임계값 (km)

# 0단계: 미술작품/포스터 필터(실제 인물 검증).
# 인물당 최대 10장씩 LLM 비전 호출이라 후보가 많으면 시간을 크게 잡아먹는다.
# False면 0단계를 건너뛰고 모든 클러스터를 실제 인물로 간주한다.
ENABLE_ARTWORK_FILTER = False

# ── person_0 나이 설정 ────────────────────────────────────────
PERSON_ZERO_AGE      = int(os.environ["LUMINA_PERSON_ZERO_AGE"]) if os.getenv("LUMINA_PERSON_ZERO_AGE") else None
PERSON_ZERO_AGE_YEAR = 2026  # 나이 기준 연도

# ── Vertex AI 초기화 ───────────────────────────────────────────
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
            location="global",
            credentials=credentials,
        )
        _llm = ChatVertexAI(
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=0.1,
            # gemini-2.5-flash는 사고형 모델이라 '사고 토큰'이 출력 토큰 한도를 잠식한다.
            # 입력이 무거우면(dual-image 등) 사고에 토큰을 다 써버려 긴 JSON이 중간에서
            # 잘리고(MAX_TOKENS) 파싱 실패 → 축소 양식으로 떨어진다.
            # ① thinking_budget으로 사고 토큰을 묶어 출력용 토큰을 보호하고
            # ② max_output_tokens를 넉넉히 올려 긴 프로필 JSON도 안 잘리게 한다.
            thinking_budget=2048,
            max_output_tokens=16384,
            credentials=credentials,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location="global",
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 공통 유틸리티
# ══════════════════════════════════════════════════════════════

def encode_image_base64(image_path: str) -> str:
    """이미지를 base64 JPEG로 인코딩한다.
    MPO(듀얼카메라/모션포토)·PNG 등을 RGB JPEG로 정규화해,
    Vertex가 거부하는 MIME(image/mpo 등)로 인한 KeyError를 방지한다."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_metadata_from_filename(filename: str) -> dict:
    patterns = [
        (r'(\d{4})[_\-](\d{2})[_\-](\d{2})[_\sT](\d{2})[_:](\d{2})', 5),
        (r'(\d{4})(\d{2})(\d{2})[_\-](\d{2})(\d{2})', 5),
        (r'(\d{4})[_\-](\d{2})[_\-](\d{2})', 3),
        (r'(\d{8})', 1),
    ]
    for pattern, group_count in patterns:
        m = re.search(pattern, filename)
        if not m:
            continue
        g = m.groups()
        try:
            if group_count == 1:
                raw = g[0]
                year, month, day = int(raw[:4]), int(raw[4:6]), int(raw[6:8])
                hour = None
            elif group_count == 3:
                year, month, day, hour = int(g[0]), int(g[1]), int(g[2]), None
            else:
                year, month, day, hour = int(g[0]), int(g[1]), int(g[2]), int(g[3])

            # 비현실적 연도 거부 (파일 시퀀스 번호 등 오파싱 방지)
            if not (1950 <= year <= 2030):
                continue
            # 월/일 기본 범위 체크
            if not (1 <= month <= 12) or not (1 <= day <= 31):
                continue

            def time_of_day(h):
                if h is None: return "unknown"
                if 5 <= h < 12: return "morning"
                if 12 <= h < 17: return "afternoon"
                if 17 <= h < 21: return "evening"
                return "night"

            return {
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "hour": hour,
                "time_of_day": time_of_day(hour),
            }
        except (ValueError, IndexError):
            pass
    return {"date": "unknown", "hour": None, "time_of_day": "unknown"}


def extract_json_from_response(text: str, expect_list: bool = False):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    pattern = r'\[.*\]' if expect_list else r'\{.*\}'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return [] if expect_list else {}


def invoke_with_retry(model, messages: list, max_retries: int = 3, base_delay: float = 2.0) -> str:
    for attempt in range(max_retries):
        try:
            response = model.invoke(messages)
            return response.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise e


# ── EXIF 유틸리티 ─────────────────────────────────────────────

def _dms_to_dd(dms_val) -> float:
    """도/분/초(DMS) → 십진도(DD) 변환. IFDRational, float, tuple of tuples 모두 처리."""
    def to_float(v):
        if isinstance(v, tuple) and len(v) == 2:
            return v[0] / v[1] if v[1] != 0 else 0.0
        return float(v)
    d, m, s = dms_val
    return to_float(d) + to_float(m) / 60.0 + to_float(s) / 3600.0


def get_photo_exif(image_path: str) -> dict:
    """JPEG EXIF에서 촬영 일시 및 GPS 위경도 추출."""
    result = {"datetime": None, "lat": None, "lon": None}
    try:
        img = Image.open(image_path)
        try:
            raw = img._getexif() or {}
        except AttributeError:
            raw = dict(img.getexif()) if hasattr(img, "getexif") else {}
        if not raw:
            return result

        exif = {TAGS.get(k, k): v for k, v in raw.items()}

        dt_raw = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if dt_raw:
            result["datetime"] = datetime.strptime(str(dt_raw), "%Y:%m:%d %H:%M:%S")

        gps_raw = exif.get("GPSInfo")
        if isinstance(gps_raw, dict):
            gps = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
            if "GPSLatitude" in gps and "GPSLatitudeRef" in gps:
                lat = _dms_to_dd(gps["GPSLatitude"])
                result["lat"] = -lat if gps["GPSLatitudeRef"] == "S" else lat
            if "GPSLongitude" in gps and "GPSLongitudeRef" in gps:
                lon = _dms_to_dd(gps["GPSLongitude"])
                result["lon"] = -lon if gps["GPSLongitudeRef"] == "W" else lon
    except Exception:
        pass
    return result


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def get_original_filepath(cropped_filename: str, original_dir: str):
    """크롭 파일명 → 원본 파일 경로 추론.
    예) IMG_20230423_142532_2.jpg → original_dir/IMG_20230423_142532.jpg
    """
    stem = Path(cropped_filename).stem
    m = re.match(r'^(.+)_\d+$', stem)
    original_stem = m.group(1) if m else stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".HEIC", ".heic"):
        p = os.path.join(original_dir, original_stem + ext)
        if os.path.isfile(p):
            return p
    return None


def get_sample_images(person_dir: str, max_count: int = 4) -> list:
    """클러스터 폴더에서 샘플 이미지 경로 목록 반환."""
    return sorted([
        os.path.join(person_dir, f)
        for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ])[:max_count]


def filter_usable_observations(observations: list) -> list:
    """흐릿하거나 옆모습인 사진을 제외하고 인식에 활용 가능한 관찰만 반환.
    필터 후 남은 사진이 없으면 원본 목록을 반환한다."""
    usable = [
        obs for obs in observations
        if obs.get("analysis", {}).get("face_quality", {}).get("usable_for_recognition") is not False
    ]
    return usable if usable else observations


_YEAR_MIN, _YEAR_MAX = 2008, 2030  # 유효 연도 범위


def _estimated_age_to_number(age_str: str) -> int | None:
    """person_0 의 추정 나이 문자열을 대표 정수 나이로 변환.
    '26세'/'26' → 26, '30대 초반' → 31, '40대 중반' → 43 처럼 범위는 중앙값을 쓴다.
    파싱 실패 시 None.
    """
    if not age_str or age_str == "알수없음":
        return None
    m = re.search(r'(\d{1,2})\s*세', age_str)          # '26세'
    if m:
        return int(m.group(1))
    lo, hi = _estimate_age_range(age_str)               # 'NN대 초/중/후반'
    if lo is not None and hi is not None:
        return (lo + hi) // 2
    m = re.search(r'\b(\d{1,2})\b', age_str)            # 단독 숫자
    if m:
        return int(m.group(1))
    return None


def build_age_timeline(observations: list, person0_age: int | None = None,
                       ref_year: int = PERSON_ZERO_AGE_YEAR) -> dict:
    """사진의 촬영 연도별 person_0의 나이를 계산.
    person0_age 가 주어지면 그 '추정 나이'(ref_year 기준)로 출생 연도를 역산하고,
    없으면 하드코딩 상수 PERSON_ZERO_AGE 로 폴백한다.
    반환: {"2015": 15, "2020": 20, ...}
    유효하지 않은 연도는 제외한다.
    """
    age_at_ref = person0_age if person0_age is not None else PERSON_ZERO_AGE
    if age_at_ref is None:
        return {}
    birth_year = ref_year - age_at_ref  # 출생 연도

    years: set = set()
    for obs in observations:
        date_str = obs.get("metadata", {}).get("date", "unknown")
        if date_str and date_str != "unknown":
            try:
                yr = int(date_str[:4])
                if _YEAR_MIN <= yr <= _YEAR_MAX:
                    years.add(yr)
            except ValueError:
                pass
    return {
        str(yr): yr - birth_year
        for yr in sorted(years)
    }


def _get_subject_year_context(observations: list, age_timeline: dict) -> tuple:
    """대상 인물의 연도별 등장 횟수(나의 나이 포함), 시간대 분포, distinct 연도 수를 반환.
    Returns (year_distribution_str, time_distribution_str, distinct_years: int)
    """
    year_counter: Counter = Counter()
    time_counter: Counter = Counter()
    for obs in observations:
        meta = obs.get("metadata", {})
        date_str = meta.get("date", "unknown")
        tod = meta.get("time_of_day", "unknown")
        if date_str and date_str != "unknown":
            try:
                yr = int(date_str[:4])
                if _YEAR_MIN <= yr <= _YEAR_MAX:
                    year_counter[str(yr)] += 1
            except ValueError:
                pass
        if tod and tod != "unknown":
            time_counter[tod] += 1

    year_lines = []
    for yr in sorted(year_counter):
        cnt = year_counter[yr]
        age = age_timeline.get(yr)
        age_str = f" (당시 나 {age}세)" if age else ""
        year_lines.append(f"  {yr}년: {cnt}장{age_str}")
    year_str = "\n".join(year_lines) if year_lines else "  연도 정보 없음"

    time_lines = [f"  {tod}: {cnt}장" for tod, cnt in time_counter.most_common()]
    time_str = "\n".join(time_lines) if time_lines else "  시간대 정보 없음"
    distinct_years = len(year_counter)
    return year_str, time_str, distinct_years


# ══════════════════════════════════════════════════════════════
# 0단계: 실제 인물 검증 에이전트 (ArtworkFilterAgent)
# ══════════════════════════════════════════════════════════════

ARTWORK_CHECK_PROMPT = """\
이 이미지에 나온 얼굴 주인공이 실제 살아있는 사람인지 판단하세요.
그림 속 인물, 조각상, 포스터, 삽화, TV/모니터 화면, 책 표지 등은 실제 인물이 아닙니다.
JSON 형식으로만 응답하세요.

{
  "is_real_person": true/false,
  "subject_type": "real_person/painting/statue/poster/screen/illustration/book_cover/unknown",
  "confidence": "high/medium/low",
  "reason": "판단 근거 한 문장"
}
"""


class ArtworkFilterAgent:
    """클러스터가 실제 인물인지 미술작품·포스터 등인지 사전 필터링."""

    def __init__(self, model):
        self.model = model

    def _check_single(self, image_path: str) -> dict:
        try:
            img_b64 = encode_image_base64(image_path)
            msg = HumanMessage(content=[
                {"type": "text", "text": ARTWORK_CHECK_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ])
            text = invoke_with_retry(self.model, [msg])
            return extract_json_from_response(text)
        except Exception:
            return {"is_real_person": True, "confidence": "low"}

    def is_real_person_cluster(self, person_dir: str, sample_size: int = 10):
        """Returns (is_real: bool, reason: str)"""
        files = get_sample_images(person_dir, sample_size)

        if not files:
            return False, "이미지 없음"

        results = [self._check_single(f) for f in files]
        real_cnt = sum(1 for r in results if r.get("is_real_person", True))
        ratio = real_cnt / len(results)

        if ratio < 0.5:
            types = [r.get("subject_type", "") for r in results if not r.get("is_real_person", True)]
            return False, f"비실제인물 ({types[0] if types else '미술작품 등'}, 실제인물 {int(ratio*100)}%)"
        return True, f"실제 인물 ({int(ratio*100)}%)"


# ══════════════════════════════════════════════════════════════
# Level 1: 관찰 에이전트 (ObserverAgent)
# ══════════════════════════════════════════════════════════════

OBSERVER_PROMPT = """\
당신은 사진 분석 전문가입니다. 이 이미지를 분석하고 아래 JSON 형식으로만 응답하세요.
다른 설명 없이 JSON만 출력하세요.

{
  "visual": {
    "estimated_age": "10대/20대/30대/40대/50대/60대이상/알수없음",
    "gender": "남성/여성/불명확",
    "glasses_type": "none/regular/sunglasses/unknown",
    "beard": true/false,
    "hair_style": "short/medium/long/bald/unknown",
    "hair_color": "black/brown/gray/white/dyed/unknown",
    "clothing_style": "casual/formal/sporty/outdoor/unknown",
    "expression": "smiling/neutral/serious/laughing/unknown"
  },
  "environment": {
    "location_type": "indoor/outdoor/unknown",
    "location_detail": "예: 식당, 공원, 사무실, 카페, 산 등",
    "objects": ["이미지에서 눈에 띄는 사물 목록"],
    "background_description": "배경 간략 설명"
  },
  "context": {
    "formality": "formal/casual/semi-formal/unknown",
    "activity": "예: 식사, 여행, 업무, 운동, 파티 등",
    "event_type": "예: 일상, 여행, 행사, 업무, 모임 등"
  },
  "companions": {
    "alone": true/false,
    "group_size": "solo/pair/small_group/large_group"
  },
  "face_quality": {
    "is_blurry": true/false,
    "is_side_profile": true/false,
    "usable_for_recognition": true/false
  }
}

face_quality 판단 기준:
- is_blurry: 얼굴이 흐릿하거나 초점이 맞지 않으면 true
- is_side_profile: 얼굴이 45도 이상 옆을 보고 있어 정면 인식이 어려우면 true
- usable_for_recognition: 얼굴이 선명하고 정면에 가까워 인물 식별에 활용 가능하면 true (blurry 또는 side_profile이면 false)
"""


# 크롭 얼굴 + 원본 전체 사진을 함께 보낼 때 앞에 붙이는 안내문.
# 인물 고유 속성은 크롭(이미지1) 기준, 장면 정보는 전체 사진(이미지2) 기준으로 판단하게 한다.
DUAL_IMAGE_HEADER = """\
두 장의 이미지가 함께 제공됩니다.
- [이미지 1] = 분석 대상 인물의 얼굴 클로즈업(크롭). 분석 대상은 바로 이 인물 한 명입니다.
- [이미지 2] = 같은 인물이 찍힌 원본 전체 사진(장면 전체).

판단 규칙(반드시 지키세요):
- visual(estimated_age, gender, glasses_type, beard, hair_style, hair_color, clothing_style, expression)과
  face_quality 는 [이미지 1] 얼굴을 기준으로 판단합니다.
- environment(location_type, location_detail, objects, background_description),
  context(formality, activity, event_type), companions(alone, group_size) 는
  [이미지 2] 전체 사진을 기준으로 판단합니다.
- [이미지 2]에 여러 사람이 있어도 분석 대상은 [이미지 1]의 인물 한 명뿐입니다.
- 절대 [이미지 1]의 좁은 배경만 보고 장소·사물을 추측하지 말고, 반드시 [이미지 2]를 사용하세요.

"""


class ObserverAgent:
    """Level 1: 개별 이미지 분석 → 비주얼/환경/맥락 태그 생성"""

    def __init__(self, model):
        self.model = model

    def analyze_single_image(self, image_path: str, original_path: str | None = None) -> dict:
        """크롭 얼굴(image_path)과, 가능하면 원본 전체 사진(original_path)을 함께 분석한다.
        장소·사물·동행 등 장면 정보는 전체 사진 기준으로 판단하도록 두 이미지를 모두 전송한다."""
        filename = os.path.basename(image_path)
        metadata = extract_metadata_from_filename(filename)
        use_dual = bool(original_path) and os.path.isfile(original_path)
        try:
            crop_b64 = encode_image_base64(image_path)
            date_note = f"참고: 이 사진은 {metadata['date']} {metadata['time_of_day']}에 촬영되었습니다."
            if use_dual:
                full_b64 = encode_image_base64(original_path)
                content = [
                    {"type": "text", "text": f"{DUAL_IMAGE_HEADER}{OBSERVER_PROMPT}\n{date_note}"},
                    {"type": "text", "text": "[이미지 1] 분석 대상 인물의 얼굴 클로즈업:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                    {"type": "text", "text": "[이미지 2] 같은 인물이 찍힌 원본 전체 사진:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
                ]
            else:
                # 원본을 못 찾으면 크롭 단독 분석으로 폴백
                content = [
                    {"type": "text", "text": f"{OBSERVER_PROMPT}\n{date_note}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                ]
            message = HumanMessage(content=content)
            text = invoke_with_retry(self.model, [message])
            analysis = extract_json_from_response(text)
            return {
                "image_path": image_path,
                "original_path": original_path if use_dual else None,
                "filename": filename,
                "metadata": metadata,
                "analysis": analysis,
                "scene_source": "full_photo" if use_dual else "crop_only",
                "analyzed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"  [Observer] 오류 - {filename}: {e}")
            return {
                "image_path": image_path,
                "original_path": original_path if use_dual else None,
                "filename": filename,
                "metadata": metadata,
                "analysis": {},
                "scene_source": "full_photo" if use_dual else "crop_only",
                "error": str(e),
            }

    def analyze_person(self, person_dir: str, max_workers: int = 4) -> list:
        image_files = [
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        person_id = os.path.basename(person_dir)

        # 각 크롭 얼굴을 원본 전체 사진에 매핑 (장면 분석용)
        tasks = []
        found = 0
        for f in image_files:
            orig = get_original_filepath(os.path.basename(f), ORIGINAL_PHOTOS_DIR)
            if orig:
                found += 1
            tasks.append((f, orig))
        print(f"  [Observer] {person_id}: {len(image_files)}장 분석 중... "
              f"(원본 전체사진 매칭 {found}/{len(image_files)})")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.analyze_single_image, f, orig): f for (f, orig) in tasks}
            for future in as_completed(futures):
                results.append(future.result())
        return results


# ══════════════════════════════════════════════════════════════
# Level 2: 추론 에이전트 (ReasonerAgent)
# ══════════════════════════════════════════════════════════════

REASONER_PROMPT_TEMPLATE = """\
당신은 데이터 분석 전문가입니다.
아래는 '{person_id}' 인물의 여러 사진에서 수집된 관찰 데이터입니다.
이 데이터를 시계열로 분석하여 라이프스타일 패턴을 도출해주세요.
JSON 형식으로만 응답하세요.

관찰 데이터:
{observations_json}

{{
  "estimated_age": "여러 사진을 종합한 나이 추정 (예: 30대 초반, 60대 후반 등)",
  "gender": "남성/여성/불명확",
  "lifestyle": {{
    "preferred_location_types": ["주로 등장하는 장소 유형 목록"],
    "activity_keywords": ["주요 활동 키워드"],
    "clothing_tendency": "casual/formal/sporty/mixed",
    "expression_tendency": "표정 경향 설명"
  }},
  "temporal_patterns": {{
    "most_active_time": "morning/afternoon/evening/night/unknown",
    "dates_recorded": ["파악된 날짜 목록"],
    "activity_frequency": "매우 자주(10+장)/자주(5-9장)/가끔(2-4장)/드물게(1장)"
  }},
  "social_patterns": {{
    "tends_to_be_alone": true/false,
    "group_preference": "solo/pair/small_group/large_group",
    "event_types_appeared": ["등장한 이벤트 유형"]
  }},
  "appearance_changes": {{
    "consistent_look": true/false,
    "notable_changes": "변화 설명 또는 null"
  }},
  "summary": "이 인물의 라이프스타일 패턴을 2-3문장으로 요약. 반드시 채워야 합니다."
}}
"""


class ReasonerAgent:
    """Level 2: 관찰 데이터를 시계열로 묶어 라이프스타일 패턴 도출"""

    def __init__(self, model):
        self.model = model

    def analyze_patterns(self, person_id: str, observations: list) -> dict:
        if len(observations) < MIN_PHOTOS_FOR_REASONING:
            summary = self._generate_minimal_summary(observations)
            age, gender = self._aggregate_age_gender(observations)
            return {
                "status": "minimal_data",
                "photo_count": len(observations),
                "estimated_age": age,
                "gender": gender,
                "summary": summary,
                "message": f"사진 {len(observations)}장으로 간략 분석",
            }

        usable = filter_usable_observations(observations)
        filtered_count = len(observations) - len(usable)
        if filtered_count > 0:
            print(f"  [Reasoner] {person_id}: 흐릿/옆모습 {filtered_count}장 제외 ({len(usable)}장 사용)")

        obs_summary = [
            {
                "filename": o.get("filename"),
                "date": o.get("metadata", {}).get("date"),
                "time_of_day": o.get("metadata", {}).get("time_of_day"),
                "analysis": o.get("analysis", {}),
            }
            for o in usable
        ]

        prompt = REASONER_PROMPT_TEMPLATE.format(
            person_id=person_id,
            observations_json=json.dumps(obs_summary, ensure_ascii=False, indent=2),
        )

        try:
            text = invoke_with_retry(self.model, [HumanMessage(content=prompt)])
            result = extract_json_from_response(text)
            result["photo_count"] = len(observations)
            result["status"] = "analyzed"
            if not result.get("summary"):
                result["summary"] = self._generate_minimal_summary(observations)
            # estimated_age/gender 누락 시 집계 값으로 폴백
            if not result.get("estimated_age"):
                result["estimated_age"], _ = self._aggregate_age_gender(observations)
            if not result.get("gender"):
                _, result["gender"] = self._aggregate_age_gender(observations)
            return result
        except Exception as e:
            print(f"  [Reasoner] 오류 - {person_id}: {e}")
            age, gender = self._aggregate_age_gender(observations)
            return {
                "status": "error",
                "error": str(e),
                "estimated_age": age,
                "gender": gender,
                "summary": self._generate_minimal_summary(observations),
            }

    @staticmethod
    def _aggregate_age_gender(observations: list):
        """L1 개별 추정값을 다수결로 집계."""
        ages, genders = [], []
        for o in observations:
            v = o.get("analysis", {}).get("visual", {})
            if v.get("estimated_age") and v["estimated_age"] != "알수없음":
                ages.append(v["estimated_age"])
            if v.get("gender") and v["gender"] != "불명확":
                genders.append(v["gender"])
        age = Counter(ages).most_common(1)[0][0] if ages else "알수없음"
        gender = Counter(genders).most_common(1)[0][0] if genders else "불명확"
        return age, gender

    def _generate_minimal_summary(self, observations: list) -> str:
        """사진이 적을 때 간단한 텍스트 요약 생성 (LLM 없이)"""
        if not observations:
            return "데이터 없음"
        analyses = [o.get("analysis", {}) for o in observations if o.get("analysis")]
        if not analyses:
            return f"사진 {len(observations)}장이 수집되었으나 분석 데이터가 부족합니다."
        locs = [a.get("environment", {}).get("location_detail", "") for a in analyses if a.get("environment")]
        acts = [a.get("context", {}).get("activity", "") for a in analyses if a.get("context")]
        loc_str = "·".join(list(set(l for l in locs if l and l != "unknown"))[:3])
        act_str = "·".join(list(set(a for a in acts if a and a != "unknown"))[:3])
        parts = []
        if loc_str:
            parts.append(f"주로 {loc_str}에서 발견됩니다")
        if act_str:
            parts.append(f"{act_str} 활동이 관찰됩니다")
        return "  ".join(parts) if parts else f"사진 {len(observations)}장 수집됨."


# ══════════════════════════════════════════════════════════════
# person_0 심층 프로파일링 (PersonZeroProfiler)
# ══════════════════════════════════════════════════════════════

_AGG_SKIP = {"unknown", "", None, "알수없음", "불명확"}


def _estimate_age_range(age_str: str) -> tuple:
    """나이 추정 문자열에서 (최소, 최대) 정수 범위를 추론. 실패 시 (None, None)."""
    m = re.search(r'(\d0)\s*대', age_str or "")
    if not m:
        return None, None
    base = int(m.group(1))
    if "초반" in age_str:
        return base, base + 3
    if "중반" in age_str:
        return base + 3, base + 6
    if "후반" in age_str:
        return base + 6, base + 9
    return base, base + 9


def _top_values(items: list, key: str, n: int = 5) -> list:
    """관찰 dict 목록에서 특정 key의 최빈값 상위 n개 반환 (의미없는 값 제외)."""
    c = Counter(v.get(key) for v in items if isinstance(v, dict) and v.get(key) not in _AGG_SKIP)
    return [k for k, _ in c.most_common(n)]


def build_profile_skeleton(observations: list, patterns: dict) -> dict:
    """LLM 심층 프로필 JSON 파싱이 실패해도 old 양식(appearance/lifestyle/
    photo_contexts/age_range_*)의 전체 키를 채울 수 있도록, Level 1 관찰
    데이터를 집계해 기본 골격을 만든다. LLM 결과는 이 골격 위에 덮어쓴다."""
    visuals = [a["visual"] for o in observations
               if isinstance((a := o.get("analysis")), dict) and a.get("visual")]
    envs = [a["environment"] for o in observations
            if isinstance((a := o.get("analysis")), dict) and a.get("environment")]
    ctxs = [a["context"] for o in observations
            if isinstance((a := o.get("analysis")), dict) and a.get("context")]
    comps = [a["companions"] for o in observations
             if isinstance((a := o.get("analysis")), dict) and a.get("companions")]

    hair = _top_values(visuals, "hair_style", 1)
    glasses_vals = [v.get("glasses_type") for v in visuals if v.get("glasses_type") not in _AGG_SKIP]
    g_cnt = Counter(glasses_vals)
    worn = g_cnt.get("regular", 0) + g_cnt.get("sunglasses", 0)
    if not glasses_vals:
        glasses_tendency = "정보 없음"
    elif worn >= len(glasses_vals) * 0.6:
        glasses_tendency = "대부분 착용"
    elif worn > 0:
        glasses_tendency = "가끔 착용"
    else:
        glasses_tendency = "착용 안함"

    activities = list(dict.fromkeys(_top_values(ctxs, "activity", 5) + _top_values(ctxs, "event_type", 5)))[:5]
    locations = _top_values(envs, "location_detail", 5) or _top_values(envs, "location_type", 3)
    alone_cnt = sum(1 for c in comps if c.get("alone") is True)
    if not comps:
        social_style = "알수없음"
    elif alone_cnt >= len(comps) * 0.6:
        social_style = "내향적"
    else:
        social_style = "외향적"

    age_str = patterns.get("estimated_age", "알수없음")
    amin, amax = _estimate_age_range(age_str)

    return {
        "estimated_age": age_str,
        "age_range_min": amin,
        "age_range_max": amax,
        "gender": patterns.get("gender", "불명확"),
        "appearance": {
            "hair_style": hair[0] if hair else "정보 없음",
            "glasses_tendency": glasses_tendency,
            "build": "정보 없음",
            "notable_features": "정보 없음",
        },
        "lifestyle": {
            "activity_types": activities or ["정보 없음"],
            "typical_locations": locations or ["정보 없음"],
            "social_style": social_style,
        },
        "photo_contexts": locations or ["정보 없음"],
        "profile_summary": patterns.get("summary", ""),
    }


def _merge_over_skeleton(skeleton: dict, llm_result: dict) -> dict:
    """골격 위에 LLM 결과를 덮어쓴다. LLM이 비운(빈 문자열/리스트/dict/None) 값은
    무시해 골격의 집계값을 보존한다."""
    merged = dict(skeleton)
    for k, v in (llm_result or {}).items():
        if v in (None, "", [], {}):
            continue
        merged[k] = v
    return merged


PERSON_ZERO_DEEP_PROMPT = """\
당신은 인물 분석 전문가입니다.
아래는 '나'(person_0, 이 사진 컬렉션의 소유자)의 여러 사진에서 수집된 관찰 데이터입니다.
이 데이터를 종합해 '나'의 기본 프로필을 추출해주세요.
이 프로필은 이후 다른 모든 인물과의 관계 분석에서 기준점(anchor)으로 사용됩니다.
JSON 형식으로만 응답하세요.

관찰 데이터:
{observations_json}

{{
  "estimated_age": "종합 나이 추정 (예: 30대 초반, 40대 중반). 최대한 구체적으로.",
  "age_range_min": 추정 최소 나이(정수),
  "age_range_max": 추정 최대 나이(정수),
  "gender": "남성/여성/불명확",
  "appearance": {{
    "hair_style": "헤어스타일 설명",
    "glasses_tendency": "대부분 착용/가끔 착용/선글라스 자주/착용 안함",
    "build": "체형 설명",
    "notable_features": "외모 특징 (피부, 체형 등)"
  }},
  "lifestyle": {{
    "activity_types": ["주요 활동 키워드"],
    "typical_locations": ["자주 가는 장소 유형"],
    "social_style": "활동적/가정적/외향적/내향적/알수없음"
  }},
  "photo_contexts": ["주로 어떤 상황·장소에서 찍혔는지 목록"],
  "profile_summary": "나의 전반적 프로필 2-3문장. 나이, 성별, 라이프스타일 포함. 반드시 작성."
}}
"""


class PersonZeroProfiler:
    """person_0(나)를 먼저 심층 분석하여 관계 분석의 기준점 프로필 생성."""

    def __init__(self, model):
        self.model = model

    def build_profile(self, observations: list, patterns: dict) -> dict:
        usable = filter_usable_observations(observations)
        obs_summary = [
            {
                "filename": o.get("filename"),
                "date": o.get("metadata", {}).get("date"),
                "analysis": o.get("analysis", {}),
            }
            for o in usable[:20]  # 최대 20장
        ]

        prompt = PERSON_ZERO_DEEP_PROMPT.format(
            observations_json=json.dumps(obs_summary, ensure_ascii=False, indent=2)
        )

        # 관찰 데이터 집계 골격: LLM JSON 파싱이 실패(사고형 모델 MAX_TOKENS 절단 등)해도
        # old 양식의 전체 키(appearance/lifestyle/photo_contexts/age_range_*)를 항상 보장한다.
        skeleton = build_profile_skeleton(observations, patterns)
        llm_profile: dict = {}
        try:
            # 빈 dict(파싱 실패)면 1회 재시도해 LLM의 상세 서술을 최대한 확보
            for _ in range(2):
                text = invoke_with_retry(self.model, [HumanMessage(content=prompt)])
                llm_profile = extract_json_from_response(text)
                if llm_profile:
                    break
        except Exception as e:
            print(f"  [PersonZeroProfiler] 오류: {e}")

        profile = _merge_over_skeleton(skeleton, llm_profile)
        if not profile.get("profile_summary"):
            profile["profile_summary"] = patterns.get("summary", "")
        profile["photo_count"] = len(observations)
        profile["status"] = "analyzed"
        return profile


# ══════════════════════════════════════════════════════════════
# Level 3: 관계 에이전트 (SocialAnalystAgent)
# ══════════════════════════════════════════════════════════════

SOCIAL_ANALYST_PROMPT_TEMPLATE = """\
당신은 인간 관계 분석 전문가입니다.
아래는 사진 컬렉션 소유자인 '나'(person_0)의 프로필과,
'나'의 사진에서 발견된 인물({person_id})에 대한 종합 데이터입니다.

═══ 나(person_0)의 기본 프로필 ═══
{person_zero_profile_json}

나의 현재 추정 나이: {p0_age}  |  성별: {p0_gender}  |  나의 전체 사진 수: {p0_photo_count}장

나의 연도별 나이 타임라인 (역산):
{p0_age_timeline_str}

═══ 분석 대상 인물 ({person_id}) ═══
추정 나이: {subject_age}  |  성별: {subject_gender}
등장 사진 수: {subject_photo_count}장 (나의 전체 사진 중 {appearance_ratio}%)
등장 연도 수: {subject_distinct_years}개 연도
나이 차이 힌트 (현재 기준): {age_gap_hint}

이 인물의 연도별 등장 분포 (당시 나의 나이 포함):
{subject_year_distribution}

이 인물의 촬영 시간대 분포:
{subject_time_distribution}

--- Level 1 관찰 데이터 (대표 샘플) ---
{observation_sample_json}

--- Level 2 패턴 분석 ---
{pattern_json}

나의 연도별 나이, 이 인물의 나이 차이, 등장 맥락, 등장 빈도, 촬영 시간대를 모두 종합해 관계를 분석하세요.

【등장 빈도 및 연도 기반 판단 기준 — 매우 중요】
- 등장 사진 수가 1~4장: 지인, 가끔 만나는 직장 동료, 우연히 찍힌 친척 등 가능성이 높습니다.
  이 경우 배우자, 부모, 자녀처럼 매일 보는 관계로 판단하지 마세요.
- 등장 사진 수가 5~9장: 가까운 친구, 자주 만나는 동료, 가까운 친척 가능성이 있습니다.
- 등장 사진 수가 10장 이상이고 다양한 상황(일상·여행·행사 등)에서 등장하면:
  배우자, 부모, 자녀 등 가장 가까운 관계 가능성이 높습니다.
- 배우자는 보통 한 명이며 사진 컬렉션 전반에 걸쳐 매우 자주 등장합니다.
  나이대와 성별만 맞는다고 배우자로 판단하지 마세요. 등장 빈도가 뒷받침되어야 합니다.

【가족 판단을 위한 다년도 등장 조건 — 반드시 확인】
- 부모(어머니/아버지), 배우자, 자녀를 가족으로 판단하려면 반드시 2개 이상의 서로 다른 연도에 걸쳐 꾸준히 등장해야 합니다.
- 등장 연도가 1개뿐이거나 연도 정보가 없으면 가족 판단을 보류하고 '지인' 또는 '알수없음'으로 분류하세요.
- 여행 한 번에 집중적으로 찍힌 사진들은 그 여행에서 만난 사람일 수 있습니다. 여행지 특화 장소가 반복되면 가족이 아닐 가능성을 높게 보세요.

【촬영 시간대 및 장소 기반 판단】
- morning/afternoon 비중이 높고 사무실·회의실 등 업무 공간에서 자주 등장: 직장 동료 가능성 높음
- evening/night 또는 여행지·식당·야외에서 자주 등장: 친구/가족 가능성
- 여행지에서만 등장하고 일상 공간에서 등장이 없음: 여행에서 만난 지인 가능성 (가족이 아닐 수 있음)
- 특정 연도에만 집중적으로 등장하고 이후 사라짐: 당시 직장동료, 학교 친구 등 시기한정 관계 가능성

【나이 차이 기반 추론 가이드 (사진 촬영 당시 나의 나이 기준)】
- 촬영 당시 나보다 20~35살 많으면 → 부모/친척 어른 가능성
- 촬영 당시 나보다 10~20살 많으면 → 상사/선배/친척 형 누나 가능성
- 비슷한 나이대면 → 친구/동료 가능성, 빈도가 높을 때만 배우자 고려
- 촬영 당시 나보다 10~30살 어리면 → 자녀/후배/조카 가능성
- 자주 야외/여행에서 함께 등장하면 → 가족/절친 가능성
- 주로 업무 공간에서 등장하면 → 직장 관계 가능성

중요: 모든 필드를 반드시 채우세요.
- relationship_narrative 는 최소 2문장 이상 서술해야 합니다.
- 데이터가 부족하면 관찰 가능한 단서와 나이 차이를 근거로 최선의 추론을 하세요.
- 알수없음으로 채우는 것보다 근거 있는 추론이 더 좋습니다.

JSON 형식으로만 응답하세요:

{{
  "relationship_type": "가족/학창시절친구/직장동료/지인/알수없음",
  "relationship_detail": "예: 부모, 고등학교 동창, 현직장 동료, 오랜 친구",
  "intimacy_score": 1~10 사이 숫자,
  "intimacy_basis": "점수 산정 근거를 2문장 이상으로 서술",
  "age_based_reasoning": "나이 차이를 바탕으로 한 관계 추론 근거",
  "meeting_context": {{
    "primary_setting": "주로 어디서 만나는지",
    "meeting_style": "단독/그룹/혼합",
    "occasion_types": ["만남의 유형"]
  }},
  "episode_keywords": ["관계를 상징하는 키워드 3개 이상"],
  "relationship_narrative": "이 사람과의 관계를 2-3문장으로 서사적으로 설명. 나이 차이와 등장 맥락 포함. 반드시 작성.",
  "knowledge_graph_edge": {{
    "from": "person_0",
    "to": "{person_id}",
    "relation": "관계 유형",
    "weight": 친밀도_점수,
    "tags": ["태그1", "태그2", "태그3"]
  }}
}}
"""


def _estimate_age_gap_hint(p0_age: str, subject_age: str) -> str:
    """두 나이 추정값에서 차이 힌트 텍스트 생성."""
    AGE_MAP = {
        "10대": 15, "20대": 25, "20대 초반": 22, "20대 중반": 25, "20대 후반": 27,
        "30대": 35, "30대 초반": 32, "30대 중반": 35, "30대 후반": 38,
        "40대": 45, "40대 초반": 42, "40대 중반": 45, "40대 후반": 48,
        "50대": 55, "50대 초반": 52, "50대 중반": 55, "50대 후반": 58,
        "60대": 65, "60대 초반": 62, "60대 중반": 65, "60대이상": 65,
        "알수없음": None,
    }
    p0_val = AGE_MAP.get(p0_age)
    sub_val = AGE_MAP.get(subject_age)

    if p0_val is None or sub_val is None:
        return "나이 차이 불명확"

    diff = sub_val - p0_val
    if diff > 25:
        return f"이 인물이 나보다 약 {diff}살 많음 → 부모/어른 가능성"
    elif diff > 10:
        return f"이 인물이 나보다 약 {diff}살 많음 → 선배/상사/형누나 가능성"
    elif diff > -10:
        return f"나이 비슷 (차이 약 {abs(diff)}살) → 친구/동료/배우자 가능성"
    elif diff > -25:
        return f"이 인물이 나보다 약 {abs(diff)}살 어림 → 후배/동생 가능성"
    else:
        return f"이 인물이 나보다 약 {abs(diff)}살 어림 → 자녀/조카 가능성"


class SocialAnalystAgent:
    """Level 3: 인물과 '나' 사이의 관계 심층 분석 + 지식 그래프 엣지 생성"""

    def __init__(self, model):
        self.model = model

    def analyze_relationship(
        self,
        person_id: str,
        observations: list,
        patterns: dict,
        person_zero_profile: dict | None = None,
        age_timeline: dict | None = None,
    ) -> dict:
        obs_sample = [
            {
                "filename": o.get("filename"),
                "date": o.get("metadata", {}).get("date"),
                "analysis": o.get("analysis", {}),
            }
            for o in observations[:10]
        ]

        p0_profile = person_zero_profile or {}
        p0_age = p0_profile.get("estimated_age", "알수없음")
        p0_gender = p0_profile.get("gender", "불명확")
        p0_photo_count = p0_profile.get("photo_count", 0)
        subject_age = patterns.get("estimated_age", "알수없음")
        subject_gender = patterns.get("gender", "불명확")
        subject_photo_count = patterns.get("photo_count", len(observations))
        appearance_ratio = (
            round(subject_photo_count / p0_photo_count * 100, 1)
            if p0_photo_count > 0 else "알수없음"
        )
        age_gap_hint = _estimate_age_gap_hint(p0_age, subject_age)

        tl = age_timeline or {}
        if tl:
            tl_lines = [f"  {yr}년: {age}세" for yr, age in sorted(tl.items())]
            p0_age_timeline_str = "\n".join(tl_lines)
        else:
            p0_age_timeline_str = "  연도별 나이 정보 없음"

        year_dist_str, time_dist_str, distinct_years = _get_subject_year_context(observations, tl)

        prompt = SOCIAL_ANALYST_PROMPT_TEMPLATE.format(
            person_id=person_id,
            person_zero_profile_json=json.dumps(p0_profile, ensure_ascii=False, indent=2),
            p0_age=p0_age,
            p0_gender=p0_gender,
            p0_photo_count=p0_photo_count,
            p0_age_timeline_str=p0_age_timeline_str,
            subject_age=subject_age,
            subject_gender=subject_gender,
            subject_photo_count=subject_photo_count,
            subject_distinct_years=distinct_years,
            appearance_ratio=appearance_ratio,
            age_gap_hint=age_gap_hint,
            subject_year_distribution=year_dist_str,
            subject_time_distribution=time_dist_str,
            observation_sample_json=json.dumps(obs_sample, ensure_ascii=False, indent=2),
            pattern_json=json.dumps(patterns, ensure_ascii=False, indent=2),
        )

        try:
            text = invoke_with_retry(self.model, [HumanMessage(content=prompt)])
            result = extract_json_from_response(text)
            result["person_id"] = person_id
            result["status"] = "analyzed"
            result["_distinct_years"] = distinct_years  # GlobalValidator용 메타
            return result
        except Exception as e:
            print(f"  [SocialAnalyst] 오류 - {person_id}: {e}")
            return {"status": "error", "person_id": person_id, "error": str(e),
                    "_distinct_years": distinct_years}


# ══════════════════════════════════════════════════════════════
# person_0 컨텍스트 확장 (PersonContextExpander)
# ══════════════════════════════════════════════════════════════

CONTEXT_SUMMARY_PROMPT = """\
다음은 'person_0(나)'가 있는 사진과 같은 시간대·장소에서 찍힌 주변 사진들의 분석 결과입니다.
이 데이터를 바탕으로 person_0의 당시 활동 컨텍스트를 추론하세요.
JSON만 응답하세요.

{analyses_json}

{{
  "inferred_activities": ["person_0의 당시 추정 활동"],
  "inferred_location": "추정 장소 설명",
  "social_context": "함께한 사람들의 상황 추론",
  "summary": "person_0의 당시 상황 2-3문장 요약. 반드시 작성."
}}
"""


class PersonContextExpander:
    """person_0(나)가 등장한 사진의 EXIF를 기반으로 같은 시간/위치 근처 사진들로 컨텍스트 확장."""

    def __init__(self, model):
        self.model = model

    def expand(self, person0_dir: str, original_photos_dir: str) -> dict:
        if not os.path.isdir(original_photos_dir):
            return {"status": "original_dir_not_found", "original_dir": original_photos_dir}

        # 1. person_0 크롭 이미지 → 원본 EXIF 수집
        p0_files = [
            f for f in os.listdir(person0_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        p0_exif_list = []
        for fname in p0_files:
            orig = get_original_filepath(fname, original_photos_dir)
            if orig:
                exif = get_photo_exif(orig)
                if exif["datetime"]:
                    p0_exif_list.append(exif)

        if not p0_exif_list:
            return {
                "status": "no_exif_data",
                "p0_photos_checked": len(p0_files),
                "hint": "원본 사진의 EXIF DateTimeOriginal이 없거나 원본 파일을 찾지 못했습니다.",
            }

        # 2. 원본 디렉토리 전체 스캔
        valid_exts = {".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG", ".HEIC"}
        all_originals = [
            os.path.join(original_photos_dir, f)
            for f in os.listdir(original_photos_dir)
            if Path(f).suffix in valid_exts
        ]

        # 3. 시간/위치 근접 사진 탐색
        nearby: dict = {}
        for orig_path in all_originals:
            exif = get_photo_exif(orig_path)
            if not exif["datetime"]:
                continue
            for ref in p0_exif_list:
                dt_diff_min = abs((exif["datetime"] - ref["datetime"]).total_seconds()) / 60.0
                if dt_diff_min > TIME_WINDOW_MINUTES:
                    continue
                if ref["lat"] and exif["lat"]:
                    dist = haversine_km(ref["lat"], ref["lon"], exif["lat"], exif["lon"])
                    if dist > DISTANCE_KM_THRESHOLD:
                        continue
                nearby[orig_path] = {
                    "path": orig_path,
                    "filename": os.path.basename(orig_path),
                    "datetime": exif["datetime"].isoformat(),
                    "minutes_diff": round(dt_diff_min, 1),
                }
                break

        sorted_nearby = sorted(nearby.values(), key=lambda x: x["minutes_diff"])
        print(f"  [ContextExpander] person_0 기준 근접 사진 {len(sorted_nearby)}장 발견")

        if not sorted_nearby:
            return {
                "status": "no_nearby_photos",
                "p0_references": len(p0_exif_list),
                "total_scanned": len(all_originals),
            }

        # 4. 근접 사진 시각 분석 (최대 10장)
        context = self._analyze_nearby(sorted_nearby[:10])
        context.update({
            "status": "success",
            "p0_references": len(p0_exif_list),
            "nearby_count": len(sorted_nearby),
            "nearby_photos": [
                {k: v for k, v in p.items() if k != "path"}
                for p in sorted_nearby[:20]
            ],
        })
        return context

    def _analyze_nearby(self, nearby_photos: list) -> dict:
        analyses = []
        for p in nearby_photos[:10]:
            try:
                img_b64 = encode_image_base64(p["path"])
                msg = HumanMessage(content=[
                    {"type": "text", "text": OBSERVER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ])
                text = invoke_with_retry(self.model, [msg])
                analyses.append({
                    "filename": p["filename"],
                    "minutes_diff": p["minutes_diff"],
                    "analysis": extract_json_from_response(text),
                })
            except Exception:
                pass

        default = self._default_context(analyses)
        if not analyses:
            return default

        prompt = CONTEXT_SUMMARY_PROMPT.format(
            analyses_json=json.dumps(analyses, ensure_ascii=False, indent=2)
        )
        llm_ctx: dict = {}
        try:
            # 빈 dict(파싱 실패)면 1회 재시도. 실패해도 default 골격으로 old 양식 키 보장.
            for _ in range(2):
                text = invoke_with_retry(self.model, [HumanMessage(content=prompt)])
                llm_ctx = extract_json_from_response(text)
                if llm_ctx:
                    break
        except Exception as e:
            print(f"  [ContextExpander] 요약 생성 오류: {e}")
        return _merge_over_skeleton(default, llm_ctx)

    @staticmethod
    def _default_context(analyses: list) -> dict:
        """근접 사진 분석들을 집계해 컨텍스트 요약 골격을 만든다.
        LLM 요약 JSON 파싱이 실패해도 old 양식 키(inferred_activities/
        inferred_location/social_context/summary)를 항상 보장한다."""
        acts, locs = [], []
        for a in analyses:
            an = a.get("analysis", {}) if isinstance(a, dict) else {}
            c = an.get("context", {}) or {}
            e = an.get("environment", {}) or {}
            if c.get("activity") not in _AGG_SKIP:
                acts.append(c.get("activity"))
            if e.get("location_detail") not in _AGG_SKIP:
                locs.append(e.get("location_detail"))
        acts = list(dict.fromkeys(acts))[:5]
        locs = list(dict.fromkeys(locs))[:5]
        if locs or acts:
            parts = []
            if locs:
                parts.append(f"주로 {'·'.join(locs[:3])}")
            if acts:
                parts.append(f"{'·'.join(acts[:3])} 활동")
            summary = " ".join(parts) + " 정황이 관찰됩니다"
        else:
            summary = "근접 사진 기반 컨텍스트 정보가 부족합니다"
        return {
            "inferred_activities": acts or ["정보 없음"],
            "inferred_location": "·".join(locs) if locs else "정보 없음",
            "social_context": "정보 없음",
            "summary": summary,
        }


# ══════════════════════════════════════════════════════════════
# 전체 관계 재검증 에이전트 (GlobalRelationshipValidator)
# ══════════════════════════════════════════════════════════════

GLOBAL_VALIDATOR_PROMPT = """\
당신은 사진 컬렉션 관계도 검증 전문가입니다.
아래는 person_0(나)와 각 인물들에 대한 1차 관계 분석 결과입니다.

person_0 프로필:
{person_zero_profile_json}

전체 인물 관계 분석 결과:
{all_relations_json}

【검증 규칙 — 반드시 준수】
1. 어머니는 최대 1명: 여러 명이 '어머니'로 분류된 경우, 등장 연도 수(distinct_years)가 가장 많고 사진 수가 많은 1명만 어머니로 유지. 나머지는 '친척', '지인' 등으로 재분류.
2. 아버지는 최대 1명: 같은 기준.
3. 배우자는 최대 1명: 등장 연도 수·사진 수가 가장 많고 일상·여행·행사 등 다양한 상황에서 등장한 1명만 유지. 나머지는 '친구', '지인' 등으로 재분류.
4. 가족(부모·배우자·자녀)으로 판단하려면 distinct_years가 2 이상이어야 합니다. 1개 연도에만 집중 등장한 인물은 가족에서 제외하세요.
5. 여행지에서만 등장하고 사진 수가 5장 미만인 인물을 부모·배우자·자녀로 판단하지 말 것.
6. 사진 수 1~4장인 인물을 부모·배우자로 판단하지 말 것.
7. 이미 올바르게 분류된 인물은 그대로 유지(keep_original: true).
8. 재분류 시 intimacy_score도 함께 조정하세요 (지인 수준이면 1~4, 친구/동료면 5~7, 가족이면 7~10).

각 인물에 대해 재검토 결과를 JSON 배열로 응답하세요:
[
  {{
    "person_id": "person_N",
    "keep_original": true/false,
    "revised_relationship_type": "가족/학창시절친구/직장동료/지인/알수없음",
    "revised_relationship_detail": "재분류 세부 설명 (예: 어머니, 친척, 여행지 지인, 직장 동료 등)",
    "revised_intimacy_score": 1~10 사이 정수,
    "revision_reason": "재분류 이유 한 문장"
  }}
]
"""


class GlobalRelationshipValidator:
    """1차 분석된 모든 관계를 전체 맥락에서 재검토.
    유일성 제약(어머니 1명, 배우자 1명 등) 위반과 빈도 미달 가족 판정을 교정."""

    def __init__(self, model):
        self.model = model

    def validate(self, all_level3: dict, person_zero_profile: dict) -> dict:
        """
        all_level3: {person_id: level3_dict}
        Returns: {person_id: revised_level3_dict}  (변경 없는 인물은 원본 그대로)
        """
        summary = []
        for pid, rel in all_level3.items():
            summary.append({
                "person_id":           pid,
                "relationship_type":   rel.get("relationship_type", "알수없음"),
                "relationship_detail": rel.get("relationship_detail", ""),
                "intimacy_score":      rel.get("intimacy_score", 5),
                "photo_count":         rel.get("_photo_count", 0),
                "distinct_years":      rel.get("_distinct_years", 0),
                "primary_setting":     rel.get("meeting_context", {}).get("primary_setting", ""),
                "occasion_types":      rel.get("meeting_context", {}).get("occasion_types", []),
            })

        prompt = GLOBAL_VALIDATOR_PROMPT.format(
            person_zero_profile_json=json.dumps(person_zero_profile, ensure_ascii=False, indent=2),
            all_relations_json=json.dumps(summary, ensure_ascii=False, indent=2),
        )

        try:
            text = invoke_with_retry(self.model, [HumanMessage(content=prompt)])
            revisions = extract_json_from_response(text, expect_list=True)
        except Exception as e:
            print(f"  [GlobalValidator] 오류: {e}")
            return all_level3

        revised = dict(all_level3)
        for item in revisions:
            pid = item.get("person_id")
            if pid not in revised:
                continue
            if item.get("keep_original"):
                continue
            original = dict(revised[pid])
            original["relationship_type"]   = item.get("revised_relationship_type", original.get("relationship_type"))
            original["relationship_detail"] = item.get("revised_relationship_detail", original.get("relationship_detail"))
            original["intimacy_score"]      = item.get("revised_intimacy_score", original.get("intimacy_score"))
            original["_revision_reason"]    = item.get("revision_reason", "")
            original["_globally_revised"]   = True
            # knowledge_graph_edge도 동기화
            if "knowledge_graph_edge" in original:
                original["knowledge_graph_edge"]["relation"] = original["relationship_type"]
                original["knowledge_graph_edge"]["weight"]   = original["intimacy_score"]
            revised[pid] = original
            print(f"  [GlobalValidator] {pid}: {item.get('revision_reason','')}")
        return revised


# ══════════════════════════════════════════════════════════════
# 오케스트레이터 (PersonProfileOrchestrator)
# ══════════════════════════════════════════════════════════════

class PersonProfileOrchestrator:
    """0~3단계 에이전트 + 전체 관계 재검증 + person_0 컨텍스트 확장을 총괄."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        llm = _get_llm()
        self.artwork_filter  = ArtworkFilterAgent(llm)
        self.observer        = ObserverAgent(llm)
        self.reasoner        = ReasonerAgent(llm)
        self.analyst         = SocialAnalystAgent(llm)
        self.p0_profiler     = PersonZeroProfiler(llm)
        self.global_validator = GlobalRelationshipValidator(llm)
        self.ctx_expander    = PersonContextExpander(llm)
        self.knowledge_graph = {
            "nodes": [{"id": "person_0", "label": "나 (사진 컬렉션 소유자)"}],
            "edges": [],
        }
        os.makedirs(output_dir, exist_ok=True)

    # ── person_0 우선 분석 ────────────────────────────────────────

    def _analyze_person_zero(self, person0_dir: str) -> dict:
        """person_0를 먼저 심층 분석하여 관계 분석 기준점 생성.
        Returns dict with keys: profile, observations, patterns, age_timeline
        """
        print(f"\n{'═'*55}")
        print(f"[person_0 우선 분석] 나의 프로필 구축 중...")
        person_out = os.path.join(self.output_dir, "person_0")
        os.makedirs(person_out, exist_ok=True)

        observations = self.observer.analyze_person(person0_dir)
        obs_path = os.path.join(person_out, "level1_observations.json")
        with open(obs_path, "w", encoding="utf-8") as f:
            json.dump(observations, f, ensure_ascii=False, indent=2)

        patterns = self.reasoner.analyze_patterns("person_0", observations)
        pat_path = os.path.join(person_out, "level2_patterns.json")
        with open(pat_path, "w", encoding="utf-8") as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)

        profile = self.p0_profiler.build_profile(observations, patterns)

        # person_0 추정 나이 기반 연도별 나이 타임라인 계산 후 프로필에 포함
        # (프로필에서 추정한 나이를 우선 사용, 실패 시 PERSON_ZERO_AGE 상수로 폴백)
        p0_age_num = _estimated_age_to_number(profile.get("estimated_age"))
        age_timeline = build_age_timeline(observations, person0_age=p0_age_num)
        profile["age_timeline"] = age_timeline
        if p0_age_num is not None:
            profile["estimated_age_years"] = p0_age_num  # 타임라인 역산 기준 나이

        profile_path = os.path.join(person_out, "profile.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        age = profile.get("estimated_age", "알수없음")
        gender = profile.get("gender", "불명확")
        summary = profile.get("profile_summary", "")
        tl_preview = ", ".join(f"{yr}:{v}세" for yr, v in list(age_timeline.items())[-3:]) if age_timeline else "없음"
        print(f"  ✔ 나의 프로필: 나이 {age} / {gender}")
        print(f"  ✔ 연도별 나이 (최근): {tl_preview}")
        print(f"  ✔ 요약: {summary[:80]}")

        # knowledge_graph person_0 노드 업데이트
        self.knowledge_graph["nodes"][0].update({
            "photo_count": len(observations),
            "estimated_age": age,
            "gender": gender,
            "lifestyle_summary": summary,
        })

        return {
            "profile": profile,
            "observations": observations,
            "patterns": patterns,
            "age_timeline": age_timeline,
        }

    # ── 단일 인물 처리 ───────────────────────────────────────────

    def process_person(
        self,
        person_dir: str,
        person_zero_profile: dict | None = None,
        age_timeline: dict | None = None,
    ) -> dict:
        person_id = os.path.basename(person_dir)
        person_out = os.path.join(self.output_dir, person_id)
        os.makedirs(person_out, exist_ok=True)

        print(f"\n{'─'*55}")
        print(f"[Orchestrator] ▶ {person_id} 처리 시작")

        # Level 1
        observations = self.observer.analyze_person(person_dir)
        obs_path = os.path.join(person_out, "level1_observations.json")
        with open(obs_path, "w", encoding="utf-8") as f:
            json.dump(observations, f, ensure_ascii=False, indent=2)
        print(f"  ✔ Level 1 완료: {len(observations)}장 분석")

        # Level 2
        patterns = self.reasoner.analyze_patterns(person_id, observations)
        pat_path = os.path.join(person_out, "level2_patterns.json")
        with open(pat_path, "w", encoding="utf-8") as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
        age_info = f"나이 {patterns.get('estimated_age','?')} / {patterns.get('gender','?')}"
        print(f"  ✔ Level 2 완료: {age_info} | {patterns.get('summary', '')[:50]}")

        # Level 3 (person_0 프로필 + 연도별 나이 타임라인 활용)
        relations = self.analyst.analyze_relationship(
            person_id, observations, patterns, person_zero_profile, age_timeline
        )
        relations["_photo_count"] = len(observations)  # 전체 재검증용 메타

        if not relations.get("relationship_narrative"):
            relations["relationship_narrative"] = self._fallback_narrative(
                person_id, patterns, relations
            )

        rel_path = os.path.join(person_out, "level3_relations.json")
        with open(rel_path, "w", encoding="utf-8") as f:
            json.dump(relations, f, ensure_ascii=False, indent=2)
        rel_type = relations.get("relationship_type", "알수없음")
        intimacy = relations.get("intimacy_score", "?")
        print(f"  ✔ Level 3 완료: {rel_type} / 친밀도 {intimacy}")

        self.knowledge_graph["nodes"].append({
            "id": person_id,
            "photo_count": len(observations),
            "estimated_age": patterns.get("estimated_age", ""),
            "gender": patterns.get("gender", ""),
            "lifestyle_summary": patterns.get("summary", ""),
            "relationship_type": rel_type,
            "intimacy_score": intimacy,
            "relationship_narrative": relations.get("relationship_narrative", ""),
            "age_based_reasoning": relations.get("age_based_reasoning", ""),
        })
        edge = relations.get("knowledge_graph_edge")
        if edge:
            self.knowledge_graph["edges"].append(edge)

        return {
            "person_id": person_id,
            "photo_count": len(observations),
            "relationship_type": rel_type,
            "intimacy_score": intimacy,
            "observations": observations,
            "relations": relations,
        }

    def _reuse_person(self, person_dir: str) -> dict:
        """이미 level1~3 이 만들어진 인물을 LLM 재호출 없이 기존 결과로 복원 (resume/이어하기)."""
        person_id = os.path.basename(person_dir)
        person_out = os.path.join(self.output_dir, person_id)
        with open(os.path.join(person_out, "level1_observations.json"), encoding="utf-8") as f:
            observations = json.load(f)
        with open(os.path.join(person_out, "level2_patterns.json"), encoding="utf-8") as f:
            patterns = json.load(f)
        with open(os.path.join(person_out, "level3_relations.json"), encoding="utf-8") as f:
            relations = json.load(f)
        rel_type = relations.get("relationship_type", "알수없음")
        intimacy = relations.get("intimacy_score", "?")
        self.knowledge_graph["nodes"].append({
            "id": person_id,
            "photo_count": len(observations),
            "estimated_age": patterns.get("estimated_age", ""),
            "gender": patterns.get("gender", ""),
            "lifestyle_summary": patterns.get("summary", ""),
            "relationship_type": rel_type,
            "intimacy_score": intimacy,
            "relationship_narrative": relations.get("relationship_narrative", ""),
            "age_based_reasoning": relations.get("age_based_reasoning", ""),
        })
        edge = relations.get("knowledge_graph_edge")
        if edge:
            self.knowledge_graph["edges"].append(edge)
        print(f"  ↻ {person_id} 재사용 (이미 분석됨 → LLM 재호출 생략)")
        return {
            "person_id": person_id,
            "photo_count": len(observations),
            "relationship_type": rel_type,
            "intimacy_score": intimacy,
            "observations": observations,
            "relations": relations,
        }

    def _fallback_narrative(self, person_id: str, patterns: dict, relations: dict) -> str:
        rel = relations.get("relationship_type", "알수없음")
        detail = relations.get("relationship_detail", "")
        score = relations.get("intimacy_score", "?")
        summary = patterns.get("summary", "")
        age = patterns.get("estimated_age", "")
        prompt = (
            f"다음 정보를 바탕으로 person_0(나)와 {person_id}의 관계를 2문장으로 서술하세요. "
            f"관계: {rel} ({detail}), 친밀도: {score}/10. "
            f"대상 나이: {age}. 라이프스타일: {summary}"
        )
        try:
            return invoke_with_retry(_get_llm(), [HumanMessage(content=prompt)])
        except Exception:
            return f"{rel} 관계({detail})로, 친밀도 {score}/10 수준입니다."

    # ── 전체 실행 ─────────────────────────────────────────────────

    def run(self, cluster_dir: str) -> list:
        person_dirs = sorted([
            os.path.join(cluster_dir, d)
            for d in os.listdir(cluster_dir)
            if os.path.isdir(os.path.join(cluster_dir, d)) and d.startswith("person_")
        ])
        print(f"총 {len(person_dirs)}명 후보 발견.\n")

        # ── 0단계: 미술작품 필터 ─────────────────────────────────
        real_dirs, filtered_out = [], []
        if ENABLE_ARTWORK_FILTER:
            print("[ 0단계 ] 실제 인물 검증 시작...")
            for pdir in person_dirs:
                pid = os.path.basename(pdir)
                is_real, reason = self.artwork_filter.is_real_person_cluster(pdir)
                if is_real:
                    real_dirs.append(pdir)
                    print(f"  ✔ {pid}: {reason}")
                else:
                    filtered_out.append({"person_id": pid, "reason": reason})
                    print(f"  ✗ {pid}: 제외 → {reason}")
            print(f"\n  실제 인물 {len(real_dirs)}명 / 제외 {len(filtered_out)}명\n")
        else:
            # 0단계 비활성화: 모든 클러스터를 실제 인물로 간주
            real_dirs = list(person_dirs)
            print(f"[ 0단계 ] 실제 인물 검증 건너뜀 (ENABLE_ARTWORK_FILTER=False) — 전체 {len(real_dirs)}명 처리\n")

        filter_path = os.path.join(self.output_dir, "artwork_filtered.json")
        with open(filter_path, "w", encoding="utf-8") as f:
            json.dump(filtered_out, f, ensure_ascii=False, indent=2)

        # ── 우선 단계: person_0 심층 분석 ────────────────────────
        p0_dir = os.path.join(cluster_dir, "person_0")
        person_zero_profile = None
        age_timeline: dict = {}
        p0_data = None

        if os.path.isdir(p0_dir) and p0_dir in real_dirs:
            p0_data = self._analyze_person_zero(p0_dir)
            person_zero_profile = p0_data["profile"]
            age_timeline = p0_data["age_timeline"]
            real_dirs = [d for d in real_dirs if os.path.basename(d) != "person_0"]
        elif os.path.isdir(p0_dir):
            # 0단계에서 필터링됐을 때도 person_0는 분석
            p0_data = self._analyze_person_zero(p0_dir)
            person_zero_profile = p0_data["profile"]
            age_timeline = p0_data["age_timeline"]

        # ── 1~3단계: 나머지 인물 프로파일링 ─────────────────────
        all_profiles = []
        all_level3: dict = {}  # 전체 재검증용

        for person_dir in real_dirs:
            try:
                pid0 = os.path.basename(person_dir)
                pout = os.path.join(self.output_dir, pid0)
                already = all(os.path.isfile(os.path.join(pout, fn)) for fn in (
                    "level1_observations.json", "level2_patterns.json", "level3_relations.json"))
                if already:
                    try:
                        result = self._reuse_person(person_dir)   # resume: 이미 분석됨 → 재사용
                    except Exception:
                        result = self.process_person(person_dir, person_zero_profile, age_timeline)
                else:
                    result = self.process_person(person_dir, person_zero_profile, age_timeline)
                pid = result["person_id"]
                all_level3[pid] = result.pop("relations")
                result.pop("observations", None)
                all_profiles.append(result)
            except Exception as e:
                print(f"  [Error] {person_dir}: {e}")

        # ── 전체 관계 재검증 ──────────────────────────────────────
        if all_level3 and person_zero_profile:
            print(f"\n[ 전체 관계 재검증 ] {len(all_level3)}명 대상 유일성·빈도 교정...")
            revised_level3 = self.global_validator.validate(all_level3, person_zero_profile)

            # 변경된 인물의 level3 파일 및 knowledge graph 업데이트
            for pid, rel in revised_level3.items():
                if rel.get("_globally_revised"):
                    rel_path = os.path.join(self.output_dir, pid, "level3_relations.json")
                    with open(rel_path, "w", encoding="utf-8") as f:
                        json.dump(rel, f, ensure_ascii=False, indent=2)
                    # knowledge graph 노드/엣지 동기화
                    for node in self.knowledge_graph["nodes"]:
                        if node["id"] == pid:
                            node["relationship_type"] = rel.get("relationship_type", node.get("relationship_type"))
                            node["intimacy_score"]    = rel.get("intimacy_score", node.get("intimacy_score"))
                            break
                    for edge in self.knowledge_graph["edges"]:
                        if edge.get("to") == pid:
                            edge["relation"] = rel.get("relationship_type", edge.get("relation"))
                            edge["weight"]   = rel.get("intimacy_score", edge.get("weight"))
                            break
            print(f"  ✔ 재검증 완료")
        else:
            revised_level3 = all_level3

        # ── person_0 컨텍스트 확장 ───────────────────────────────
        if os.path.isdir(p0_dir):
            print(f"\n[ person_0 컨텍스트 확장 ] EXIF 기반 탐색...")
            ctx = self.ctx_expander.expand(p0_dir, ORIGINAL_PHOTOS_DIR)
            ctx_path = os.path.join(self.output_dir, "person_0", "context_expansion.json")
            os.makedirs(os.path.dirname(ctx_path), exist_ok=True)
            with open(ctx_path, "w", encoding="utf-8") as f:
                json.dump(ctx, f, ensure_ascii=False, indent=2)
            print(f"  ✔ person_0 컨텍스트 저장: {ctx_path}")
            summary = ctx.get("summary", "")
            if summary and not self.knowledge_graph["nodes"][0].get("context_summary"):
                self.knowledge_graph["nodes"][0]["context_summary"] = summary

        # ── 결과 저장 ────────────────────────────────────────────
        kg_path = os.path.join(self.output_dir, "knowledge_graph.json")
        with open(kg_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_graph, f, ensure_ascii=False, indent=2)

        summary_path = os.path.join(self.output_dir, "all_profiles_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_profiles, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*55}")
        print(f"[완료] 모든 프로파일 생성 완료!")
        print(f"  결과 경로    : {self.output_dir}")
        print(f"  지식 그래프  : {kg_path}")
        print(f"  미술작품 제외: {filter_path}")
        self._print_knowledge_graph(kg_path)
        return all_profiles

    def _print_knowledge_graph(self, kg_path: str):
        with open(kg_path, "r", encoding="utf-8") as f:
            kg = json.load(f)
        print("\n" + "=" * 60)
        print("  지식 그래프 (Knowledge Graph)")
        print("=" * 60)

        p0_node = next((n for n in kg["nodes"] if n["id"] == "person_0"), {})
        print(f"\n  [나] 나이: {p0_node.get('estimated_age','?')} | 성별: {p0_node.get('gender','?')}")
        print(f"       {p0_node.get('lifestyle_summary','')[:80]}")

        node_map = {n["id"]: n for n in kg["nodes"]}
        for edge in kg["edges"]:
            dst = edge.get("to", "?")
            dst_info = node_map.get(dst, {})
            age_info = f"{dst_info.get('estimated_age','?')} {dst_info.get('gender','')}"
            print(f"\n  person_0 ──[{edge.get('relation','?')} / 친밀도 {edge.get('weight','?')}]──▶ {dst}")
            print(f"    나이/성별    : {age_info}")
            print(f"    사진 수      : {dst_info.get('photo_count','?')}장")
            print(f"    태그         : {'/'.join(edge.get('tags', []))}")
            narrative = dst_info.get("relationship_narrative", "")
            if narrative:
                print(f"    관계 서사    : {narrative[:80]}")
            reasoning = dst_info.get("age_based_reasoning", "")
            if reasoning:
                print(f"    나이 추론    : {reasoning[:80]}")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    orchestrator = PersonProfileOrchestrator(output_dir=OUTPUT_DIR)
    orchestrator.run(cluster_dir=CLUSTER_DIR)
