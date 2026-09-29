"""
data_info_agent.py — 라이프스타일 심층 분석 에이전트

사진에 EXIF datetime이 있으면 내가 직접 찍은 사진이다.
따라서 InsightFace 클러스터 결과 없이도 ORIGINAL_PHOTOS_DIR만으로 분석한다.

데이터 흐름:
1. ORIGINAL_PHOTOS_DIR에서 EXIF datetime이 있는 사진 전부 수집
2. 각 사진을 LLM으로 심층 분석 (음식/풍경/사람/활동/무드)
3. 스타일 집계: 음식 취향, 풍경 선호(바다·산·도시), 사람 만나는 빈도, 활동성
4. LLM이 통계 기반 페르소나 요약 생성
5. 수집된 전체 사진의 썸네일로 타임라인 HTML 생성

산출물 (OUTPUT_DIR):
  - collected_photos.json   : 수집된 사진 메타데이터
  - deep_analyses.json      : 사진별 LLM 심층 분석
  - style_stats.json        : 통계 집계
  - style_summary.json      : LLM 페르소나 요약
  - thumbnails/             : 전체 사진 썸네일
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import json
import base64
import re
import time
import math
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageOps  # type: ignore[import-untyped]
from PIL.ExifTags import TAGS, GPSTAGS  # type: ignore[import-untyped]

# 매우 큰 사진(예: 1.5억 픽셀)에서 DecompressionBomb 오류 방지 (본인 사진이라 안전)
Image.MAX_IMAGE_PIXELS = None

import vertexai
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage


# ── 설정 ──────────────────────────────────────────────────────
MODEL_ID    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PROJECT_ID  = os.environ["GOOGLE_CLOUD_PROJECT"]   # gcp-key.json 의 프로젝트
LOCATION    = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# 서비스 계정 키 (이 파일과 같은 폴더의 gcp-key.json 사용)
GCP_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# 원본 사진 폴더 (EXIF datetime이 있는 사진 = 내가 직접 찍은 사진)
ORIGINAL_PHOTOS_DIR = os.getenv("LUMINA_PHOTO_ROOT", "runtime/photos")

# 결과 저장 폴더
OUTPUT_DIR = os.path.join(ORIGINAL_PHOTOS_DIR, "person0_deep_profile")

# 최소 연도 필터 (이전 사진 제외)
MIN_YEAR = 2009

# LLM 분석 제한 (전체 분석 원하면 충분히 큰 값으로 설정)
MAX_DEEP_ANALYZE = 10000
ANALYZE_WORKERS  = 8

# 타임라인 썸네일 (OUTPUT_DIR/thumbnails/ 에 파일로 저장)
THUMBNAIL_SIZE    = (400, 300)
THUMBNAIL_QUALITY = 75

# 기존 분석 결과 재사용 여부 (HTML 반복 재생성 시 LLM 재호출 방지)
REUSE_CACHED_ANALYSES = True


# ── Vertex AI 초기화 (서비스 계정 JSON 키 기반) ───────────────────
from google.oauth2 import service_account

_credentials = service_account.Credentials.from_service_account_file(
    GCP_KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=_credentials)

llm = ChatVertexAI(
    model_name=MODEL_ID,
    temperature=0.2,
    # gemini-2.5-flash는 사고형 모델이라 '사고 토큰'이 출력 토큰 한도를 잠식한다.
    # 입력(통계 JSON 등)이 크면 사고에 토큰을 다 써 긴 JSON 출력이 중간에서 잘리고
    # (MAX_TOKENS) 파싱 실패 → 빈 dict로 떨어진다.
    # ① thinking_budget으로 사고 토큰을 묶어 출력용 토큰을 보호하고
    # ② max_output_tokens를 넉넉히 올려 긴 요약 JSON도 안 잘리게 한다.
    thinking_budget=2048,
    max_output_tokens=16384,
    credentials=_credentials,
    project=PROJECT_ID,
    location=LOCATION,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_jpeg_base64(image_path: str, max_side: int = 2048, quality: int = 90) -> str:
    """이미지를 표준 단일 프레임 JPEG로 재인코딩해 base64 반환.
    MPO(멀티프레임 JPEG)·CMYK·EXIF 회전 등을 정규화하여 LLM 전송 실패('mpo' 등)를 방지하고,
    과도하게 큰 이미지는 축소하여 토큰/메모리를 절약한다."""
    img = Image.open(image_path)
    # MPO 등 멀티프레임은 첫 프레임만 사용
    try:
        img.seek(0)
    except Exception:
        pass
    # EXIF 회전 반영 후 RGB 정규화
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_thumbnail_base64(image_path: str, size=THUMBNAIL_SIZE, quality=THUMBNAIL_QUALITY) -> str:
    """이미지를 축소해 base64로 인코딩 — HTML에 임베드용."""
    img = Image.open(image_path)
    img.thumbnail(size)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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


def invoke_with_retry(model, messages, max_retries: int = 3, base_delay: float = 2.0) -> str:
    for attempt in range(max_retries):
        try:
            return model.invoke(messages).content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise e


def extract_json_with_retry(model, messages, expect_list: bool = False, parse_retries: int = 2):
    """LLM 호출 + JSON 파싱. 사고형 모델 절단 등으로 빈 결과({}/[])가 나오면
    parse_retries 만큼 재호출해 상세 JSON을 최대한 확보한다."""
    result = [] if expect_list else {}
    for _ in range(parse_retries):
        text = invoke_with_retry(model, messages)
        result = extract_json_from_response(text, expect_list=expect_list)
        if result:
            break
    return result


# ── EXIF / GPS ────────────────────────────────────────────────

def _dms_to_dd(dms_val) -> float:
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
            try:
                result["datetime"] = datetime.strptime(str(dt_raw), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

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
    """크롭 파일명 → 원본 파일 경로 추론. 예) IMG_xxx_2.jpg → IMG_xxx.jpg"""
    stem = Path(cropped_filename).stem
    m = re.match(r'^(.+)_\d+$', stem)
    original_stem = m.group(1) if m else stem
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".HEIC", ".heic"):
        p = os.path.join(original_dir, original_stem + ext)
        if os.path.isfile(p):
            return p
    return None


# ══════════════════════════════════════════════════════════════
# 사진 수집기 — EXIF datetime이 있는 사진 전부
# ══════════════════════════════════════════════════════════════

class PhotoCollector:
    """EXIF datetime이 있는 사진 = 내가 직접 찍은 사진으로 간주하여 전부 수집."""

    def collect(self, original_dir: str) -> list:
        valid_exts = {".jpg", ".jpeg", ".png", ".heic"}
        all_files = sorted([
            os.path.join(original_dir, f)
            for f in os.listdir(original_dir)
            if Path(f).suffix.lower() in valid_exts
        ])

        print(f"  [Collector] 전체 {len(all_files)}장 EXIF 스캔 중...")
        photos = []
        skipped_year = 0
        for i, path in enumerate(all_files):
            exif = get_photo_exif(path)
            if not exif["datetime"]:
                continue
            if exif["datetime"].year < MIN_YEAR:
                skipped_year += 1
                continue
            photos.append({
                "path": path,
                "filename": os.path.basename(path),
                "datetime": exif["datetime"],
                "lat": exif["lat"],
                "lon": exif["lon"],
            })
            if (i + 1) % 500 == 0:
                print(f"    스캔 진행: {i+1}/{len(all_files)} (EXIF 보유 {len(photos)}장)")

        photos.sort(key=lambda x: x["datetime"])
        no_exif = len(all_files) - len(photos) - skipped_year
        print(f"  [Collector] EXIF 있음 {len(photos)}장 / EXIF 없음 {no_exif}장 / {MIN_YEAR}년 이전 {skipped_year}장 제외")
        return photos


# ══════════════════════════════════════════════════════════════
# 심층 사진 분석기
# ══════════════════════════════════════════════════════════════

DEEP_PHOTO_PROMPT = """\
당신은 사진 내용 분석 전문가입니다.
이 사진을 분석하여 person_0(사진 컬렉션 소유자)의 라이프스타일 파악에 필요한 정보를 추출하세요.
음식·풍경·사람·활동·무드 다섯 측면을 분류하고 JSON으로만 응답하세요.

{
  "scene": {
    "primary_type": "food/landscape/portrait/people/event/architecture/object/document/animal/other 중 하나",
    "description": "사진 내용 한 문장 설명"
  },
  "food": {
    "present": true/false,
    "cuisine": "한식/일식/중식/양식/디저트/카페음료/주류/패스트푸드/홈쿡/길거리음식/unknown",
    "meal_type": "breakfast/lunch/dinner/snack/dessert/drink/unknown",
    "specific_items": ["보이는 음식 이름 최대 5개"],
    "setting": "restaurant/home/cafe/street/picnic/bar/unknown"
  },
  "landscape": {
    "present": true/false,
    "type": "sea/beach/mountain/forest/river/lake/city/urban_street/countryside/sunset/night_view/snow/garden/sky/indoor_view/unknown",
    "weather": "sunny/cloudy/rainy/snowy/foggy/unknown",
    "season_hint": "spring/summer/autumn/winter/unknown",
    "description": "풍경 한 문장 설명"
  },
  "people": {
    "count_estimate": 정수 (사진 속 사람 수, 0이면 없음),
    "group_type": "none/solo/pair/small_group/large_group/crowd",
    "interaction": "posed/candid/event/meal/activity/none",
    "person0_likely_present": true/false
  },
  "activity": {
    "label": "한 단어로 활동명 (예: 식사, 산책, 등산, 여행, 업무, 휴식, 쇼핑, 관람, 운동)",
    "category": "dining/leisure/sport/travel/work/social/shopping/culture/rest/commute/unknown",
    "intensity": "high/medium/low/none",
    "indoor_outdoor": "indoor/outdoor/mixed"
  },
  "mood": {
    "atmosphere": "한 단어 (예: 활기찬/평화로운/로맨틱/격식있는)",
    "emotional_tone": "joyful/calm/excited/serious/melancholy/neutral"
  }
}
"""


class DeepPhotoAnalyzer:
    """원본 사진을 LLM으로 심층 분석."""

    def __init__(self, model):
        self.model = model

    def analyze(self, image_path: str) -> dict:
        try:
            b64 = encode_image_jpeg_base64(image_path)
            msg = HumanMessage(content=[
                {"type": "text", "text": DEEP_PHOTO_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ])
            return extract_json_with_retry(self.model, [msg])
        except Exception as e:
            print(f"    [DeepPhotoAnalyzer] 실패 ({os.path.basename(image_path)}): {e}")
            return {"error": str(e)}

    def analyze_batch(self, photo_entries: list, max_workers: int = ANALYZE_WORKERS) -> list:
        total = len(photo_entries)
        results: list = []
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.analyze, p["path"]): p for p in photo_entries}
            for fut in as_completed(futures):
                src = futures[fut]
                results.append({**src, "analysis": fut.result()})
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"    진행: {completed}/{total}")
        results.sort(key=lambda x: x["datetime"])
        return results


# ══════════════════════════════════════════════════════════════
# 스타일 집계기
# ══════════════════════════════════════════════════════════════

SEA_TYPES      = {"sea", "beach"}
MOUNTAIN_TYPES = {"mountain", "forest"}
CITY_TYPES     = {"city", "urban_street", "night_view"}
NATURE_EXTRAS  = {"river", "lake", "countryside", "garden", "sky", "sunset", "snow"}


class StyleProfiler:
    """deep 분석 결과 → 음식·풍경·사회·활동 통계 집계."""

    @staticmethod
    def _coerce_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _scalar(v):
        """LLM이 스칼라(문자열) 대신 list/None 등을 반환해도 Counter 키로 쓸 수 있는
        해시 가능 문자열로 변환. list면 첫 유효 문자열만 사용, 아니면 None."""
        if isinstance(v, str):
            v = v.strip()
            return v or None
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        return None

    @staticmethod
    def _dict(v):
        """LLM이 dict 대신 list 등을 반환해도 .get() 안전하도록 dict만 통과."""
        return v if isinstance(v, dict) else {}

    def aggregate(self, photos_with_analysis: list) -> dict:
        analyses = [p["analysis"] for p in photos_with_analysis if isinstance(p.get("analysis"), dict)]
        total = len(analyses)
        if total == 0:
            return {"total_photos": 0}

        # 음식 ───────────────────────
        food_analyses = [a for a in analyses if self._dict(a.get("food")).get("present")]
        cuisines = Counter()
        meal_types = Counter()
        food_settings = Counter()
        all_food_items: list = []
        for a in food_analyses:
            f = self._dict(a.get("food"))
            c = self._scalar(f.get("cuisine"))
            if c and c != "unknown":
                cuisines[c] += 1
            mt = self._scalar(f.get("meal_type"))
            if mt and mt != "unknown":
                meal_types[mt] += 1
            st = self._scalar(f.get("setting"))
            if st and st != "unknown":
                food_settings[st] += 1
            items = f.get("specific_items")
            if isinstance(items, list):
                all_food_items.extend(x.strip() for x in items if isinstance(x, str) and x.strip())
            elif isinstance(items, str) and items.strip():
                all_food_items.append(items.strip())

        # 풍경 ───────────────────────
        landscapes = [a for a in analyses if self._dict(a.get("landscape")).get("present")]
        landscape_types = Counter()
        weather_counter = Counter()
        season_counter = Counter()
        for a in landscapes:
            l = self._dict(a.get("landscape"))
            t = self._scalar(l.get("type"))
            if t and t != "unknown":
                landscape_types[t] += 1
            w = self._scalar(l.get("weather"))
            if w and w != "unknown":
                weather_counter[w] += 1
            s = self._scalar(l.get("season_hint"))
            if s and s != "unknown":
                season_counter[s] += 1

        sea_score      = sum(landscape_types.get(t, 0) for t in SEA_TYPES)
        mountain_score = sum(landscape_types.get(t, 0) for t in MOUNTAIN_TYPES)
        city_score     = sum(landscape_types.get(t, 0) for t in CITY_TYPES)
        nature_total   = sea_score + mountain_score + sum(landscape_types.get(t, 0) for t in NATURE_EXTRAS)

        scored = [("sea", sea_score), ("mountain", mountain_score), ("city", city_score)]
        preferred_landscape = max(scored, key=lambda x: x[1])[0] if any(s for _, s in scored) else "unknown"

        # 사람 ───────────────────────
        people_counts: list = []
        interactions = Counter()
        group_types = Counter()
        for a in analyses:
            people = self._dict(a.get("people"))
            cnt = self._coerce_int(people.get("count_estimate"))
            if cnt is not None:
                people_counts.append(cnt)
            inter = self._scalar(people.get("interaction"))
            if inter and inter != "none":
                interactions[inter] += 1
            gt = self._scalar(people.get("group_type"))
            if gt and gt not in ("none", "unknown"):
                group_types[gt] += 1

        solo  = sum(1 for c in people_counts if c <= 1)
        pair  = sum(1 for c in people_counts if c == 2)
        small = sum(1 for c in people_counts if 3 <= c <= 5)
        large = sum(1 for c in people_counts if c >= 6)
        with_others = pair + small + large
        avg_people  = sum(people_counts) / max(len(people_counts), 1)

        # 활동 ───────────────────────
        intensity = Counter()
        category = Counter()
        indoor_outdoor = Counter()
        activity_labels = Counter()
        for a in analyses:
            act = self._dict(a.get("activity"))
            inten = self._scalar(act.get("intensity"))
            if inten and inten != "none":
                intensity[inten] += 1
            cat = self._scalar(act.get("category"))
            if cat and cat != "unknown":
                category[cat] += 1
            io = self._scalar(act.get("indoor_outdoor"))
            if io and io != "unknown":
                indoor_outdoor[io] += 1
            lab = self._scalar(act.get("label"))
            if lab:
                activity_labels[lab] += 1

        intensity_total = sum(intensity.values()) or 1
        activity_score = (
            intensity.get("high", 0) * 3
            + intensity.get("medium", 0) * 2
            + intensity.get("low", 0) * 1
        ) / intensity_total  # 1.0(낮음) ~ 3.0(높음)

        # 무드 ───────────────────────
        mood_atmosphere = Counter()
        mood_tone = Counter()
        for a in analyses:
            m = self._dict(a.get("mood"))
            atm = self._scalar(m.get("atmosphere"))
            if atm:
                mood_atmosphere[atm] += 1
            tone = self._scalar(m.get("emotional_tone"))
            if tone:
                mood_tone[tone] += 1

        return {
            "total_photos": total,
            "food": {
                "ratio": round(len(food_analyses) / total, 3),
                "photo_count": len(food_analyses),
                "top_cuisines": cuisines.most_common(8),
                "meal_types": dict(meal_types),
                "settings": dict(food_settings),
                "frequent_items": Counter(all_food_items).most_common(10),
            },
            "landscape": {
                "ratio": round(len(landscapes) / total, 3),
                "photo_count": len(landscapes),
                "type_distribution": dict(landscape_types),
                "sea_score": sea_score,
                "mountain_score": mountain_score,
                "city_score": city_score,
                "nature_score": nature_total,
                "preferred_landscape": preferred_landscape,
                "weather": dict(weather_counter),
                "season": dict(season_counter),
            },
            "social": {
                "solo_ratio": round(solo / max(len(people_counts), 1), 3),
                "with_others_ratio": round(with_others / max(len(people_counts), 1), 3),
                "avg_people_per_photo": round(avg_people, 2),
                "distribution": {
                    "solo": solo, "pair": pair, "small_group": small, "large_group": large,
                },
                "interaction_types": dict(interactions),
                "group_types": dict(group_types),
            },
            "activity": {
                "intensity_distribution": dict(intensity),
                "activity_score_1to3": round(activity_score, 2),
                "category_distribution": dict(category),
                "indoor_outdoor": dict(indoor_outdoor),
                "frequent_labels": activity_labels.most_common(10),
            },
            "mood": {
                "atmosphere": dict(mood_atmosphere),
                "emotional_tone": dict(mood_tone),
            },
        }


# ══════════════════════════════════════════════════════════════
# LLM 페르소나 요약
# ══════════════════════════════════════════════════════════════

STYLE_SUMMARY_PROMPT = """\
다음은 person_0(나)의 사진 컬렉션을 정량 분석한 통계입니다.
이 데이터를 자연스러운 한국어 문장으로 해석하여 페르소나 프로필을 작성하세요.

집계 데이터:
{stats_json}

다음 항목을 채워 JSON으로만 응답하세요. 모든 필드는 반드시 작성하세요.

{{
  "food_style": "음식 취향 설명 2-3문장 (선호 종류, 외식·홈쿡 비율, 자주 먹는 메뉴 등을 데이터 기반으로)",
  "landscape_style": "풍경 취향 설명 2-3문장 (바다·산·도시 중 어디를 선호하는지 명확히 답하고 근거 제시)",
  "social_style": "사회 활동 설명 2-3문장 (혼자 vs 함께 비율, 만남의 규모, 모임 빈도)",
  "activity_style": "활동성 설명 2-3문장 (활동성 점수, 주요 활동 유형, 실내·실외 비율)",
  "overall_persona": "위 모든 것을 종합한 person_0 페르소나 3-4문장 서술",
  "highlight_tags": ["페르소나를 대표하는 한국어 태그 5-7개"]
}}
"""


class StyleSummarizer:
    def __init__(self, model):
        self.model = model

    def summarize(self, stats: dict) -> dict:
        prompt = STYLE_SUMMARY_PROMPT.format(
            stats_json=json.dumps(stats, ensure_ascii=False, indent=2, default=str)
        )
        skeleton = self._skeleton_summary(stats)
        try:
            llm_result = extract_json_with_retry(self.model, [HumanMessage(content=prompt)])
        except Exception as e:
            print(f"  [StyleSummarizer] 오류: {e}")
            llm_result = {}
        # LLM 파싱이 실패(절단 등)해도 통계 기반 골격으로 모든 페르소나 필드를 보장
        merged = dict(skeleton)
        for k, v in (llm_result or {}).items():
            if v in (None, "", [], {}):
                continue
            merged[k] = v
        return merged

    @staticmethod
    def _skeleton_summary(stats: dict) -> dict:
        """LLM 요약 실패 시에도 페르소나 필드를 채우는 통계 기반 골격."""
        def _top(pairs):
            return pairs[0][0] if isinstance(pairs, list) and pairs and isinstance(pairs[0], (list, tuple)) else None

        food = stats.get("food", {}) if isinstance(stats, dict) else {}
        land = stats.get("landscape", {})
        social = stats.get("social", {})
        activity = stats.get("activity", {})

        cuisine = _top(food.get("top_cuisines"))
        pref = land.get("preferred_landscape", "unknown")
        pref_kr = {"sea": "바다", "mountain": "산", "city": "도시"}.get(pref, "다양한 풍경")
        solo_ratio = social.get("solo_ratio", 0) or 0
        social_kr = "주로 혼자 다니는" if solo_ratio >= 0.6 else "사람들과 함께 어울리는"
        act_score = activity.get("activity_score_1to3", 0) or 0
        act_kr = "활동적인" if act_score >= 2.3 else ("보통 활동량의" if act_score >= 1.6 else "차분한")
        top_label = _top(activity.get("frequent_labels"))

        tags = [t for t in [cuisine, pref_kr, social_kr.replace(" ", ""), top_label] if t]

        return {
            "food_style": (f"음식 사진에서 '{cuisine}' 계열이 가장 자주 보입니다." if cuisine
                           else "음식 취향을 특정할 데이터가 부족합니다."),
            "landscape_style": f"풍경 사진은 {pref_kr}을(를) 선호하는 경향이 관찰됩니다.",
            "social_style": f"{social_kr} 편입니다 (혼자 비율 {round(solo_ratio*100)}%).",
            "activity_style": (f"{act_kr} 편이며, 주요 활동은 '{top_label}'입니다." if top_label
                               else f"{act_kr} 편입니다."),
            "overall_persona": (f"{pref_kr}을(를) 좋아하고 {social_kr} {act_kr} 사람입니다."),
            "highlight_tags": tags or ["정보 부족"],
        }


# ══════════════════════════════════════════════════════════════
# 타임라인 빌더 (썸네일 파일로 저장 → HTML은 상대경로 참조)
# HTML 크기 문제(수천 장 × 수십 KB = 수백 MB)를 방지하기 위해
# base64 임베드 대신 OUTPUT_DIR/thumbnails/ 폴더에 저장.
# ══════════════════════════════════════════════════════════════

class TimelineBuilder:
    def __init__(self, thumbnail_size=THUMBNAIL_SIZE, thumbnail_quality=THUMBNAIL_QUALITY):
        self.thumbnail_size = thumbnail_size
        self.thumbnail_quality = thumbnail_quality

    def _save_thumbnail(self, image_path: str, save_path: str) -> bool:
        try:
            img = Image.open(image_path)
            img.thumbnail(self.thumbnail_size)
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            img.save(save_path, format="JPEG", quality=self.thumbnail_quality, optimize=True)
            return True
        except Exception:
            return False

    def _to_event(self, idx: int, photo: dict, thumbnails_dir: str) -> dict:
        dt = photo["datetime"]
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt)
            except ValueError:
                dt = None

        # 파일명: 00001_원본stem.jpg — HTML에서 thumbnails/파일명 으로 참조
        thumb_filename = f"{idx:05d}_{Path(photo['filename']).stem}.jpg"
        thumb_abs = os.path.join(thumbnails_dir, thumb_filename)
        thumb_rel = f"thumbnails/{thumb_filename}" if self._save_thumbnail(photo["path"], thumb_abs) else ""

        return {
            "id": idx,
            "filename": photo["filename"],
            "datetime": dt.isoformat() if isinstance(dt, datetime) else str(photo["datetime"]),
            "timestamp_ms": int(dt.timestamp() * 1000) if isinstance(dt, datetime) else 0,
            "lat": photo.get("lat"),
            "lon": photo.get("lon"),
            "analysis": photo.get("analysis", {}),
            "thumbnail_path": thumb_rel,
        }

    def build(self, photos_with_analysis: list, output_dir: str) -> list:
        thumbnails_dir = os.path.join(output_dir, "thumbnails")
        os.makedirs(thumbnails_dir, exist_ok=True)
        total = len(photos_with_analysis)
        print(f"  [TimelineBuilder] 썸네일 {total}장 → {thumbnails_dir}")
        return [self._to_event(i, p, thumbnails_dir) for i, p in enumerate(photos_with_analysis)]


# ══════════════════════════════════════════════════════════════
# 시간축 집계 (연도/월/일 단위 + 캘린더 밀집도)
# ══════════════════════════════════════════════════════════════

class TemporalAggregator:
    """사진들을 연/월/일 버킷으로 묶고, 캘린더 밀집도 매트릭스 + 연·월별 통계 생성."""

    def __init__(self, profiler: StyleProfiler):
        self.profiler = profiler

    @staticmethod
    def _parse_dt(p: dict):
        dt = p.get("datetime")
        if isinstance(dt, str):
            try:
                return datetime.fromisoformat(dt)
            except ValueError:
                return None
        return dt if isinstance(dt, datetime) else None

    def aggregate(self, photos_with_analysis: list) -> dict:
        from collections import defaultdict

        by_year: dict = defaultdict(list)
        by_month: dict = defaultdict(list)
        calendar: dict = defaultdict(dict)  # year -> {"MM-DD": count}

        for p in photos_with_analysis:
            dt = self._parse_dt(p)
            if not dt:
                continue
            year = dt.year
            month_key = f"{year}-{dt.month:02d}"
            day_key = f"{dt.month:02d}-{dt.day:02d}"
            by_year[year].append(p)
            by_month[month_key].append(p)
            calendar[year][day_key] = calendar[year].get(day_key, 0) + 1

        # 연도별 통계 (programmatic only — LLM은 별도 단계에서)
        year_stats = {}
        for year, plist in by_year.items():
            year_stats[str(year)] = {
                "count": len(plist),
                "stats": self.profiler.aggregate(plist),
            }

        # 월별 통계
        month_stats = {}
        for mkey, plist in by_month.items():
            month_stats[mkey] = {
                "count": len(plist),
                "stats": self.profiler.aggregate(plist),
            }

        return {
            "years": sorted(by_year.keys()),
            "year_stats": year_stats,
            "month_stats": dict(sorted(month_stats.items())),
            "calendar": {str(y): cells for y, cells in sorted(calendar.items())},
        }


# ══════════════════════════════════════════════════════════════
# 연도별 LLM 서사 요약
# ══════════════════════════════════════════════════════════════

YEAR_NARRATIVE_PROMPT = """\
다음은 person_0가 {year}년 한 해 동안 찍은 사진 {count}장의 분석 통계입니다.
이 해 person_0의 라이프스타일을 데이터 기반으로 요약하세요.

집계 통계:
{stats_json}

JSON으로만 응답하세요. 모든 필드 필수:

{{
  "headline": "이 해를 한 문장으로 표현하는 헤드라인 (예: '바다와 카페가 가득한 한 해')",
  "summary": "이 해 라이프스타일 요약 2-3문장 (음식·풍경·사회·활동 측면)",
  "key_themes": ["이 해의 주요 테마 키워드 3-5개"],
  "highlight": "특기할 변화나 두드러진 패턴 1문장"
}}
"""


class YearlySummarizer:
    """연도별 LLM 서사 요약 생성."""

    def __init__(self, model):
        self.model = model

    def summarize(self, year_stats: dict) -> dict:
        results = {}
        for year, data in sorted(year_stats.items()):
            count = data.get("count", 0)
            if count < 3:
                results[year] = {
                    "headline": f"{year}년 사진 {count}장",
                    "summary": f"이 해 수집된 사진이 적어({count}장) 분석 데이터가 부족합니다.",
                    "key_themes": [],
                    "highlight": "",
                }
                continue
            prompt = YEAR_NARRATIVE_PROMPT.format(
                year=year,
                count=count,
                stats_json=json.dumps(data["stats"], ensure_ascii=False, indent=2, default=str),
            )
            try:
                res = extract_json_with_retry(self.model, [HumanMessage(content=prompt)])
                if not res:
                    res = {
                        "headline": f"{year}년 ({count}장)",
                        "summary": f"{year}년 사진 {count}장을 수집했으나 요약 생성에 실패했습니다.",
                        "key_themes": [],
                        "highlight": "",
                    }
                results[year] = res
                print(f"    [{year}] {results[year].get('headline', '')[:60]}")
            except Exception as e:
                results[year] = {"error": str(e), "headline": f"{year}년 요약 실패"}
        return results


# ══════════════════════════════════════════════════════════════
# HTML 타임라인 렌더러 (single-file, self-contained)
# ══════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>person_0 라이프스타일 타임라인</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Malgun Gothic', sans-serif; }
  body { background: #0d1117; color: #e6edf3; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  header { padding: 12px 24px; background: #161b22; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
  header h1 { font-size: 17px; font-weight: 600; }
  header .meta { font-size: 12px; color: #7d8590; }
  header .persona { flex: 1; min-width: 300px; font-size: 12px; color: #c9d1d9; max-height: 36px; overflow: hidden; text-overflow: ellipsis; }
  header .tags { display: flex; gap: 6px; flex-wrap: wrap; }
  header .tag-pill { padding: 3px 10px; background: #1f6feb33; border: 1px solid #1f6feb; color: #79c0ff; border-radius: 12px; font-size: 11px; }

  main { flex: 1; display: grid; grid-template-columns: minmax(0, 1.7fr) 380px; gap: 14px; padding: 14px; overflow: hidden; min-height: 0; }

  .photo-panel { display: flex; flex-direction: column; gap: 10px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; overflow: hidden; min-height: 0; }
  .photo-display { flex: 1; display: flex; align-items: center; justify-content: center; background: #010409; border-radius: 6px; overflow: hidden; min-height: 0; }
  .photo-display img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .photo-info { display: flex; justify-content: space-between; gap: 12px; font-size: 12px; color: #7d8590; align-items: center; flex-wrap: wrap; }
  .photo-info .filename { color: #e6edf3; font-weight: 500; }
  .photo-info .badge { padding: 2px 8px; border-radius: 10px; background: #f59e0b; color: #0d1117; font-size: 10px; font-weight: 700; }
  .photo-info .desc { color: #8b949e; font-style: italic; flex-basis: 100%; }
  .photo-tags { display: flex; flex-wrap: wrap; gap: 6px; }
  .tag { padding: 3px 10px; background: #21262d; border: 1px solid #30363d; border-radius: 12px; font-size: 11px; color: #c9d1d9; }
  .tag.food { border-color: #fb8500; color: #fb8500; }
  .tag.landscape { border-color: #06b6d4; color: #06b6d4; }
  .tag.people { border-color: #a855f7; color: #a855f7; }
  .tag.activity { border-color: #22c55e; color: #22c55e; }
  .tag.mood { border-color: #ec4899; color: #ec4899; }

  .stats-panel { display: flex; flex-direction: column; gap: 10px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; overflow-y: auto; }
  .stats-panel h2 { font-size: 12px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
  .window-controls { display: flex; gap: 6px; }
  .window-controls button { flex: 1; padding: 6px 8px; background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; cursor: pointer; font-size: 11px; transition: all 0.15s; }
  .window-controls button:hover { background: #30363d; }
  .window-controls button.active { background: #1f6feb; border-color: #1f6feb; color: white; }

  .stat-card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 10px 12px; }
  .stat-card-title { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; font-size: 12px; font-weight: 600; color: #c9d1d9; }
  .stat-card-title .count { color: #7d8590; font-weight: normal; font-size: 11px; }
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 11px; }
  .bar-row .label { flex: 0 0 80px; color: #c9d1d9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .bar-row .bar-bg { flex: 1; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }
  .bar-row .bar-fill { height: 100%; background: #1f6feb; border-radius: 3px; transition: width 0.25s; }
  .bar-row .bar-fill.sea      { background: linear-gradient(90deg, #0ea5e9, #06b6d4); }
  .bar-row .bar-fill.mountain { background: linear-gradient(90deg, #22c55e, #15803d); }
  .bar-row .bar-fill.city     { background: linear-gradient(90deg, #f59e0b, #d97706); }
  .bar-row .bar-fill.nature   { background: linear-gradient(90deg, #84cc16, #65a30d); }
  .bar-row .bar-fill.food     { background: linear-gradient(90deg, #fb8500, #ea580c); }
  .bar-row .bar-fill.high     { background: linear-gradient(90deg, #ef4444, #dc2626); }
  .bar-row .bar-fill.medium   { background: linear-gradient(90deg, #f59e0b, #ea580c); }
  .bar-row .bar-fill.low      { background: linear-gradient(90deg, #6b7280, #4b5563); }
  .bar-row .value { flex: 0 0 32px; text-align: right; color: #7d8590; font-variant-numeric: tabular-nums; }

  footer { background: #161b22; border-top: 1px solid #30363d; padding: 10px 24px 14px; }
  .timeline-controls { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .play-btn { width: 36px; height: 36px; border-radius: 50%; background: #1f6feb; color: white; border: none; cursor: pointer; font-size: 14px; display: flex; align-items: center; justify-content: center; font-weight: bold; }
  .play-btn:hover { background: #388bfd; }
  .time-label { font-size: 13px; color: #e6edf3; font-variant-numeric: tabular-nums; min-width: 170px; font-weight: 500; }
  .speed-select { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 10px; border-radius: 6px; cursor: pointer; }
  .legend { display: flex; gap: 12px; margin-left: auto; font-size: 11px; color: #7d8590; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

  .timeline-track { position: relative; height: 56px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; cursor: pointer; user-select: none; }
  .timeline-events { position: absolute; inset: 0; }
  .event-marker { position: absolute; top: 6px; width: 4px; height: 22px; background: #1f6feb; border-radius: 2px; cursor: pointer; opacity: 0.7; transition: all 0.15s; }
  .event-marker:hover { opacity: 1; transform: scaleY(1.15); }
  .event-marker.anchor    { background: #f59e0b; height: 30px; top: 2px; width: 5px; opacity: 1; }
  .event-marker.food      { background: #fb8500; }
  .event-marker.landscape { background: #06b6d4; }
  .event-marker.people    { background: #a855f7; }
  .event-marker.portrait  { background: #a855f7; }
  .timeline-axis { position: absolute; bottom: 3px; left: 0; right: 0; display: flex; justify-content: space-between; font-size: 9px; color: #7d8590; padding: 0 4px; pointer-events: none; }
  .playhead { position: absolute; top: 0; bottom: 0; width: 2px; background: #f85149; z-index: 10; pointer-events: none; box-shadow: 0 0 8px rgba(248,81,73,0.5); }
  .playhead::before { content: ''; position: absolute; top: -5px; left: -6px; width: 14px; height: 14px; background: #f85149; border-radius: 50%; box-shadow: 0 0 10px rgba(248,81,73,0.7); }

  .empty { color: #7d8590; font-size: 12px; text-align: center; padding: 40px 20px; }
  .persona-card { background: linear-gradient(135deg, #1f6feb22, #a855f722); border: 1px solid #1f6feb; }
  .persona-card .stat-card-title { color: #79c0ff; }
  .persona-card p { font-size: 11px; line-height: 1.6; color: #c9d1d9; margin: 4px 0; }

  /* 캘린더 히트맵 */
  .calendar-panel { background: #161b22; border-top: 1px solid #30363d; padding: 10px 16px; }
  .calendar-header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
  .calendar-header h2 { font-size: 12px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.6px; }
  .calendar-header select { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  .calendar-stats { font-size: 11px; color: #7d8590; margin-left: auto; }
  .calendar-body { display: flex; gap: 16px; align-items: center; }
  .calendar-grid { display: grid; grid-template-columns: 26px repeat(31, 1fr); gap: 2px; flex: 1; max-width: 100%; }
  .calendar-row-label { font-size: 9px; color: #7d8590; text-align: right; padding-right: 4px; line-height: 12px; align-self: center; }
  .calendar-col-label { font-size: 8px; color: #7d8590; text-align: center; line-height: 10px; }
  .calendar-cell { aspect-ratio: 1; min-height: 12px; border-radius: 2px; background: #0d1117; border: 1px solid #21262d; cursor: pointer; transition: outline 0.1s; }
  .calendar-cell:hover { outline: 1px solid #79c0ff; outline-offset: 1px; }
  .calendar-cell.l1 { background: #0e4429; border-color: #0e4429; }
  .calendar-cell.l2 { background: #006d32; border-color: #006d32; }
  .calendar-cell.l3 { background: #26a641; border-color: #26a641; }
  .calendar-cell.l4 { background: #39d353; border-color: #39d353; }
  .calendar-cell.empty-cell { visibility: hidden; }
  .calendar-cell.today { outline: 2px solid #f85149; outline-offset: 1px; }
  .calendar-legend { display: flex; align-items: center; gap: 4px; font-size: 10px; color: #7d8590; flex-shrink: 0; }
  .legend-cell { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }

  /* 시간 단위 스텝 */
  .step-btn { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 13px; min-width: 32px; }
  .step-btn:hover { background: #30363d; }

  /* 연도별 분석 카드 */
  .year-analysis-card { background: linear-gradient(135deg, #f5970022, #1f6feb22); border: 1px solid #f59e0b66; }
  .year-analysis-card .stat-card-title { color: #f59e0b; }
  .year-headline { font-size: 13px; font-weight: 600; color: #e6edf3; margin: 4px 0; }
  .year-summary { font-size: 11px; line-height: 1.6; color: #c9d1d9; margin: 4px 0; }
  .year-themes { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
  .year-themes .tag { font-size: 10px; padding: 2px 8px; }
  .year-highlight { font-size: 11px; color: #79c0ff; margin-top: 6px; padding-top: 6px; border-top: 1px solid #30363d; }
</style>
</head>
<body>
<header>
  <h1>person_0 라이프스타일 타임라인</h1>
  <span class="meta" id="meta-summary">로딩 중...</span>
  <div class="persona" id="persona-text"></div>
  <div class="tags" id="header-tags"></div>
</header>

<main>
  <section class="photo-panel">
    <div class="photo-display" id="photo-display">
      <span class="empty">사진 선택 또는 재생 시작</span>
    </div>
    <div class="photo-info">
      <span class="filename" id="photo-filename">—</span>
      <span id="photo-datetime">—</span>
      <span class="badge" id="photo-badge" style="display:none">P0 ANCHOR</span>
      <span class="desc" id="photo-desc"></span>
    </div>
    <div class="photo-tags" id="photo-tags"></div>
  </section>

  <aside class="stats-panel">
    <h2>윈도우 통계</h2>
    <div class="window-controls">
      <button data-days="7">±3.5일</button>
      <button data-days="30" class="active">±15일</button>
      <button data-days="90">±45일</button>
      <button data-days="-1">전체</button>
    </div>
    <div id="stats-content"></div>

    <div class="stat-card year-analysis-card">
      <div class="stat-card-title">
        연도별 분석
        <select id="year-selector" style="background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:2px 6px;border-radius:4px;font-size:11px;"></select>
      </div>
      <div id="year-analysis-content"></div>
    </div>
  </aside>
</main>

<section class="calendar-panel">
  <div class="calendar-header">
    <h2>캘린더 밀집도</h2>
    <select id="calendar-year"></select>
    <span class="calendar-stats" id="calendar-stats"></span>
  </div>
  <div class="calendar-body">
    <div class="calendar-grid" id="calendar-grid"></div>
    <div class="calendar-legend">
      <span>적음</span>
      <span class="legend-cell" style="background:#0d1117;border:1px solid #21262d"></span>
      <span class="legend-cell" style="background:#0e4429"></span>
      <span class="legend-cell" style="background:#006d32"></span>
      <span class="legend-cell" style="background:#26a641"></span>
      <span class="legend-cell" style="background:#39d353"></span>
      <span>많음</span>
    </div>
  </div>
</section>

<footer>
  <div class="timeline-controls">
    <button class="step-btn" id="step-back" title="이전 단위">◀</button>
    <button class="play-btn" id="play-btn">▶</button>
    <button class="step-btn" id="step-forward" title="다음 단위">▶</button>
    <select class="speed-select" id="step-unit">
      <option value="hour">1시간</option>
      <option value="day" selected>1일</option>
      <option value="week">1주일</option>
      <option value="month">1개월</option>
      <option value="year">1년</option>
    </select>
    <span class="time-label" id="time-label">—</span>
    <div class="legend">
      <span class="legend-item"><span class="legend-dot" style="background:#fb8500"></span>음식</span>
      <span class="legend-item"><span class="legend-dot" style="background:#06b6d4"></span>풍경</span>
      <span class="legend-item"><span class="legend-dot" style="background:#a855f7"></span>인물</span>
      <span class="legend-item"><span class="legend-dot" style="background:#1f6feb"></span>기타</span>
    </div>
  </div>
  <div class="timeline-track" id="timeline-track">
    <div class="timeline-events" id="timeline-events"></div>
    <div class="playhead" id="playhead" style="left:0"></div>
    <div class="timeline-axis" id="timeline-axis"></div>
  </div>
</footer>

<script id="timeline-data" type="application/json">__TIMELINE_DATA__</script>
<script id="style-summary" type="application/json">__STYLE_SUMMARY__</script>
<script id="temporal-data" type="application/json">__TEMPORAL_DATA__</script>
<script id="yearly-data" type="application/json">__YEARLY_DATA__</script>

<script>
const DATA = JSON.parse(document.getElementById('timeline-data').textContent);
const SUMMARY = JSON.parse(document.getElementById('style-summary').textContent);
const TEMPORAL = JSON.parse(document.getElementById('temporal-data').textContent);
const YEARLY = JSON.parse(document.getElementById('yearly-data').textContent);
const events = DATA.events || [];

if (!events.length) {
  document.getElementById('photo-display').innerHTML = '<span class="empty">사진이 없습니다</span>';
} else {
  init();
}

function init() {
  const tMin = Math.min(...events.map(e => e.timestamp_ms));
  const tMax = Math.max(...events.map(e => e.timestamp_ms));
  const tRange = Math.max(tMax - tMin, 1);

  const track         = document.getElementById('timeline-track');
  const eventsLayer   = document.getElementById('timeline-events');
  const playhead      = document.getElementById('playhead');
  const timeLabel     = document.getElementById('time-label');
  const metaSummary   = document.getElementById('meta-summary');
  const personaText   = document.getElementById('persona-text');
  const headerTags    = document.getElementById('header-tags');

  // 헤더 메타
  const dStart = new Date(tMin).toISOString().slice(0, 10);
  const dEnd = new Date(tMax).toISOString().slice(0, 10);
  const yearsAvailable = (TEMPORAL && TEMPORAL.years) || [];
  metaSummary.textContent = `${dStart} ~ ${dEnd}  •  총 ${events.length}장  •  ${yearsAvailable.length}개 연도`;

  if (SUMMARY && SUMMARY.overall_persona) {
    personaText.textContent = SUMMARY.overall_persona;
    personaText.title = SUMMARY.overall_persona;
  }
  if (SUMMARY && SUMMARY.highlight_tags) {
    SUMMARY.highlight_tags.forEach(t => {
      const el = document.createElement('span');
      el.className = 'tag-pill';
      el.textContent = t;
      headerTags.appendChild(el);
    });
  }

  // 이벤트 마커
  events.forEach((ev) => {
    const m = document.createElement('div');
    m.className = 'event-marker';
    const pct = ((ev.timestamp_ms - tMin) / tRange) * 100;
    m.style.left = `calc(${pct}% - 2px)`;
    const t = (ev.analysis && ev.analysis.scene && ev.analysis.scene.primary_type) || '';
    if (t === 'food') m.classList.add('food');
    else if (t === 'landscape') m.classList.add('landscape');
    else if (t === 'people' || t === 'portrait') m.classList.add('people');
    m.title = `${ev.filename}\n${ev.datetime}`;
    m.addEventListener('click', (e) => { e.stopPropagation(); setPlayhead(ev.timestamp_ms); });
    eventsLayer.appendChild(m);
  });

  // 축 라벨
  const axis = document.getElementById('timeline-axis');
  for (let i = 0; i < 5; i++) {
    const t = tMin + (tRange * i / 4);
    const span = document.createElement('span');
    span.textContent = new Date(t).toISOString().slice(0, 10);
    axis.appendChild(span);
  }

  let currentT = tMin;
  let playing = false;
  let playInterval = null;
  let windowDays = 30;
  let currentCalendarYear = null;
  let currentYearAnalysis = null;

  function setPlayhead(t) {
    currentT = Math.max(tMin, Math.min(tMax, t));
    const pct = ((currentT - tMin) / tRange) * 100;
    playhead.style.left = `${pct}%`;
    timeLabel.textContent = new Date(currentT).toISOString().slice(0, 19).replace('T', ' ');

    const nearest = events.reduce((a, b) =>
      Math.abs(b.timestamp_ms - currentT) < Math.abs(a.timestamp_ms - currentT) ? b : a
    );
    showPhoto(nearest);
    renderStats(currentT, windowDays);

    // 캘린더/연도 분석 동기화
    const year = new Date(currentT).getFullYear();
    if (year !== currentCalendarYear) {
      currentCalendarYear = year;
      const calSel = document.getElementById('calendar-year');
      if (calSel.value != String(year)) calSel.value = String(year);
      renderCalendar(year);
    }
    if (year !== currentYearAnalysis) {
      currentYearAnalysis = year;
      const yrSel = document.getElementById('year-selector');
      if (yrSel.value != String(year)) yrSel.value = String(year);
      renderYearAnalysis(year);
    }
    // 캘린더 'today' 마커 갱신
    document.querySelectorAll('.calendar-cell.today').forEach(c => c.classList.remove('today'));
    const d = new Date(currentT);
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    const todayCell = document.querySelector(`.calendar-cell[data-key="${mm}-${dd}"]`);
    if (todayCell) todayCell.classList.add('today');
  }

  function showPhoto(ev) {
    const display = document.getElementById('photo-display');
    if (ev.thumbnail_path) {
      display.innerHTML = `<img src="${ev.thumbnail_path}" loading="lazy">`;
    } else {
      display.innerHTML = '<span class="empty">썸네일 없음</span>';
    }
    document.getElementById('photo-filename').textContent = ev.filename;
    document.getElementById('photo-datetime').textContent = ev.datetime.replace('T', ' ').slice(0, 19);

    const a = ev.analysis || {};
    document.getElementById('photo-desc').textContent =
      (a.scene && a.scene.description) || '';

    const tags = document.getElementById('photo-tags');
    tags.innerHTML = '';
    if (a.food && a.food.present) {
      const c = a.food.cuisine || '';
      const m = a.food.meal_type || '';
      const txt = [c, m].filter(x => x && x !== 'unknown').join(' / ');
      if (txt) addTag(tags, '음식: ' + txt, 'food');
    }
    if (a.landscape && a.landscape.present) {
      const t = a.landscape.type || '';
      if (t && t !== 'unknown') addTag(tags, '풍경: ' + t, 'landscape');
    }
    const cnt = parseInt(a.people && a.people.count_estimate) || 0;
    if (cnt > 0) addTag(tags, '인물: ' + cnt + '명', 'people');
    if (a.activity && a.activity.label) addTag(tags, '활동: ' + a.activity.label, 'activity');
    if (a.mood && a.mood.emotional_tone) addTag(tags, '무드: ' + a.mood.emotional_tone, 'mood');
  }

  function addTag(container, text, cls) {
    const t = document.createElement('span');
    t.className = 'tag ' + cls;
    t.textContent = text;
    container.appendChild(t);
  }

  function renderStats(centerT, days) {
    let inWindow;
    if (days < 0) {
      inWindow = events.slice();
    } else {
      const halfMs = days * 24 * 60 * 60 * 1000 / 2;
      inWindow = events.filter(e => Math.abs(e.timestamp_ms - centerT) <= halfMs);
    }
    document.getElementById('stats-content').innerHTML = buildStatsHTML(inWindow, days);
  }

  function buildStatsHTML(window, days) {
    if (!window.length) {
      return '<div class="empty">이 시간 윈도우에 사진이 없습니다</div>';
    }

    const food = window.filter(e => e.analysis && e.analysis.food && e.analysis.food.present);
    const landscape = window.filter(e => e.analysis && e.analysis.landscape && e.analysis.landscape.present);

    const seaT      = ['sea', 'beach'];
    const mountainT = ['mountain', 'forest'];
    const cityT     = ['city', 'urban_street', 'night_view'];
    let sea = 0, mountain = 0, city = 0, otherNature = 0;
    landscape.forEach(e => {
      const t = e.analysis.landscape.type;
      if (seaT.includes(t)) sea++;
      else if (mountainT.includes(t)) mountain++;
      else if (cityT.includes(t)) city++;
      else otherNature++;
    });

    let solo = 0, pair = 0, smallG = 0, largeG = 0;
    window.forEach(e => {
      const c = parseInt(e.analysis && e.analysis.people && e.analysis.people.count_estimate) || 0;
      if (c <= 1) solo++;
      else if (c === 2) pair++;
      else if (c <= 5) smallG++;
      else largeG++;
    });
    const withOthers = pair + smallG + largeG;

    let high = 0, medium = 0, low = 0;
    window.forEach(e => {
      const i = e.analysis && e.analysis.activity && e.analysis.activity.intensity;
      if (i === 'high') high++;
      else if (i === 'medium') medium++;
      else if (i === 'low') low++;
    });
    const totalAct = high + medium + low;
    const activityScore = totalAct
      ? ((high*3 + medium*2 + low) / totalAct).toFixed(2)
      : '—';

    const cuisines = {};
    food.forEach(e => {
      const c = e.analysis.food.cuisine;
      if (c && c !== 'unknown') cuisines[c] = (cuisines[c] || 0) + 1;
    });
    const topCuisines = Object.entries(cuisines).sort((a,b) => b[1]-a[1]).slice(0, 5);

    const labels = {};
    window.forEach(e => {
      const l = e.analysis && e.analysis.activity && e.analysis.activity.label;
      if (l) labels[l] = (labels[l] || 0) + 1;
    });
    const topLabels = Object.entries(labels).sort((a,b) => b[1]-a[1]).slice(0, 5);

    function bar(label, val, total, cls) {
      const pct = total ? Math.round(val/total*100) : 0;
      return `<div class="bar-row">
        <span class="label">${escapeHtml(label)}</span>
        <span class="bar-bg"><span class="bar-fill ${cls}" style="width:${pct}%"></span></span>
        <span class="value">${val}</span>
      </div>`;
    }

    const windowLabel = days < 0 ? '전체' : `±${(days/2)}일`;
    let html = `<div class="stat-card">
      <div class="stat-card-title">사진 ${window.length}장 <span class="count">(${windowLabel})</span></div>
    </div>`;

    html += `<div class="stat-card">
      <div class="stat-card-title">풍경 선호 <span class="count">${landscape.length}장</span></div>
      ${bar('바다', sea, landscape.length, 'sea')}
      ${bar('산/숲', mountain, landscape.length, 'mountain')}
      ${bar('도시', city, landscape.length, 'city')}
      ${bar('기타 자연', otherNature, landscape.length, 'nature')}
    </div>`;

    html += `<div class="stat-card">
      <div class="stat-card-title">사회성 <span class="count">평균 ${(window.reduce((s,e)=>s+(parseInt(e.analysis&&e.analysis.people&&e.analysis.people.count_estimate)||0),0)/window.length).toFixed(1)}명/장</span></div>
      ${bar('혼자(0-1)', solo, window.length, '')}
      ${bar('둘(2)', pair, window.length, '')}
      ${bar('소그룹(3-5)', smallG, window.length, '')}
      ${bar('대그룹(6+)', largeG, window.length, '')}
    </div>`;

    html += `<div class="stat-card">
      <div class="stat-card-title">활동성 <span class="count">${activityScore !== '—' ? '지수 ' + activityScore + '/3' : ''}</span></div>
      ${bar('높음', high, totalAct, 'high')}
      ${bar('중간', medium, totalAct, 'medium')}
      ${bar('낮음', low, totalAct, 'low')}
    </div>`;

    if (topCuisines.length) {
      html += `<div class="stat-card">
        <div class="stat-card-title">음식 종류 <span class="count">${food.length}장</span></div>
        ${topCuisines.map(([c,n]) => bar(c, n, food.length, 'food')).join('')}
      </div>`;
    }

    if (topLabels.length) {
      html += `<div class="stat-card">
        <div class="stat-card-title">주요 활동</div>
        ${topLabels.map(([l,n]) => bar(l, n, window.length, '')).join('')}
      </div>`;
    }

    return html;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // 트랙 드래그/클릭
  let dragging = false;
  function trackPosToTime(clientX) {
    const rect = track.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    return tMin + tRange * pct;
  }
  track.addEventListener('mousedown', (e) => {
    if (e.target.classList.contains('event-marker')) return;
    dragging = true;
    setPlayhead(trackPosToTime(e.clientX));
  });
  document.addEventListener('mousemove', (e) => {
    if (dragging) setPlayhead(trackPosToTime(e.clientX));
  });
  document.addEventListener('mouseup', () => { dragging = false; });

  // 시간 단위 스텝
  const playBtn = document.getElementById('play-btn');
  const unitSelect = document.getElementById('step-unit');

  function stepTime(direction) {
    const unit = unitSelect.value;
    const d = new Date(currentT);
    if (unit === 'hour')      d.setHours(d.getHours() + direction);
    else if (unit === 'day')  d.setDate(d.getDate() + direction);
    else if (unit === 'week') d.setDate(d.getDate() + direction * 7);
    else if (unit === 'month')d.setMonth(d.getMonth() + direction);
    else if (unit === 'year') d.setFullYear(d.getFullYear() + direction);
    let nt = d.getTime();
    if (nt > tMax) nt = tMin;
    if (nt < tMin) nt = tMax;
    setPlayhead(nt);
  }
  function play() {
    playing = true;
    playBtn.textContent = '⏸';
    playInterval = setInterval(() => stepTime(1), 600);
  }
  function pause() {
    playing = false;
    playBtn.textContent = '▶';
    if (playInterval) clearInterval(playInterval);
  }
  playBtn.addEventListener('click', () => playing ? pause() : play());
  document.getElementById('step-back').addEventListener('click', () => stepTime(-1));
  document.getElementById('step-forward').addEventListener('click', () => stepTime(1));

  // 윈도우 크기 버튼
  document.querySelectorAll('.window-controls button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.window-controls button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      windowDays = parseInt(btn.dataset.days);
      renderStats(currentT, windowDays);
    });
  });

  // 키보드 단축키
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); playing ? pause() : play(); }
    else if (e.code === 'ArrowLeft') { stepTime(-1); }
    else if (e.code === 'ArrowRight') { stepTime(1); }
  });

  // ── 캘린더 히트맵 ─────────────────────────────────
  function countToLevel(count, max) {
    if (!count || count === 0) return 0;
    if (max <= 1) return 1;
    const ratio = count / max;
    if (ratio < 0.25) return 1;
    if (ratio < 0.5) return 2;
    if (ratio < 0.75) return 3;
    return 4;
  }

  function renderCalendar(year) {
    const grid = document.getElementById('calendar-grid');
    grid.innerHTML = '';
    const yearStr = String(year);
    const calData = (TEMPORAL.calendar && TEMPORAL.calendar[yearStr]) || {};

    // 헤더 행: 빈 셀 + 일자 1~31
    const corner = document.createElement('div');
    corner.className = 'calendar-col-label';
    grid.appendChild(corner);
    for (let d = 1; d <= 31; d++) {
      const hdr = document.createElement('div');
      hdr.className = 'calendar-col-label';
      hdr.textContent = (d % 5 === 0 || d === 1) ? d : '';
      grid.appendChild(hdr);
    }

    const monthNames = ['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'];
    let total = 0;
    let maxCount = 0;
    Object.values(calData).forEach(c => { total += c; if (c > maxCount) maxCount = c; });

    for (let m = 1; m <= 12; m++) {
      const lbl = document.createElement('div');
      lbl.className = 'calendar-row-label';
      lbl.textContent = monthNames[m - 1];
      grid.appendChild(lbl);

      const daysInMonth = new Date(year, m, 0).getDate();
      for (let d = 1; d <= 31; d++) {
        const cell = document.createElement('div');
        cell.className = 'calendar-cell';
        if (d > daysInMonth) {
          cell.classList.add('empty-cell');
        } else {
          const mm = String(m).padStart(2, '0');
          const dd = String(d).padStart(2, '0');
          const key = `${mm}-${dd}`;
          const count = calData[key] || 0;
          const level = countToLevel(count, maxCount);
          if (level) cell.classList.add('l' + level);
          cell.title = `${year}-${mm}-${dd}: ${count}장`;
          cell.dataset.key = key;
          if (count > 0) {
            cell.addEventListener('click', () => {
              // 그 날짜의 첫 사진으로 점프 (12:00 기준)
              const target = new Date(year, m - 1, d, 12, 0, 0).getTime();
              const nearest = events.reduce((a, b) => {
                const aDay = new Date(a.timestamp_ms).toDateString();
                const bDay = new Date(b.timestamp_ms).toDateString();
                const tgtDay = new Date(target).toDateString();
                if (aDay === tgtDay && bDay !== tgtDay) return a;
                if (bDay === tgtDay && aDay !== tgtDay) return b;
                return Math.abs(b.timestamp_ms - target) < Math.abs(a.timestamp_ms - target) ? b : a;
              });
              setPlayhead(nearest.timestamp_ms);
            });
          }
        }
        grid.appendChild(cell);
      }
    }

    const yStat = (TEMPORAL.year_stats && TEMPORAL.year_stats[yearStr]) || {};
    document.getElementById('calendar-stats').textContent =
      `${year}년 · 총 ${yStat.count || total}장 · 활성일 ${Object.keys(calData).length}일 · 최대 ${maxCount}장/일`;
  }

  // ── 연도별 분석 ───────────────────────────────────
  function renderYearAnalysis(year) {
    const content = document.getElementById('year-analysis-content');
    const yearly = YEARLY[String(year)];
    const yStat = (TEMPORAL.year_stats && TEMPORAL.year_stats[String(year)]) || {};
    if (!yearly) {
      content.innerHTML = '<div class="empty" style="padding:12px">이 해 LLM 요약 없음</div>';
      return;
    }
    let html = '';
    if (yearly.headline) html += `<div class="year-headline">${escapeHtml(yearly.headline)}</div>`;
    if (yearly.summary) html += `<p class="year-summary">${escapeHtml(yearly.summary)}</p>`;
    if (yearly.key_themes && yearly.key_themes.length) {
      html += '<div class="year-themes">' +
        yearly.key_themes.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('') + '</div>';
    }
    if (yStat.count) {
      const s = yStat.stats || {};
      const ls = s.landscape || {};
      const sc = s.social || {};
      html += `<p class="year-summary" style="color:#7d8590;font-size:10px">` +
        `${yStat.count}장 · 바다 ${ls.sea_score||0}/산 ${ls.mountain_score||0}/도시 ${ls.city_score||0} · ` +
        `함께 ${Math.round((sc.with_others_ratio||0)*100)}%</p>`;
    }
    if (yearly.highlight) html += `<p class="year-highlight">💡 ${escapeHtml(yearly.highlight)}</p>`;
    content.innerHTML = html;
  }

  // 캘린더/연도 셀렉터 초기화
  const calYearSel = document.getElementById('calendar-year');
  const yrAnalysisSel = document.getElementById('year-selector');
  yearsAvailable.forEach(y => {
    const o1 = document.createElement('option'); o1.value = y; o1.textContent = y + '년';
    calYearSel.appendChild(o1);
    const o2 = document.createElement('option'); o2.value = y; o2.textContent = y + '년';
    yrAnalysisSel.appendChild(o2);
  });
  calYearSel.addEventListener('change', () => {
    currentCalendarYear = parseInt(calYearSel.value);
    renderCalendar(currentCalendarYear);
  });
  yrAnalysisSel.addEventListener('change', () => {
    currentYearAnalysis = parseInt(yrAnalysisSel.value);
    renderYearAnalysis(currentYearAnalysis);
  });

  setPlayhead(tMin);
}
</script>
</body>
</html>
"""


class HTMLTimelineRenderer:
    """타임라인 + 통계 패널 + 캘린더 + 연도별 분석 self-contained HTML 생성."""

    @staticmethod
    def _safe_json(obj) -> str:
        # </script> 가 데이터에 들어가도 스크립트가 닫히지 않도록 보호
        return json.dumps(obj, ensure_ascii=False, default=str).replace("</", "<\\/")

    def render(self, events: list, style_summary: dict, output_path: str,
               temporal: dict | None = None, yearly: dict | None = None):
        timeline_json = self._safe_json({"events": events})
        summary_json  = self._safe_json(style_summary or {})
        temporal_json = self._safe_json(temporal or {})
        yearly_json   = self._safe_json(yearly or {})

        html = (HTML_TEMPLATE
                .replace("__TIMELINE_DATA__", timeline_json)
                .replace("__STYLE_SUMMARY__", summary_json)
                .replace("__TEMPORAL_DATA__", temporal_json)
                .replace("__YEARLY_DATA__", yearly_json))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        size_mb = len(html.encode("utf-8")) / 1024 / 1024
        print(f"  [Renderer] HTML 저장: {output_path}  ({size_mb:.2f} MB)")


# ══════════════════════════════════════════════════════════════
# 오케스트레이터
# ══════════════════════════════════════════════════════════════

class DataInfoOrchestrator:
    """person_0 라이프스타일 심층 분석 파이프라인 총괄."""

    def __init__(self, output_dir: str, original_dir: str):
        self.output_dir = output_dir
        self.original_dir = original_dir

        self.collector       = PhotoCollector()
        self.analyzer        = DeepPhotoAnalyzer(llm)
        self.profiler        = StyleProfiler()
        self.summarizer      = StyleSummarizer(llm)
        self.temporal_agg    = TemporalAggregator(self.profiler)
        self.yearly_summarizer = YearlySummarizer(llm)
        self.tl_builder      = TimelineBuilder()
        self.renderer        = HTMLTimelineRenderer()

        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def _serialize_photo(p: dict) -> dict:
        dt = p.get("datetime")
        return {
            **{k: v for k, v in p.items() if k != "datetime"},
            "datetime": dt.isoformat() if isinstance(dt, datetime) else str(dt),
        }

    def _sample_within_budget(self, photos: list) -> list:
        """MAX_DEEP_ANALYZE 초과 시 시간 분포 보존하며 균등 샘플링."""
        if len(photos) <= MAX_DEEP_ANALYZE:
            return photos
        step = max(1, len(photos) // MAX_DEEP_ANALYZE)
        sampled = photos[::step][:MAX_DEEP_ANALYZE]
        print(f"  비용 절감: {len(photos)}장 → {len(sampled)}장 균등 샘플링")
        return sampled

    def _load_cached_analyses(self, path: str) -> list | None:
        if not (REUSE_CACHED_ANALYSES and os.path.exists(path)):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                if isinstance(d.get("datetime"), str):
                    try:
                        d["datetime"] = datetime.fromisoformat(d["datetime"])
                    except ValueError:
                        d["datetime"] = datetime.now()
            print(f"  ✔ 기존 분석 결과 재사용: {path} ({len(data)}장)")
            return data
        except Exception as e:
            print(f"  ! 캐시 로드 실패, 새로 분석: {e}")
            return None

    def _retry_failed(self, analyzed: list, analyzed_path: str) -> list:
        """캐시 중 실패(error)했던 사진만 재분석해 병합 저장.
        MPO 인코더 개선 등으로 이전에 실패한 사진을 전체 재실행 없이 복구한다."""
        def is_failed(a: dict) -> bool:
            an = a.get("analysis")
            return (not isinstance(an, dict)) or (not an) or ("error" in an)

        failed = [a for a in analyzed
                  if is_failed(a) and a.get("path") and os.path.isfile(a["path"])]
        if not failed:
            return analyzed

        print(f"  [재분석] 실패했던 {len(failed)}장 재시도 (MPO 등 인코더 개선 반영)...")
        redone = self.analyzer.analyze_batch(failed, max_workers=ANALYZE_WORKERS)
        by_path = {r["path"]: r.get("analysis") for r in redone if r.get("path")}

        recovered = 0
        for a in analyzed:
            new = by_path.get(a.get("path"))
            if isinstance(new, dict) and new and "error" not in new:
                a["analysis"] = new
                recovered += 1

        if recovered:
            with open(analyzed_path, "w", encoding="utf-8") as f:
                json.dump([self._serialize_photo(a) for a in analyzed], f,
                          ensure_ascii=False, indent=2, default=str)
        print(f"  ✔ 재분석 완료: {recovered}/{len(failed)}장 복구")
        return analyzed

    def run(self) -> dict:
        print("\n" + "=" * 60)
        print("  person_0 라이프스타일 심층 분석 시작")
        print("=" * 60)

        # 1. 사진 수집 ─────────────────────────────────
        print(f"\n[1단계] EXIF 사진 수집 ({self.original_dir})")
        photos = self.collector.collect(self.original_dir)
        if not photos:
            print("  ✗ 수집된 사진 없음 (EXIF 없음 또는 원본 미발견)")
            return {"status": "no_photos"}

        photos = self._sample_within_budget(photos)
        with open(os.path.join(self.output_dir, "collected_photos.json"), "w", encoding="utf-8") as f:
            json.dump([self._serialize_photo(p) for p in photos], f, ensure_ascii=False, indent=2)

        # 2. 심층 분석 (캐시 우선) ─────────────────────
        analyzed_path = os.path.join(self.output_dir, "deep_analyses.json")
        cached = self._load_cached_analyses(analyzed_path)
        if cached is not None:
            analyzed = self._retry_failed(cached, analyzed_path)
        else:
            print(f"\n[2단계] LLM 심층 분석 ({len(photos)}장 / {ANALYZE_WORKERS} workers)")
            t0 = time.time()
            analyzed = self.analyzer.analyze_batch(photos, max_workers=ANALYZE_WORKERS)
            print(f"  ✔ {time.time()-t0:.1f}초 소요")
            with open(analyzed_path, "w", encoding="utf-8") as f:
                json.dump([self._serialize_photo(a) for a in analyzed], f,
                          ensure_ascii=False, indent=2, default=str)

        # 3. 통계 집계 ────────────────────────────────
        print(f"\n[3단계] 스타일 통계 집계")
        stats = self.profiler.aggregate(analyzed)
        with open(os.path.join(self.output_dir, "style_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

        # 4. LLM 페르소나 요약 ────────────────────────
        print(f"\n[4단계] 페르소나 요약 생성")
        summary = self.summarizer.summarize(stats)
        with open(os.path.join(self.output_dir, "style_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self._print_summary(stats, summary)

        # 5. 시간축 집계 (연도/월/일 + 캘린더) ──────────
        print(f"\n[5단계] 시간축 집계 (연/월/일)")
        temporal = self.temporal_agg.aggregate(analyzed)
        with open(os.path.join(self.output_dir, "temporal_stats.json"), "w", encoding="utf-8") as f:
            json.dump(temporal, f, ensure_ascii=False, indent=2, default=str)
        print(f"  ✔ 연도 {len(temporal['years'])}개 · 월 {len(temporal['month_stats'])}개 · 캘린더 활성일 "
              f"{sum(len(c) for c in temporal['calendar'].values())}일")

        # 6. 연도별 LLM 서사 (캐시 사용) ───────────────
        yearly_path = os.path.join(self.output_dir, "yearly_summaries.json")
        if REUSE_CACHED_ANALYSES and os.path.exists(yearly_path):
            with open(yearly_path, "r", encoding="utf-8") as f:
                yearly_summaries = json.load(f)
            print(f"\n[6단계] 연도별 요약 캐시 재사용 ({len(yearly_summaries)}개)")
        else:
            print(f"\n[6단계] 연도별 LLM 서사 생성 ({len(temporal['year_stats'])}개 연도)")
            yearly_summaries = self.yearly_summarizer.summarize(temporal["year_stats"])
            with open(yearly_path, "w", encoding="utf-8") as f:
                json.dump(yearly_summaries, f, ensure_ascii=False, indent=2)

        # 7. 타임라인 HTML ────────────────────────────
        print(f"\n[7단계] 타임라인 HTML 생성")
        events = self.tl_builder.build(analyzed, self.output_dir)
        html_path = os.path.join(self.output_dir, "person0_timeline.html")
        self.renderer.render(events, summary, html_path,
                             temporal=temporal, yearly=yearly_summaries)

        print("\n" + "=" * 60)
        print(f"  ✔ 완료")
        print(f"     결과 폴더 : {self.output_dir}")
        print(f"     HTML     : {html_path}")
        print("=" * 60)

        return {
            "status": "ok",
            "stats": stats,
            "summary": summary,
            "html_path": html_path,
            "photo_count": len(photos),
        }

    def _print_summary(self, stats: dict, summary: dict):
        if not summary or summary.get("error"):
            print("  ! 요약 생성 실패")
            return
        ls = stats.get("landscape", {})
        sc = stats.get("social", {})
        ac = stats.get("activity", {})

        print(f"\n  ── 풍경 점수 ──")
        print(f"     바다 {ls.get('sea_score',0)}  |  산 {ls.get('mountain_score',0)}  |  도시 {ls.get('city_score',0)}")
        print(f"     선호: {ls.get('preferred_landscape','?')}")
        print(f"\n  ── 사회성 ──")
        print(f"     혼자 {sc.get('solo_ratio',0)*100:.0f}%  |  함께 {sc.get('with_others_ratio',0)*100:.0f}%  "
              f"|  평균 {sc.get('avg_people_per_photo',0)}명/장")
        print(f"\n  ── 활동성 ──")
        print(f"     지수 {ac.get('activity_score_1to3','?')}/3")
        print(f"     실내/실외: {ac.get('indoor_outdoor', {})}")
        print(f"\n  ── LLM 요약 ──")
        for k in ("food_style", "landscape_style", "social_style", "activity_style"):
            print(f"     [{k}] {summary.get(k,'')[:120]}")
        tags = summary.get("highlight_tags") or []
        if tags:
            print(f"\n     태그: {' · '.join(tags)}")


if __name__ == "__main__":
    orchestrator = DataInfoOrchestrator(
        output_dir=OUTPUT_DIR,
        original_dir=ORIGINAL_PHOTOS_DIR,
    )
    orchestrator.run()
