# face_crop_embedding.py
# data_agent_myfulldata.ipynb 의 두 셀(① 얼굴 크롭, ② 임베딩 + 클러스터링)을 그대로 옮긴 스크립트.
# - 모든 경로/설정값은 아래 [설정] 블록으로 분리.
# - "동일인 판별을 위한 임베딩"만 InsightFace -> AdaFace(ir_101) 로 교체.
#   (얼굴 검출/정렬은 기존과 동일하게 InsightFace 사용)
# - 군집화는 Agglomerative(cosine + average linkage, 정밀도 우선)로 교체.
#   (cluster_benchmark.py 의 stratified CV 결과 기반)
# - [개선] 임베딩 단계 입력 품질 향상 (ai_server_experiment.py 실험 근거):
#     (1) 클러스터링용 얼굴 선택을 faces[0] -> '가장 큰 얼굴'로 변경
#     (2) 저품질 얼굴(낮은 검출신뢰도/소형/흐림) 품질 필터로 군집에서 제외
#     (3) ★검출 전 여백 패딩 -- 타이트 크롭 재검출 실패(약 77%)를 대폭 회수.
#         실측(work 크롭 1500장): 검출 성공률 22.7% -> 70.5% (약 3배).
#         => 그동안 클러스터링에서 통째로 빠지던 얼굴들이 대량 합류.
#     (4) ★★단일 패스 -- 크롭 단계의 det-1280 랜드마크를 저장(_face_landmarks.json)해
#         클러스터링에서 재검출 없이 직접 정렬. 재검출 실패 누락 0, 39px 재검출보다
#         정렬 정확도↑, 속도 2배. 랜드마크 없는 크롭만 (3)의 패딩 재검출로 폴백.
#   ※ 검출/임베딩 모델(InsightFace+AdaFace)은 그대로 — 신모델 교체는 저화질에서
#     오히려 성능이 떨어짐을 실험으로 확인(AdaFace 가 저화질 최강). 입력 품질만 개선.
#
# ※ AdaFace 사전학습 모델은 입력을 BGR 채널로 받는다 (InsightFace 의 RGB 와 다름).
#   InsightFace 의 norm_crop 정렬 결과(cv2, BGR)를 채널 반전 없이 그대로 정규화해 넣는다.

# =====================================================================
# [설정] 경로 / 파라미터 (가장 앞)
# =====================================================================
import os

# ---- 공통 경로 ----
SOURCE_DIR = os.getenv("LUMINA_PHOTO_ROOT", os.path.join(os.path.dirname(__file__), "runtime", "photos"))
CROP_SAVE_DIR = os.path.join(SOURCE_DIR, "cropped_insight")    # ① 크롭 결과 저장 폴더
DB_PATH = CROP_SAVE_DIR                                        # ② 임베딩 대상 폴더(= 크롭 결과)
OUTPUT_PATH = os.path.join(DB_PATH, "adaface_results")         # ② 클러스터링 분류 결과 폴더

# ---- 실행 범위 ----
RUN_CROP = True        # ① 크롭 단계 실행 여부
RUN_CLUSTER = True     # ② 임베딩+클러스터링 단계 실행 여부

# ---- ① 크롭 단계 ----
MAX_WORKERS = 2        # RTX 5070 12GB 기준 2~3 권장 (det_size 1280 모델 다중 로드 시 VRAM 주의)
CROP_VALID_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.PNG')

# ---- ② 임베딩/클러스터링 단계 ----
CLUSTER_VALID_EXT = ('.jpg', '.jpeg', '.png')
DET_SIZE = (640, 640)          # 클러스터링용 InsightFace 검출 크기(노트북과 동일)
# 클러스터링: Agglomerative (cosine + average linkage) — 정밀도 우선.
# stratified CV(86명/816장)에서 정밀도 0.93으로 최고. distance_threshold 는 cosine 거리.
CLUSTER_DISTANCE_THRESHOLD = 0.68   # 작을수록 엄격(정밀도↑, 같은 사람 확신해야 묶음). 권장 0.60~0.75
CLUSTER_LINKAGE = "average"         # average / complete / single
MIN_CLUSTER_SIZE = 2                # 이 미만 크기의 그룹은 unknown 으로 (1장짜리=미분류)

# ---- 품질 필터 (저품질 얼굴 제외 → 군집 오염 방지) ----
# ai_server_experiment.py 의 저화질 실험에서 확인된 두 가지:
#   (a) AdaFace 는 '중간 정도' 저화질은 우아하게 처리한다(과도한 필터링은 회수율만 손해).
#   (b) 따라서 필터는 '명백히 사용 불가한 얼굴'(배경의 초소형/심한 흐림/오검출)만
#       보수적으로 제거하도록 임계값을 낮게 잡는다. 끄려면 QUALITY_FILTER=False.
QUALITY_FILTER = True
MIN_DET_SCORE  = 0.45   # InsightFace 검출 신뢰도 하한 (0~1) — 오검출 제거
MIN_FACE_PX    = 36     # 재검출된 얼굴의 최소 변 길이(px) — 배경의 초소형 얼굴만 제외
MIN_BLUR_VAR   = 8.0    # 112 정렬 얼굴 Laplacian 분산 하한 — 거의 평면(심한 흐림)만 제외

# ---- [개선3] 검출용 여백 패딩 (가장 큰 효과) ----
# 크롭은 얼굴이 프레임을 꽉 채워(실측: 최소변 중앙값 ~39px) 검출기가 "맥락 없는
# 과대 얼굴"로 보고 자주 실패한다. 재검출 전 reflect 여백을 덧대면 검출 성공률이
# 22.7% -> 70.5% 로 급증(실측: work 크롭 1500장) -> 누락되던 얼굴 대량 회수.
DETECT_PAD_RATIO = 0.6  # 긴 변 대비 사방 여백 비율 (0 이면 패딩 끔)

# ---- AdaFace 설정 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
ADAFACE_CKPT = os.getenv("ADAFACE_CKPT", os.path.join(_HERE, "weights", "adaface_ir101_webface12m.ckpt"))
ADAFACE_ARCH = "ir_101"        # 체크포인트와 일치해야 함
DEVICE = "cuda"                # 'cuda' 또는 'cpu' (cuda 불가 시 자동 cpu 폴백)

# ① 크롭 워커 모듈 위치 (insightface 기반, 20260511 폴더). import 가능하도록 경로 추가.
import sys
_CROP_WORKER_DIR = _HERE
if _CROP_WORKER_DIR not in sys.path:
    sys.path.insert(0, _CROP_WORKER_DIR)
# =====================================================================

import time
import json
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2
from tqdm import tqdm

# ① 크롭 워커 (InsightFace 검출 기반) — 기존 모듈 재사용
from data_agent_myfulldata_crop import process_single_image, init_worker


# =====================================================================
# ① 얼굴 크롭 (멀티프로세싱) — 노트북 셀 1
# =====================================================================
def main_crop():
    os.makedirs(CROP_SAVE_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(CROP_VALID_EXT)]

    if not image_files:
        print("이미지 파일이 없습니다.")
        return

    total_files = len(image_files)
    print(f"InsightFace 멀티프로세싱 분석 시작 (대상: {total_files}장)...")

    success_count = 0
    skip_count = 0
    error_count = 0
    all_landmarks = {}   # [개선4] save_name -> 크롭 좌표계 5점 랜드마크 (단일 패스용)

    # 함수에 전달할 인자 묶기 (filename, source_dir, save_dir)
    tasks = [(f, SOURCE_DIR, CROP_SAVE_DIR) for f in image_files]

    # Windows 환경 대응을 위해 외부 module 함수와 initializer 지정
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker) as executor:
        # submit 대신 순서대로 반환받는 map을 사용하면 tqdm 연동이 더 깔끔합니다.
        results = executor.map(process_single_image, tasks)

        with tqdm(total=total_files, desc="얼굴 크롭 진행률") as pbar:
            for idx, result in enumerate(results, 1):
                # 워커는 (상태문자열, 랜드마크dict) 튜플을 반환 (구버전 문자열도 호환)
                if isinstance(result, tuple):
                    status, lms = result
                    if lms:
                        all_landmarks.update(lms)
                else:
                    status = result
                if status:
                    if "[성공]" in status:
                        success_count += 1
                    elif "[Skip]" in status:
                        skip_count += 1
                    elif "[Error]" in status:
                        error_count += 1
                        tqdm.write(status)

                pbar.update(1)

                if idx % 100 == 0:
                    tqdm.write(f"[{idx}/{total_files}] 처리 완료 - (성공: {success_count}, 스킵: {skip_count}, 에러: {error_count})")

    # [개선4] 크롭 단계 랜드마크 저장 -> run_clustering 이 재검출 없이 직접 정렬
    lm_path = os.path.join(CROP_SAVE_DIR, "_face_landmarks.json")
    try:
        with open(lm_path, "w", encoding="utf-8") as fp:
            json.dump(all_landmarks, fp)
        print(f"랜드마크 저장: {len(all_landmarks)}개 -> {lm_path}")
    except Exception as e:
        print(f"[WARN] 랜드마크 저장 실패(폴백 재검출 사용): {e}")

    print(f"\n최종 결과 - 성공: {success_count}장, 스킵: {skip_count}장, 에러: {error_count}장")


# =====================================================================
# AdaFace 임베딩 유틸 (InsightFace -> AdaFace 교체 부분)
# =====================================================================
def _load_adaface(arch, ckpt_path, device):
    """AdaFace 백본을 빌드하고 체크포인트(state_dict의 'model.' 접두사)를 로드."""
    import torch
    import net  # 같은 폴더의 vendored AdaFace 아키텍처

    model = net.build_model(arch)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    statedict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # PyTorch-Lightning 체크포인트: 백본 가중치만 추출 (head.* 는 학습용이라 제외)
    model_statedict = {k[len("model."):]: v for k, v in statedict.items() if k.startswith("model.")}
    model.load_state_dict(model_statedict)
    model.eval()
    model.to(device)
    return model


def _pad_for_detect(img_bgr, ratio=None):
    """타이트 크롭에 reflect 여백을 덧대 검출 성공률을 높인다.
    검출된 랜드마크는 패딩된 좌표계 기준이므로, 정렬(norm_crop)도 패딩 이미지에서 한다."""
    import cv2 as _cv2
    r = DETECT_PAD_RATIO if ratio is None else ratio
    if not r or r <= 0:
        return img_bgr
    m = int(max(img_bgr.shape[:2]) * r)
    return _cv2.copyMakeBorder(img_bgr, m, m, m, m, _cv2.BORDER_REFLECT)


def _adaface_preprocess(aligned_bgr):
    """norm_crop 결과(112x112, BGR uint8) -> AdaFace 입력 텐서.

    AdaFace 는 BGR 입력을 기대한다. 정렬 결과가 이미 BGR(cv2) 이므로 채널 반전 없이
    ((x/255)-0.5)/0.5 정규화만 적용한다.
    """
    import torch
    img = ((aligned_bgr.astype(np.float32) / 255.0) - 0.5) / 0.5
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


# =====================================================================
# ② 임베딩 + 클러스터링 (동일인 판별) — 노트북 셀 2
#    검출/정렬: InsightFace 유지 / 임베딩: AdaFace 로 교체
# =====================================================================
def run_clustering():
    import torch
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    from sklearn.cluster import AgglomerativeClustering
    from collections import Counter

    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    device = DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"

    # 1-a. InsightFace 검출/정렬 모델 로드 (Buffalo_L)
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    # 1-b. AdaFace 임베딩 모델 로드 (InsightFace 임베딩 대체)
    adaface = _load_adaface(ADAFACE_ARCH, ADAFACE_CKPT, device)

    image_files = [f for f in os.listdir(DB_PATH) if f.lower().endswith(CLUSTER_VALID_EXT)]

    # [개선4] 단일 패스: 크롭 단계가 저장한 랜드마크(_face_landmarks.json)를 우선 사용.
    #   크롭 시점의 det-1280 풀이미지 랜드마크는 39px 타이트 크롭 재검출보다 정확하고,
    #   재검출 실패로 인한 누락이 0 이 된다. 랜드마크가 없는 크롭만 패딩 재검출로 폴백.
    landmarks_map = {}
    lm_path = os.path.join(DB_PATH, "_face_landmarks.json")
    if os.path.exists(lm_path):
        try:
            with open(lm_path, "r", encoding="utf-8") as fp:
                landmarks_map = json.load(fp)
            print(f"   랜드마크 로드: {len(landmarks_map)}개 (단일 패스 정렬)")
        except Exception as e:
            print(f"   랜드마크 로드 실패(폴백 재검출 사용): {e}")

    embeddings = []
    valid_files = []
    n_lowq = 0       # 품질 필터로 제외된 얼굴 수
    n_landmark = 0   # 저장 랜드마크로 직접 정렬한 수
    n_redetect = 0   # 폴백 재검출로 정렬한 수
    n_nodet = 0      # 폴백 재검출도 실패한 수 (랜드마크 없는 크롭만 해당)

    print(f"1. AdaFace 특징 추출 (대상: {len(image_files)}장)...")

    for f in tqdm(image_files):
        try:
            file_path = os.path.join(DB_PATH, f)
            # Windows 한글/유니코드 경로 대응: cv2.imread 대신 np.fromfile + imdecode
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
            if img is None:
                continue

            kps = landmarks_map.get(f)
            if kps is not None:
                # 단일 패스: 저장된 랜드마크로 직접 정렬 (재검출 생략 -> 누락 0, 정렬 정확↑)
                aligned = face_align.norm_crop(
                    img, landmark=np.array(kps, dtype=np.float32), image_size=112)
                n_landmark += 1
                # 랜드마크 경로: 이미 det-1280 검출을 통과한 실제 얼굴이므로
                # 크기/신뢰도 필터는 생략하고 흐림(blur)만 보수적으로 본다.
                drop = False
                if QUALITY_FILTER:
                    blur = cv2.Laplacian(
                        cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    drop = blur < MIN_BLUR_VAR
            else:
                # 폴백: [개선3] 검출 전 여백 패딩 후 재검출 + [개선1] 가장 큰 얼굴 선택.
                img_det = _pad_for_detect(img)
                faces = app.get(img_det)
                if len(faces) == 0:
                    n_nodet += 1
                    continue
                face = max(faces, key=lambda fc: (fc.bbox[2] - fc.bbox[0]) *
                           (fc.bbox[3] - fc.bbox[1]))
                aligned = face_align.norm_crop(img_det, landmark=face.kps, image_size=112)
                n_redetect += 1
                drop = False
                if QUALITY_FILTER:
                    fw = float(face.bbox[2] - face.bbox[0])
                    fh = float(face.bbox[3] - face.bbox[1])
                    det_score = float(getattr(face, "det_score", 1.0))
                    blur = cv2.Laplacian(
                        cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    drop = (det_score < MIN_DET_SCORE or min(fw, fh) < MIN_FACE_PX
                            or blur < MIN_BLUR_VAR)

            if drop:
                n_lowq += 1
                continue  # 저품질 -> 군집에서 제외 (어느 person 폴더에도 넣지 않음)

            # AdaFace 임베딩 (BGR 입력) -> 512차원 L2 정규화 벡터
            with torch.no_grad():
                feat, _ = adaface(_adaface_preprocess(aligned).to(device))
            embeddings.append(feat.cpu().numpy()[0])
            valid_files.append(f)
        except Exception as e:
            print(f"에러: {f} -> {e}")

    print(f"   정렬 경로 - 랜드마크 직접: {n_landmark}, 폴백 재검출: {n_redetect}, "
          f"재검출 실패: {n_nodet}")
    if QUALITY_FILTER:
        print(f"   품질 필터 제외: {n_lowq}장 "
              f"(det<{MIN_DET_SCORE} | 변<{MIN_FACE_PX}px | blur<{MIN_BLUR_VAR})")

    if not embeddings:
        print("추출된 임베딩이 없습니다.")
        return

    embeddings = np.array(embeddings)

    print("2. Agglomerative 클러스터링 (cosine + average linkage)...")

    # [설정 가이드]
    # distance_threshold: cosine 거리. 작을수록 엄격(정밀도↑). 벤치마크 권장 ~0.68.
    # Agglomerative 는 노이즈(-1) 개념이 없으므로, MIN_CLUSTER_SIZE 미만 그룹은 unknown 으로 보낸다.
    if len(embeddings) < 2:
        labels = np.zeros(len(embeddings), dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage=CLUSTER_LINKAGE,
            distance_threshold=CLUSTER_DISTANCE_THRESHOLD,
        ).fit_predict(embeddings)

    # 그룹 크기 계산 -> 작은 그룹(예: 1장짜리)은 unknown 처리 (DBSCAN min_samples=2 의미 보존)
    sizes = Counter(labels.tolist())

    # 3. 결과 정리 (person_id 는 0부터 연속 부여)
    person_id = {}
    next_pid = 0
    for i, label in enumerate(labels):
        if sizes[label] < MIN_CLUSTER_SIZE:
            target_dir = os.path.join(OUTPUT_PATH, "unknown")
        else:
            if label not in person_id:
                person_id[label] = next_pid
                next_pid += 1
            target_dir = os.path.join(OUTPUT_PATH, f"person_{person_id[label]}")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        shutil.copy(os.path.join(DB_PATH, valid_files[i]), target_dir)

    n_clusters = next_pid
    print(f"최종 분류: {n_clusters}명의 그룹 발견 (unknown 제외).")


# =====================================================================
# 실행: ① 크롭 -> ② 임베딩+클러스터링 순차 실행
# =====================================================================
def main():
    if RUN_CROP:
        main_crop()
    if RUN_CLUSTER:
        run_clustering()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"전체 작업 완료! 소요 시간: {time.time() - start_time:.2f}초")
