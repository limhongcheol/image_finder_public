# face_crop_worker.py
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

app = None

def init_worker():
    """자식 프로세스가 시작될 때 독립적으로 모델을 로드"""
    global app
    # GPU 메모리가 12GB(RTX 5070 등)로 넉넉하더라도, 프로세스당 VRAM을 쪼개 쓰므로
    # max_workers 개수에 따라 OOM(메모리 부족)이 날 수 있습니다.
    # 안전하게 초기화하기 위해 온디바이스 세션을 생성합니다.
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(1280, 1280))

def imread_korean(path):
    try:
        img_array = np.fromfile(path, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return None

def process_single_image(args):
    """ProcessPoolExecutor.map을 위해 인자를 튜플로 받습니다.

    반환: (상태문자열, 랜드마크dict)
      - 랜드마크dict: { save_name(크롭파일명) -> [[x,y]*5] } (크롭 좌표계 5점 랜드마크)
        [개선4] 단일 패스용 — 클러스터링이 재검출 없이 이 랜드마크로 바로 정렬한다.
    """
    filename, source_dir, save_dir = args
    global app

    file_path = os.path.join(source_dir, filename)
    try:
        img = imread_korean(file_path)
        if img is None:
            return (f"[Skip] 이미지를 읽을 수 없음: {filename}", {})

        faces = app.get(img)
        if not faces:
            return (None, {})

        img_pil = Image.open(file_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')

        width, height = img_pil.size

        landmarks = {}
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox
            w, h = x2 - x1, y2 - y1
            margin_w, margin_h = w * 0.25, h * 0.25

            left = max(0, int(x1 - margin_w))
            top = max(0, int(y1 - margin_h))
            right = min(width, int(x2 + margin_w))
            bottom = min(height, int(y2 + margin_h))

            cropped_img = img_pil.crop((left, top, right, bottom))
            save_name = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            cropped_img.save(os.path.join(save_dir, save_name), quality=95)

            # [개선4] 크롭 좌표계 기준 5점 랜드마크(눈/코/입) 저장 — 풀이미지 좌표에서 크롭 원점 차감
            try:
                kps = face.kps  # (5,2) 풀이미지 좌표
                landmarks[save_name] = [[float(px - left), float(py - top)] for px, py in kps]
            except Exception:
                pass

        return (f"[성공] {filename}: {len(faces)}명 검출", landmarks)
    except Exception as e:
        return (f"[Error] {filename}: {e}", {})
