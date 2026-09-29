"""
Integrated runner for the Lumina data pipeline.

The implementation is stored in the following readable source files:
- face_crop_embedding.py
- make_listofpeople.py
- data_agent.py
- data_info_agent.py

Local helper modules used by the face pipeline:
- net.py
- data_agent_myfulldata_crop.py

Run this file from this directory to execute the full pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_PHOTO_ROOT = os.getenv("LUMINA_PHOTO_ROOT", str(SCRIPT_DIR / "runtime" / "photos"))


def get_pipeline_paths(photo_root: str) -> dict:
    """파이프라인의 모든 경로를 한곳에서 정의한다.

    경로를 바꾸고 싶으면 **이 함수만** 수정하면 전체 단계에 반영된다.
    기본값은 photo_root 하위로 구성되지만, 각 값을 원하는 절대경로로
    직접 바꿔도 된다 (예: OUTPUT_DIR만 다른 드라이브로 빼고 싶을 때).
    """
    crop_dir = os.path.join(photo_root, "cropped_insight")
    cluster_dir = os.path.join(crop_dir, "adaface_results")

    return {
        # ── 공통 (입력 사진 루트 및 파생 폴더) ──
        "photo_root":  photo_root,
        "crop_dir":    crop_dir,     # 얼굴 크롭 저장 폴더
        "cluster_dir": cluster_dir,  # 클러스터(person_N) 결과 폴더

        # ── 1단계: face_crop_embedding ──
        "face_crop_embedding": {
            "SOURCE_DIR":    photo_root,   # 원본 사진 입력
            "CROP_SAVE_DIR": crop_dir,     # 크롭 저장
            "DB_PATH":       crop_dir,     # 임베딩 DB
            "OUTPUT_PATH":   cluster_dir,  # 클러스터 결과
        },

        # ── 3단계: data_agent ──
        "data_agent": {
            "CLUSTER_DIR":         cluster_dir,  # 입력: 클러스터 폴더
            "OUTPUT_DIR":          photo_root,   # 출력: person_N 분석 결과
            "ORIGINAL_PHOTOS_DIR": photo_root,   # 원본 사진 (EXIF 컨텍스트)
        },

        # ── 4단계: data_info_agent ──
        "data_info_agent": {
            "ORIGINAL_PHOTOS_DIR": photo_root,
            "OUTPUT_DIR":          os.path.join(photo_root, "person0_deep_profile"),
        },
    }

@dataclass(frozen=True)
class PipelineStep:
    name: str
    original_file: str
    description: str

PIPELINE_STEPS: tuple[PipelineStep, ...] = (
    PipelineStep("face_crop_embedding", "face_crop_embedding.py", "face crop + AdaFace embedding clustering"),
    PipelineStep("make_listofpeople", "make_listofpeople.py", "image-to-person CSV mapping"),
    PipelineStep("data_agent", "data_agent.py", "person relationship/profile analysis"),
    PipelineStep("data_info_agent", "data_info_agent.py", "person_0 lifestyle/timeline analysis"),
)

EMBEDDED_SOURCES = {'face_crop_embedding': 'face_crop_embedding.py', 'make_listofpeople': 'make_listofpeople.py', 'data_agent': 'data_agent.py', 'data_info_agent': 'data_info_agent.py'}

EMBEDDED_MODULES = {'net': 'net.py', 'data_agent_myfulldata_crop': 'data_agent_myfulldata_crop.py'}


def _decode(chunks: dict[str, str], name: str) -> str:
    return (SCRIPT_DIR / chunks[name]).read_text(encoding="utf-8")


def _install_embedded_module(module_name: str, virtual_file: str) -> None:
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    module.__file__ = virtual_file
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(_decode(EMBEDDED_MODULES, module_name), virtual_file, "exec")
        exec(code, module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _install_face_helpers() -> None:
    _install_embedded_module("net", str(SCRIPT_DIR / "net.py"))
    _install_embedded_module(
        "data_agent_myfulldata_crop",
        str(SCRIPT_DIR / "data_agent_myfulldata_crop.py"),
    )


def _exec_embedded(step: PipelineStep) -> dict:
    if step.name == "face_crop_embedding":
        _install_face_helpers()

    virtual_file = str(SCRIPT_DIR / step.original_file)
    namespace = {
        "__name__": f"__integrated_{step.name}__",
        "__file__": virtual_file,
        "__package__": None,
    }
    # data_agent / data_info_agent 은 외부 원본 .py 가 있으면 그것을 우선 사용한다.
    # (resume 멱등성 등 수정을 재빌드 없이 즉시 반영. 없으면 임베드본으로 폴백.)
    _OVERRIDABLE = {"data_agent", "data_info_agent"}
    _ext = SCRIPT_DIR / step.original_file
    if step.name in _OVERRIDABLE and _ext.is_file():
        _src = _ext.read_text(encoding="utf-8")
        print(f"[integrated] {step.name}: 외부 원본 사용 ({_ext})", flush=True)
    else:
        _src = _decode(EMBEDDED_SOURCES, step.name)
    code = compile(_src, virtual_file, "exec")
    exec(code, namespace)
    return namespace


def _cluster_dir(photo_root: str) -> str:
    return get_pipeline_paths(photo_root)["cluster_dir"]


def _crop_dir(photo_root: str) -> str:
    return get_pipeline_paths(photo_root)["crop_dir"]


def _write_image_people_mapping(base_path: str) -> None:
    import pandas as pd
    from collections import defaultdict

    image_people_map = defaultdict(list)
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"cluster result directory not found: {base_path}")

    person_dirs = sorted(
        d for d in os.listdir(base_path)
        if d.startswith("person_") and os.path.isdir(os.path.join(base_path, d))
    )
    for person_id in person_dirs:
        person_folder = os.path.join(base_path, person_id)
        for filename in os.listdir(person_folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            name_without_ext, ext = os.path.splitext(filename)
            if "_" in name_without_ext:
                clean_name = "_".join(name_without_ext.split("_")[:-1]) + ext
            else:
                clean_name = filename
            image_people_map[clean_name].append(person_id)

    rows = [
        {"image_name": image_name, "people": ",".join(people_list)}
        for image_name, people_list in sorted(image_people_map.items())
    ]
    output_csv = os.path.join(base_path, "image_people_mapping.csv")
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[make_listofpeople] {len(rows)} unique images found", flush=True)
    print(f"[make_listofpeople] CSV saved: {output_csv}", flush=True)


def _count_cluster_images(person_dir: str) -> int:
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    return sum(
        1
        for filename in os.listdir(person_dir)
        if filename.lower().endswith(valid_exts)
    )


def _force_largest_cluster_to_person_zero(cluster_dir: str) -> None:
    if not os.path.isdir(cluster_dir):
        raise FileNotFoundError(f"cluster result directory not found: {cluster_dir}")

    clusters = []
    for dirname in os.listdir(cluster_dir):
        if not dirname.startswith("person_"):
            continue
        suffix = dirname.removeprefix("person_")
        if not suffix.isdigit():
            continue
        path = os.path.join(cluster_dir, dirname)
        if not os.path.isdir(path):
            continue
        clusters.append((dirname, int(suffix), _count_cluster_images(path)))

    if not clusters:
        print("[person_0] No person_N cluster directories found.", flush=True)
        return

    largest_name, _, largest_count = sorted(
        clusters,
        key=lambda item: (-item[2], item[1]),
    )[0]

    if largest_name == "person_0":
        print(f"[person_0] Already largest cluster ({largest_count} images).", flush=True)
        return

    largest_path = os.path.join(cluster_dir, largest_name)
    zero_path = os.path.join(cluster_dir, "person_0")
    temp_path = os.path.join(cluster_dir, f"__person_0_swap_{int(time.time())}")
    while os.path.exists(temp_path):
        temp_path += "_"

    if os.path.exists(zero_path):
        zero_count = _count_cluster_images(zero_path)
        os.replace(zero_path, temp_path)
        os.replace(largest_path, zero_path)
        os.replace(temp_path, os.path.join(cluster_dir, largest_name))
        print(
            f"[person_0] Swapped {largest_name} ({largest_count} images) "
            f"with person_0 ({zero_count} images).",
            flush=True,
        )
    else:
        os.replace(largest_path, zero_path)
        print(
            f"[person_0] Renamed {largest_name} ({largest_count} images) to person_0.",
            flush=True,
        )


def _apply_common_paths(namespace: dict, step_name: str, photo_root: str) -> None:
    # get_pipeline_paths()에 정의된 단계별 경로를 그대로 임베드 모듈 네임스페이스에 주입.
    step_paths = get_pipeline_paths(photo_root).get(step_name, {})
    for key, value in step_paths.items():
        namespace[key] = value


def _run_step(step: PipelineStep, *, keep_going: bool, photo_root: str) -> bool:
    print("=" * 80, flush=True)
    print(f"[START] {step.name} - {step.description}", flush=True)
    started = time.time()
    try:
        if step.name == "make_listofpeople":
            _force_largest_cluster_to_person_zero(_cluster_dir(photo_root))
            _write_image_people_mapping(_cluster_dir(photo_root))
        else:
            namespace = _exec_embedded(step)
            _apply_common_paths(namespace, step.name, photo_root)

        if step.name == "face_crop_embedding":
            namespace["main"]()
            _force_largest_cluster_to_person_zero(_cluster_dir(photo_root))
        elif step.name == "make_listofpeople":
            pass
        elif step.name == "data_agent":
            _force_largest_cluster_to_person_zero(_cluster_dir(photo_root))
            orchestrator = namespace["PersonProfileOrchestrator"](output_dir=namespace["OUTPUT_DIR"])
            orchestrator.run(cluster_dir=namespace["CLUSTER_DIR"])
        elif step.name == "data_info_agent":
            orchestrator = namespace["DataInfoOrchestrator"](
                output_dir=namespace["OUTPUT_DIR"],
                original_dir=namespace["ORIGINAL_PHOTOS_DIR"],
            )
            orchestrator.run()
        else:
            raise ValueError(f"Unknown step: {step.name}")
    except Exception as exc:
        elapsed = time.time() - started
        print(f"[FAIL] {step.name}: {exc} ({elapsed:.1f}s)", flush=True)
        if keep_going:
            return True
        return False

    elapsed = time.time() - started
    print(f"[DONE] {step.name} ({elapsed:.1f}s)", flush=True)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the modular Lumina analysis pipeline.",
    )
    parser.add_argument(
        "--from-step",
        choices=[step.name for step in PIPELINE_STEPS],
        help="Start from a specific step instead of the beginning.",
    )
    parser.add_argument(
        "--only",
        choices=[step.name for step in PIPELINE_STEPS],
        help="Run only one step.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running later steps even if one step fails.",
    )
    parser.add_argument(
        "--photo-root",
        default=os.environ.get("INTEGRATED_PHOTO_ROOT", DEFAULT_PHOTO_ROOT),
        help=f"Photo root used by all steps. Default: {DEFAULT_PHOTO_ROOT}",
    )
    return parser.parse_args()


def select_steps(args: argparse.Namespace) -> tuple[PipelineStep, ...]:
    if args.only:
        return tuple(step for step in PIPELINE_STEPS if step.name == args.only)
    if args.from_step:
        names = [step.name for step in PIPELINE_STEPS]
        start_index = names.index(args.from_step)
        return PIPELINE_STEPS[start_index:]
    return PIPELINE_STEPS


def main() -> int:
    args = parse_args()
    steps = select_steps(args)

    os.environ.setdefault("PYTHONUTF8", "1")
    print("[INTEGRATED] Pipeline started", flush=True)
    print(f"[DIR] {SCRIPT_DIR}", flush=True)
    print(f"[PHOTO_ROOT] {args.photo_root}", flush=True)
    print("[ORDER] " + " -> ".join(step.name for step in steps), flush=True)

    started = time.time()
    for step in steps:
        if not _run_step(step, keep_going=args.keep_going, photo_root=args.photo_root):
            print("[INTEGRATED] Pipeline stopped because a step failed.", flush=True)
            return 1

    print("=" * 80, flush=True)
    print(f"[INTEGRATED] Embedded pipeline finished ({time.time() - started:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
