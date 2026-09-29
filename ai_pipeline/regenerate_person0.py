"""person_0의 profile.json / context_expansion.json만 표적 재생성.

이미 정상 생성된 level1_observations.json·level2_patterns.json을 재사용하므로
Level 1(전체 사진 재분석)을 다시 돌리지 않는다. LLM 호출은
 - 심층 프로필: 1~2회
 - 컨텍스트 확장: 근접 사진 분석 최대 10회 + 요약 1회
정도로 가볍다.

실행:  python regenerate_person0.py
"""
import os
import json

import data_agent as da


def main():
    person0_crop_dir = os.path.join(da.CLUSTER_DIR, "person_0")   # 크롭 얼굴 폴더
    out_dir = os.path.join(da.OUTPUT_DIR, "person_0")             # 결과 폴더
    obs_path = os.path.join(out_dir, "level1_observations.json")
    pat_path = os.path.join(out_dir, "level2_patterns.json")

    if not os.path.isfile(obs_path):
        raise SystemExit(f"level1_observations.json 없음: {obs_path} — 먼저 전체 분석 필요")

    observations = json.load(open(obs_path, encoding="utf-8"))
    patterns = (json.load(open(pat_path, encoding="utf-8"))
                if os.path.isfile(pat_path) else {})
    print(f"[재생성] person_0: 관찰 {len(observations)}장 재사용 (Level 1 재분석 안 함)")

    llm = da._get_llm()

    # ── profile.json 재생성 (skeleton 병합으로 old 양식 키 보장) ──
    profile = da.PersonZeroProfiler(llm).build_profile(observations, patterns)
    _p0_age = da._estimated_age_to_number(profile.get("estimated_age"))
    profile["age_timeline"] = da.build_age_timeline(observations, person0_age=_p0_age)
    if _p0_age is not None:
        profile["estimated_age_years"] = _p0_age
    with open(os.path.join(out_dir, "profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"  ✔ profile.json 재생성: 키 {list(profile.keys())}")

    # ── context_expansion.json 재생성 (근접 사진 재분석) ──
    ctx = da.PersonContextExpander(llm).expand(person0_crop_dir, da.ORIGINAL_PHOTOS_DIR)
    with open(os.path.join(out_dir, "context_expansion.json"), "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)
    print(f"  ✔ context_expansion.json 재생성: 키 {list(ctx.keys())}")

    print("[완료] person_0 두 파일이 old 양식으로 재생성되었습니다.")


if __name__ == "__main__":
    main()
