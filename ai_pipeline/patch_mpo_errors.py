"""MPO 인코딩 오류로 분석 실패한 관찰만 선별 재분석하는 타겟 패치.
- 오류(error) 있는 Level1 관찰만 재분석해 병합 (정상 1763장은 건드리지 않음)
- 영향받은 인물 + person_0 의 Level2/3·프로필·age_timeline 재계산
- knowledge_graph.json 노드/엣지 패치
원본 분석 전체 재실행 대비 거의 무료 수준의 비용으로 오류만 제거한다.
"""
import json
import os
import glob

import data_agent as da
from data_agent import (
    get_original_filepath, build_age_timeline, _estimated_age_to_number,
    invoke_with_retry, ORIGINAL_PHOTOS_DIR, OUTPUT_DIR,
)
from langchain_core.messages import HumanMessage

llm = da._get_llm()
observer = da.ObserverAgent(llm)
reasoner = da.ReasonerAgent(llm)
analyst = da.SocialAnalystAgent(llm)
p0_profiler = da.PersonZeroProfiler(llm)


def _read(p):
    return json.load(open(p, encoding="utf-8"))


def _write(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def reanalyze_errored(pid):
    """error 있는 관찰만 재분석해 같은 위치에 병합. (obs, 재분석수) 반환."""
    f = os.path.join(OUTPUT_DIR, pid, "level1_observations.json")
    obs = _read(f)
    changed = 0
    for i, o in enumerate(obs):
        if not o.get("error"):
            continue
        crop = o.get("image_path")
        orig = get_original_filepath(os.path.basename(crop or ""), ORIGINAL_PHOTOS_DIR)
        obs[i] = observer.analyze_single_image(crop, orig)
        changed += 1
    if changed:
        _write(f, obs)
    return obs, changed


def fallback_narrative(pid, patterns, relations):
    rel = relations.get("relationship_type", "알수없음")
    detail = relations.get("relationship_detail", "")
    score = relations.get("intimacy_score", "?")
    prompt = (
        f"다음 정보를 바탕으로 person_0(나)와 {pid}의 관계를 2문장으로 서술하세요. "
        f"관계: {rel} ({detail}), 친밀도: {score}/10. "
        f"대상 나이: {patterns.get('estimated_age','')}. 라이프스타일: {patterns.get('summary','')}"
    )
    try:
        return invoke_with_retry(llm, [HumanMessage(content=prompt)])
    except Exception:
        return f"{rel} 관계({detail})로, 친밀도 {score}/10 수준입니다."


def affected_person_ids():
    ids = []
    for d in glob.glob(os.path.join(OUTPUT_DIR, "person_*")):
        f = os.path.join(d, "level1_observations.json")
        if not os.path.isfile(f):
            continue
        try:
            obs = _read(f)
        except Exception:
            continue
        if any(o.get("error") for o in obs):
            ids.append(os.path.basename(d))
    return ids


def main():
    affected = affected_person_ids()
    print(f"영향 인물 {len(affected)}명: {', '.join(sorted(affected))}")

    # ── 1) person_0 먼저 (모든 Level3 추론의 기준) ──────────────────
    profile0 = age_timeline0 = None
    if "person_0" in affected:
        print("\n[person_0] 오류 관찰 재분석 + 프로필 재구축")
        obs0, ch = reanalyze_errored("person_0")
        patterns0 = reasoner.analyze_patterns("person_0", obs0)
        _write(os.path.join(OUTPUT_DIR, "person_0", "level2_patterns.json"), patterns0)
        profile0 = p0_profiler.build_profile(obs0, patterns0)
        _p0_age = _estimated_age_to_number(profile0.get("estimated_age"))
        age_timeline0 = build_age_timeline(obs0, person0_age=_p0_age)
        profile0["age_timeline"] = age_timeline0
        if _p0_age is not None:
            profile0["estimated_age_years"] = _p0_age
        _write(os.path.join(OUTPUT_DIR, "person_0", "profile.json"), profile0)
        print(f"  ✔ person_0 재분석 {ch}장 / 나이 {profile0.get('estimated_age')} / {profile0.get('gender')}")
        affected = [p for p in affected if p != "person_0"]
    else:
        pf = os.path.join(OUTPUT_DIR, "person_0", "profile.json")
        if os.path.isfile(pf):
            profile0 = _read(pf)
            age_timeline0 = profile0.get("age_timeline", {})

    # ── 2) 나머지 영향 인물: Level1 병합 + Level2/3 재계산 ──────────
    patched = {}  # pid -> (patterns, relations)
    for pid in sorted(affected):
        obs, ch = reanalyze_errored(pid)
        patterns = reasoner.analyze_patterns(pid, obs)
        _write(os.path.join(OUTPUT_DIR, pid, "level2_patterns.json"), patterns)
        relations = analyst.analyze_relationship(pid, obs, patterns, profile0, age_timeline0)
        relations["_photo_count"] = len(obs)
        if not relations.get("relationship_narrative"):
            relations["relationship_narrative"] = fallback_narrative(pid, patterns, relations)
        _write(os.path.join(OUTPUT_DIR, pid, "level3_relations.json"), relations)
        patched[pid] = (patterns, relations)
        print(f"  ✔ {pid}: 재분석 {ch}장 / {relations.get('relationship_type')} 친밀도 {relations.get('intimacy_score')}")

    # ── 3) knowledge_graph.json 패치 ──────────────────────────────
    kg_path = os.path.join(OUTPUT_DIR, "knowledge_graph.json")
    kg = _read(kg_path)
    nodes = {n["id"]: n for n in kg.get("nodes", [])}

    if profile0 is not None and "person_0" in nodes:
        nodes["person_0"].update({
            "estimated_age": profile0.get("estimated_age", nodes["person_0"].get("estimated_age")),
            "gender": profile0.get("gender", nodes["person_0"].get("gender")),
            "lifestyle_summary": profile0.get("profile_summary", nodes["person_0"].get("lifestyle_summary")),
        })

    for pid, (patterns, relations) in patched.items():
        if pid in nodes:
            nodes[pid].update({
                "estimated_age": patterns.get("estimated_age", ""),
                "gender": patterns.get("gender", ""),
                "lifestyle_summary": patterns.get("summary", ""),
                "relationship_type": relations.get("relationship_type", "알수없음"),
                "intimacy_score": relations.get("intimacy_score", "?"),
                "relationship_narrative": relations.get("relationship_narrative", ""),
                "age_based_reasoning": relations.get("age_based_reasoning", ""),
            })
        new_edge = relations.get("knowledge_graph_edge")
        if new_edge:
            replaced = False
            for i, e in enumerate(kg.get("edges", [])):
                if e.get("to") == pid:
                    kg["edges"][i] = new_edge
                    replaced = True
                    break
            if not replaced:
                kg.setdefault("edges", []).append(new_edge)

    _write(kg_path, kg)
    print(f"\n✔ knowledge_graph.json 패치 완료 (노드 {1 if profile0 else 0}+{len(patched)}건)")
    print("패치 종료")


if __name__ == "__main__":
    main()
