# cluster_benchmark.py
# 라벨된 얼굴 데이터로 "동일인 군집화" 알고리즘들을 stratified 교차검증으로 비교/튜닝한다.
#
# 비교 대상:
#   - DBSCAN            (밀도 기반, 기존 방식)
#   - Agglomerative     (계층적, cosine + average linkage)
#   - HDBSCAN           (DBSCAN 개선판, sklearn 내장)
#   - Infomap           (kNN 그래프 + 커뮤니티 검출, igraph / 무학습)
#   - PairwiseLink      (소규모 데이터 학습용: 엣지 단위 same-person 분류기 + 연결요소)
#
# 평가지표: BCubed P/R/F (얼굴 군집 표준), Pairwise F, NMI, ARI, 군집 수
# 검증: RepeatedStratifiedKFold (인물별 층화) — 임계값/모델은 train fold 에서만 선택,
#       val fold 로만 성능 측정.
#
# 임베딩은 face_crop_embedding.py 의 AdaFace 로직을 재사용한다.

import os
import sys
import json
import time
import argparse

import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# =====================================================================
# [설정]
# =====================================================================
LABELED_ROOT = os.getenv("LUMINA_LABELED_FACES", os.path.join(_HERE, "runtime", "labeled_faces"))
EMB_CACHE = os.path.join(_HERE, "labeled_embeddings.npz")          # 임베딩 캐시

DET_SIZE = (640, 640)        # InsightFace 검출 크기
KNN_K = 20                   # 그래프/엣지 피처용 최근접 이웃 수
N_SPLITS = 2                 # 층화 K-fold 분할 수 (최소 클래스 표본수=2 이므로 최대 2)
N_REPEATS = 5                # 반복 횟수 (안정적인 평균±표준편차)
RANDOM_STATE = 42
SELECT_METRIC = "bcubed_f"   # train fold 에서 하이퍼파라미터 선택 기준

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")

# 하이퍼파라미터 스윕 범위 (cosine distance = 1 - cos sim)
EPS_GRID = np.round(np.arange(0.15, 0.85, 0.025), 4)        # DBSCAN
AGG_GRID = np.round(np.arange(0.20, 1.00, 0.025), 4)        # Agglomerative distance_threshold
HDB_MCS_GRID = [2, 3, 4, 5, 6]                              # HDBSCAN min_cluster_size
TAU_GRID = np.round(np.arange(0.20, 0.75, 0.025), 4)        # Infomap 최소 유사도
PROB_GRID = np.round(np.arange(0.50, 0.99, 0.02), 4)        # PairwiseLink 링크 확률 임계값
# =====================================================================


# ---------------------------------------------------------------------
# 1. 임베딩 추출 (AdaFace, 캐시)
# ---------------------------------------------------------------------
def build_dataset(force=False):
    if (not force) and os.path.exists(EMB_CACHE):
        d = np.load(EMB_CACHE, allow_pickle=True)
        X, y = d["X"], d["y"]
        names = d["names"]
        print(f"[cache] 임베딩 로드: X={X.shape}, 인물={len(set(y.tolist()))}명")
        return X.astype(np.float32), y, names

    import cv2
    import torch
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    import face_crop_embedding as fce

    # 라벨 수집: 하위 폴더명 = 인물 라벨
    samples = []  # (path, label)
    for person in sorted(os.listdir(LABELED_ROOT)):
        pdir = os.path.join(LABELED_ROOT, person)
        if not os.path.isdir(pdir):
            continue
        for f in os.listdir(pdir):
            if f.lower().endswith(VALID_EXT):
                samples.append((os.path.join(pdir, f), person))

    if not samples:
        raise SystemExit(f"라벨 이미지가 없습니다: {LABELED_ROOT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    adaface = fce._load_adaface(fce.ADAFACE_ARCH, fce.ADAFACE_CKPT, device)

    def imread_korean(p):
        # Windows 한글/유니코드 경로 대응: cv2.imread 대신 np.fromfile + imdecode
        try:
            return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None

    embs, labels = [], []
    n_noface = 0
    n_read_fail = 0
    print(f"AdaFace 임베딩 추출 (대상: {len(samples)}장)...")
    for path, label in tqdm(samples):
        img = imread_korean(path)
        if img is None:
            n_read_fail += 1
            continue
        faces = app.get(img)
        if not faces:
            n_noface += 1
            continue
        # 가장 큰 얼굴 선택 (라벨 크롭은 보통 한 명이지만 안전하게)
        face = max(faces, key=lambda fc: (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]))
        aligned = face_align.norm_crop(img, landmark=face.kps, image_size=112)
        with torch.no_grad():
            feat, _ = adaface(fce._adaface_preprocess(aligned).to(device))
        embs.append(feat.cpu().numpy()[0])
        labels.append(label)

    X = np.asarray(embs, dtype=np.float32)
    # L2 정규화 보장 (AdaFace 출력은 이미 정규화돼 있음)
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    names = np.asarray(labels)
    print(f"추출 완료: {X.shape[0]}장 (읽기실패 {n_read_fail}장, 검출실패 {n_noface}장 제외), "
          f"인물 {len(set(labels))}명")
    np.savez(EMB_CACHE, X=X, y=names, names=names)
    return X, names, names


# ---------------------------------------------------------------------
# 2. 평가지표
# ---------------------------------------------------------------------
def _relabel_noise(pred):
    """DBSCAN/HDBSCAN 의 노이즈(-1)를 각각 고유 싱글톤 군집으로 변환 (표준 관례)."""
    pred = np.asarray(pred).copy()
    next_id = (pred.max() + 1) if pred.size and pred.max() >= 0 else 0
    out = pred.copy()
    for i in np.where(pred == -1)[0]:
        out[i] = next_id
        next_id += 1
    return out


def bcubed(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = _relabel_noise(y_pred)
    n = len(y_true)
    # 군집/클래스별 인덱스
    from collections import defaultdict
    pred_idx = defaultdict(list)
    for i, c in enumerate(y_pred):
        pred_idx[c].append(i)
    true_idx = defaultdict(list)
    for i, c in enumerate(y_true):
        true_idx[c].append(i)

    prec = 0.0
    rec = 0.0
    for i in range(n):
        same_pred = pred_idx[y_pred[i]]
        same_true = true_idx[y_true[i]]
        # 같은 예측군집 내에서 같은 실제라벨 개수
        correct = sum(1 for j in same_pred if y_true[j] == y_true[i])
        prec += correct / len(same_pred)
        rec += correct / len(same_true)
    prec /= n
    rec /= n
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f


def pairwise_f(y_true, y_pred):
    from sklearn.metrics.cluster import pair_confusion_matrix
    y_pred = _relabel_noise(y_pred)
    (tn, fp), (fn, tp) = pair_confusion_matrix(y_true, y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f


def all_metrics(y_true, y_pred):
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    bp, br, bf = bcubed(y_true, y_pred)
    pp, pr, pf = pairwise_f(y_true, y_pred)
    pred_rl = _relabel_noise(y_pred)
    n_clusters = len(set(pred_rl.tolist()))
    return {
        "bcubed_p": bp, "bcubed_r": br, "bcubed_f": bf,
        "pair_f": pf,
        "nmi": normalized_mutual_info_score(y_true, pred_rl),
        "ari": adjusted_rand_score(y_true, pred_rl),
        "n_clusters": n_clusters,
    }


# ---------------------------------------------------------------------
# 3. 그래프/이웃 유틸
# ---------------------------------------------------------------------
def cosine_dist_matrix(X):
    S = X @ X.T
    np.clip(S, -1.0, 1.0, out=S)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D


def knn(X, k):
    """각 점의 상위 k 이웃 (자기 제외). 반환: idx(n,k), sim(n,k)."""
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    # 첫 열은 자기 자신
    return idx[:, 1:], (1.0 - dist[:, 1:])


# ---------------------------------------------------------------------
# 4. 클러스터러들 (param 1개로 군집화 -> labels)
# ---------------------------------------------------------------------
def cluster_dbscan(X, D, eps):
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=eps, min_samples=2, metric="precomputed").fit_predict(D)


def cluster_agg(X, D, thr):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=thr,
    ).fit_predict(D)


def cluster_hdbscan(X, D, mcs):
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=int(mcs), metric="precomputed").fit_predict(D)


def cluster_infomap(X, D, tau, k=KNN_K):
    import igraph
    n = X.shape[0]
    idx, sim = knn(X, k)
    edges, weights = [], []
    seen = set()
    for i in range(n):
        for jj, s in zip(idx[i], sim[i]):
            if s < tau:
                continue
            a, b = (i, int(jj)) if i < jj else (int(jj), i)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append((a, b))
            weights.append(float(s))
    g = igraph.Graph(n=n, edges=edges)
    if weights:
        g.es["weight"] = weights
        comm = g.community_infomap(edge_weights="weight")
    else:
        comm = g.community_infomap()
    return np.asarray(comm.membership)


# ---------------------------------------------------------------------
# 5. PairwiseLink: 소규모 데이터 학습 모델
#    엣지(후보쌍) 단위 same-person 분류기 + 확률 임계값 + 연결요소
# ---------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _pair_features(X, k=KNN_K):
    """후보 엣지와 인물 비의존적 피처 생성. 반환: edges(m,2), feats(m,F), nbr_sets."""
    n = X.shape[0]
    idx, sim = knn(X, k)
    kk = idx.shape[1]
    nbr_set = [set(idx[i].tolist()) for i in range(n)]
    dens = sim.mean(axis=1)  # 자기 이웃 평균 유사도(국소 밀도)
    # i->j 순위/유사도 룩업
    rank_lut = [dict() for _ in range(n)]
    sim_lut = [dict() for _ in range(n)]
    for i in range(n):
        for r, (j, s) in enumerate(zip(idx[i], sim[i])):
            rank_lut[i][int(j)] = r
            sim_lut[i][int(j)] = float(s)

    edges, feats = [], []
    seen = set()
    for i in range(n):
        for r, j in enumerate(idx[i]):
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            s_ij = sim_lut[i][j]
            rank_i = rank_lut[i].get(j, kk) / kk
            rank_j = rank_lut[j].get(i, kk) / kk
            mutual = 1.0 if (i in nbr_set[j] and j in nbr_set[i]) else 0.0
            inter = len(nbr_set[i] & nbr_set[j])
            union = len(nbr_set[i] | nbr_set[j])
            jacc = inter / union if union else 0.0
            shared = inter / kk
            feats.append([
                s_ij, rank_i, rank_j, mutual, jacc, shared,
                dens[i], dens[j], s_ij - dens[i], s_ij - dens[j],
            ])
            edges.append((a, b))
    return np.asarray(edges), np.asarray(feats, dtype=np.float32)


def _components_from_edges(n, edges, keep_mask):
    uf = _UnionFind(n)
    for (a, b), keep in zip(edges, keep_mask):
        if keep:
            uf.union(int(a), int(b))
    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


def train_pairwise(X_train, y_train, k=KNN_K):
    from sklearn.ensemble import HistGradientBoostingClassifier
    edges, feats = _pair_features(X_train, k)
    if len(edges) == 0:
        return None, 0.5
    lbl = np.array([1 if y_train[a] == y_train[b] else 0 for a, b in edges])
    if lbl.sum() == 0 or lbl.sum() == len(lbl):
        return None, 0.5
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, max_iter=200,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    clf.fit(feats, lbl)
    # train 에서 확률 임계값 선택 (BCubed-F 최대)
    prob = clf.predict_proba(feats)[:, 1]
    best_p, best_f = 0.5, -1.0
    for p in PROB_GRID:
        labels = _components_from_edges(len(X_train), edges, prob >= p)
        f = bcubed(y_train, labels)[2]
        if f > best_f:
            best_f, best_p = f, p
    return clf, best_p


def cluster_pairwise(X, clf, prob_thr, k=KNN_K):
    edges, feats = _pair_features(X, k)
    if clf is None or len(edges) == 0:
        return np.arange(len(X))  # 전부 싱글톤
    prob = clf.predict_proba(feats)[:, 1]
    return _components_from_edges(len(X), edges, prob >= prob_thr)


# ---------------------------------------------------------------------
# 6. 하이퍼파라미터 튜닝 (train fold)
# ---------------------------------------------------------------------
def _best_param(cluster_fn, X, D, y, grid):
    best_p, best_score = grid[0], -1.0
    for p in grid:
        labels = cluster_fn(X, D, p)
        score = bcubed(y, labels)[2]
        if score > best_score:
            best_score, best_p = score, p
    return best_p


# ---------------------------------------------------------------------
# 7. 교차검증 실행
# ---------------------------------------------------------------------
def run_cv(X, y_names, methods):
    from sklearn.model_selection import RepeatedStratifiedKFold

    # 라벨 -> 정수 코드
    classes, y = np.unique(y_names, return_inverse=True)
    min_class = np.bincount(y).min()
    n_splits = min(N_SPLITS, min_class)
    if n_splits < 2:
        raise SystemExit("층화 검증 불가: 표본 1개짜리 인물이 있습니다.")
    print(f"\n층화 교차검증: n_splits={n_splits}, n_repeats={N_REPEATS} "
          f"(총 {n_splits * N_REPEATS} folds) | 인물 {len(classes)}명, 표본 {len(y)}장")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    results = {m: [] for m in methods}
    params_log = {m: [] for m in methods}

    fold_id = 0
    for tr, va in tqdm(list(rskf.split(X, y)), desc="CV folds"):
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]
        Dtr, Dva = cosine_dist_matrix(Xtr), cosine_dist_matrix(Xva)
        fold_id += 1

        for m in methods:
            if m == "DBSCAN":
                p = _best_param(cluster_dbscan, Xtr, Dtr, ytr, EPS_GRID)
                pred = cluster_dbscan(Xva, Dva, p)
            elif m == "Agglomerative":
                p = _best_param(cluster_agg, Xtr, Dtr, ytr, AGG_GRID)
                pred = cluster_agg(Xva, Dva, p)
            elif m == "HDBSCAN":
                p = _best_param(cluster_hdbscan, Xtr, Dtr, ytr, HDB_MCS_GRID)
                pred = cluster_hdbscan(Xva, Dva, p)
            elif m == "Infomap":
                p = _best_param(lambda X_, D_, t: cluster_infomap(X_, D_, t), Xtr, Dtr, ytr, TAU_GRID)
                pred = cluster_infomap(Xva, Dva, p)
            elif m == "PairwiseLink":
                clf, p = train_pairwise(Xtr, ytr)
                pred = cluster_pairwise(Xva, clf, p)
            else:
                continue
            results[m].append(all_metrics(yva, pred))
            params_log[m].append(float(p))

    return results, params_log, classes


# ---------------------------------------------------------------------
# 8. 결과 출력
# ---------------------------------------------------------------------
def summarize(results, params_log):
    def mean_std(vals):
        a = np.array(vals)
        return a.mean(), a.std()

    rows = []
    for m, recs in results.items():
        if not recs:
            continue
        bf = mean_std([r["bcubed_f"] for r in recs])
        bp = mean_std([r["bcubed_p"] for r in recs])
        br = mean_std([r["bcubed_r"] for r in recs])
        pf = mean_std([r["pair_f"] for r in recs])
        nmi = mean_std([r["nmi"] for r in recs])
        ari = mean_std([r["ari"] for r in recs])
        nc = mean_std([r["n_clusters"] for r in recs])
        pm = mean_std(params_log[m])
        rows.append((m, bf, bp, br, pf, nmi, ari, nc, pm))

    rows.sort(key=lambda r: r[1][0], reverse=True)

    print("\n" + "=" * 100)
    print(f"{'방법':<14}{'BCubed-F':>14}{'BCubed-P':>12}{'BCubed-R':>12}"
          f"{'Pair-F':>11}{'NMI':>10}{'ARI':>10}{'#군집':>9}{'param':>9}")
    print("-" * 100)
    for (m, bf, bp, br, pf, nmi, ari, nc, pm) in rows:
        print(f"{m:<14}{bf[0]:>8.3f}±{bf[1]:<4.2f}{bp[0]:>9.3f}{br[0]:>12.3f}"
              f"{pf[0]:>11.3f}{nmi[0]:>10.3f}{ari[0]:>10.3f}{nc[0]:>9.1f}{pm[0]:>9.3f}")
    print("=" * 100)
    if rows:
        best = rows[0]
        print(f"\n최고 성능: {best[0]} (BCubed-F={best[1][0]:.3f}±{best[1][1]:.2f}, "
              f"권장 param~={best[8][0]:.3f})")
    return rows


# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="얼굴 군집 알고리즘 stratified 비교/튜닝")
    parser.add_argument("--force-embed", action="store_true", help="임베딩 캐시 무시하고 재추출")
    parser.add_argument("--methods", default="DBSCAN,Agglomerative,HDBSCAN,Infomap,PairwiseLink")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    X, y_names, _ = build_dataset(force=args.force_embed)
    results, params_log, classes = run_cv(X, y_names, methods)
    rows = summarize(results, params_log)

    # 권장 설정 저장
    if rows:
        best = rows[0]
        out = {
            "best_method": best[0],
            "recommended_param": round(best[8][0], 4),
            "bcubed_f": round(best[1][0], 4),
            "n_identities": int(len(classes)),
            "n_samples": int(len(X)),
        }
        with open(os.path.join(_HERE, "cluster_benchmark_result.json"), "w", encoding="utf-8") as fp:
            json.dump(out, fp, ensure_ascii=False, indent=2)
        print(f"\n권장 설정 저장: cluster_benchmark_result.json -> {out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n완료! 소요 시간: {time.time() - t0:.1f}초")
