import torch
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from flask import Flask, render_template, request, send_from_directory
import logging

# --- [1. 경로 및 설정] ---
PT_DIR = os.getenv("LUMINA_EMBEDDINGS_DIR", "runtime/embeddings")
IMG_DIR = os.getenv("LUMINA_IMAGES_DIR", "runtime/images")
OUTPUT_TXT = "selected_images_list.txt"
N_CLUSTERS = 50

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

cluster_results = {}
df_clusters = None

# --- [2. 클러스터링 로직 (동일)] ---
def perform_clustering():
    global cluster_results, df_clusters
    print("📦 9번 레이어 기반 클러스터링 시작 (전체 데이터 로드 중)...")

    features = []
    file_names = []

    pt_files = [f for f in os.listdir(PT_DIR) if f.endswith('.pt')]

    for f in pt_files:
        try:
            path = os.path.join(PT_DIR, f)
            data = torch.load(path, map_location='cpu')
            h9 = data['hidden_states9'].float().mean(dim=0).numpy()
            features.append(h9)
            file_names.append(f.replace('.pt', ''))
        except: continue

    features = np.array(features)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    df_clusters = pd.DataFrame({'file_name': file_names, 'cluster': labels})
    for i in range(N_CLUSTERS):
        cluster_results[i] = df_clusters[df_clusters['cluster'] == i]['file_name'].tolist()
    print(f"✅ 클러스터링 완료!")

# --- [3. Flask 라우팅] ---

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)

@app.route('/')
def index():
    display_data = []
    for cid, fnames in cluster_results.items():
        # [수정] 제한 없이 해당 클러스터의 모든 이미지를 전달합니다.
        display_data.append({
            'cluster_id': cid,
            'count': len(fnames),
            'samples': fnames
        })

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>COCO Full Cluster Inspector</title>
        <style>
            body { font-family: 'Pretendard', sans-serif; background: #1a1a1b; color: #eee; padding: 20px; margin: 0; }
            .header { background: #272729; padding: 20px; border-radius: 0 0 15px 15px; margin-bottom: 30px; border-bottom: 2px solid #3498db; position: sticky; top: 0; z-index: 1001; }

            .cluster-box { background: #272729; margin-bottom: 40px; padding: 20px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }

            /* 가로 스크롤 컨테이너 핵심 설정 */
            .img-scroll-container {
                display: flex;
                gap: 15px;
                margin: 20px 0;
                overflow-x: auto; /* 가로 스크롤 활성화 */
                padding: 10px;
                white-space: nowrap;
                scrollbar-width: thick; /* 파이어폭스용 스크롤바 스타일 */
                scrollbar-color: #3498db #1a1a1b;
            }

            /* 크롬/사파리용 스크롤바 디자인 */
            .img-scroll-container::-webkit-scrollbar { height: 12px; }
            .img-scroll-container::-webkit-scrollbar-track { background: #1a1a1b; border-radius: 10px; }
            .img-scroll-container::-webkit-scrollbar-thumb { background: #3498db; border-radius: 10px; }

            .img-item { flex: 0 0 auto; width: 200px; text-align: center; }
            img {
                width: 200px; height: 200px;
                object-fit: cover;
                border-radius: 10px;
                border: 2px solid #3e3e42;
                transition: transform 0.2s, border-color 0.2s;
                loading: lazy; /* 브라우저 성능을 위해 지연 로딩 적용 */
            }
            img:hover { transform: scale(1.05); border-color: #3498db; }

            .meta { font-size: 1.2em; font-weight: bold; display: flex; align-items: center; gap: 15px; }
            .meta input[type="checkbox"] { width: 25px; height: 25px; cursor: pointer; }

            .count-badge { background: #3498db; color: white; padding: 3px 10px; border-radius: 20px; font-size: 0.8em; }

            .btn-save {
                position: fixed; bottom: 40px; right: 40px;
                padding: 20px 50px; background: #3498db; color: white;
                border: none; border-radius: 50px; cursor: pointer;
                font-size: 20px; font-weight: bold;
                box-shadow: 0 10px 20px rgba(52, 152, 219, 0.4); z-index: 1002;
            }
            .btn-save:hover { background: #2980b9; transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📦 Cluster Inspector (Layer 9 Full-View)</h1>
            <p>총 {{ data|length }}개의 클러스터가 생성되었습니다. 각 줄을 가로로 스크롤하여 검수하세요.</p>
        </div>

        <form action="/save" method="post">
            <button type="submit" class="btn-save">선택한 클러스터 리스트 확정 저장</button>
            {% for cluster in data %}
            <div class="cluster-box">
                <label class="meta">
                    <input type="checkbox" name="selected_clusters" value="{{ cluster.cluster_id }}">
                    <span>Cluster #{{ cluster.cluster_id }}</span>
                    <span class="count-badge">{{ cluster.count }} items</span>
                </label>
                <div class="img-scroll-container">
                    {% for img in cluster.samples %}
                    <div class="img-item">
                        <img src="/images/{{ img }}" loading="lazy" onerror="this.style.display='none'">
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </form>
    </body>
    </html>
    """
    return render_template(app.jinja_env.from_string(html_template), data=display_data)

@app.route('/save', methods=['POST'])
def save():
    selected_ids = request.form.getlist('selected_clusters')
    selected_ids = [int(i) for i in selected_ids]
    all_selected_files = []
    for sid in selected_ids:
        all_selected_files.extend(cluster_results[sid])

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for fname in all_selected_files:
            f.write(fname + "\n")

    return f"<h2>저장 완료!</h2><p>{len(all_selected_files)}개의 파일이 {OUTPUT_TXT}에 기록되었습니다.</p><a href='/'>돌아가기</a>"

if __name__ == "__main__":
    perform_clustering()
    app.run(host='0.0.0.0', port=5000, debug=False)
