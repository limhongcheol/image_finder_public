import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 기본 경로 설정
base_path = os.path.join(os.getenv("LUMINA_PHOTO_ROOT", "runtime/photos"), "cropped_insight", "adaface_results")

# 이미지 이름을 키로, person 폴더들을 값으로 저장
image_people_map = defaultdict(list)

# person_0 ~ person_175 폴더 순회
for person_idx in range(176):
    person_folder = os.path.join(base_path, f"person_{person_idx}")

    if not os.path.exists(person_folder):
        continue

    # 각 person 폴더 내의 이미지 파일 처리
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # 마지막 _숫자 제거
            # 예: "2012-06-06 12.20.33_2.jpg" -> "2012-06-06 12.20.33.jpg"
            name_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]

            # 마지막 _숫자 패턴 제거
            if '_' in name_without_ext:
                clean_name = '_'.join(name_without_ext.split('_')[:-1]) + ext
            else:
                clean_name = filename

            # 딕셔너리에 추가
            image_people_map[clean_name].append(f"person_{person_idx}")

# DataFrame 생성
data = []
for image_name, people_list in sorted(image_people_map.items()):
    data.append({
        'image_name': image_name,
        'people': ','.join(people_list)
    })

df = pd.DataFrame(data)

# CSV 파일로 저장
output_csv = os.path.join(base_path, 'image_people_mapping.csv')
df.to_csv(output_csv, index=False, encoding='utf-8')

print(f"총 {len(df)} 개의 고유 이미지 발견")
print(f"CSV 파일 저장: {output_csv}")
print("\n샘플 데이터:")
print(df.head(10))
