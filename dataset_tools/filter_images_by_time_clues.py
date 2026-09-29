import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import pandas as pd
import google.generativeai as genai
from PIL import Image
from tqdm import tqdm
import json
import time

# =====================================================================
# 1. Gemini API 키 설정
# =====================================================================
# TODO: 여기에 발급받은 Gemini API 키를 직접 입력하거나 환경 변수로 설정하세요.
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# Gemini 1.5 Flash 모델 사용 (이미지 처리에 빠르고 효율적)
model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

# =====================================================================
# 2. 파일 경로 설정
# =====================================================================
INPUT_CSV_PATH = os.getenv("LUMINA_INPUT_CSV", "runtime/dataset.csv")
OUTPUT_CSV_PATH = os.getenv("LUMINA_FILTERED_CSV", "runtime/filtered_images.csv")

# =====================================================================
# 3. 프롬프트 정의
# =====================================================================
SYSTEM_PROMPT = """
**Role:** You are an expert data annotator specializing in visual temporal grounding. Your task is to analyze an image and its accompanying description to determine if there are **definitive, unambiguous visual clues** indicating the time of day or the season.

**Task:** Output a JSON object classifying whether the image contains reliable temporal clues.

**[Strict Evaluation Criteria]**

**Classify as `1` (True - Temporal Clues Present) ONLY IF at least one of the following is clearly visible:**
1. **Time of Day (Natural Light):**
   - Distinctive sunlight/shadows indicating daytime (e.g., harsh midday directional shadows, clear blue sky).
   - Golden hour phenomena (e.g., visible sunrise, sunset, deep orange/pink skies).
   - Nighttime indicators in an outdoor setting (e.g., pitch-black sky, visible stars, streetlights or car headlights illuminating the dark environment).
   - Indoor scenes where an outside window clearly reveals broad daylight, sunset, or a dark night.
2. **Season (Nature & Weather):**
   - Unambiguous seasonal weather phenomena (e.g., accumulated snow on the ground, blooming cherry blossoms, fully turning autumn foliage).
3. **Season (Apparel - Outdoors ONLY):**
   - People outdoors wearing extreme weather clothing (e.g., heavy winter puffer jackets, thick scarves) or highly specific summer attire (e.g., swimwear at a beach). *Note: Indoor clothing is NOT a reliable clue due to heating/cooling.*

**Classify as `0` (False - No Reliable Temporal Clues) IF the image falls into any of these categories:**
1. **Ambiguous Indoor Lighting:** Artificial lights in an indoor setting with no visible windows, or windows with drawn blinds. (Without a window, it could be 2 PM or 2 AM).
2. **Timeless Subjects:** Close-up macro shots of objects, food, text, screens, or studio portraits with solid backdrops where the surrounding environment is cropped out.
3. **Ambiguous Apparel:** People wearing generic clothing (e.g., a t-shirt or hoodie) indoors or in environments where the outside weather cannot be corroborated.
4. **Overcast/Featureless Skies:** A grey, cloudy sky without any other environmental context to indicate season or time of day.
5. **Clocks/Watches:** A visible clock face without environmental corroboration (e.g., an analog clock showing "8:00" does not differentiate between AM and PM).

**[Output Format]**
You must return ONLY a valid JSON object. Do not include markdown formatting like ```json or any conversational text.
{
  "is_temporal": <integer 1 or 0>,
  "reason": "<A brief, 1-2 sentence explanation in English detailing exactly which visual/textual element triggered the 1 or 0>"
}
"""

def analyze_image(image_path, description):
    try:
        # 이미지 로드
        img = Image.open(image_path).convert('RGB')

        # 프롬프트 구성
        prompt = f"{SYSTEM_PROMPT}\n\n[image description]\n{description}\n\nPlease analyze the image and description to determine if there are reliable temporal clues."

        # API 호출 (재시도 로직 포함)
        for attempt in range(3):
            try:
                response = model.generate_content(
                    [prompt, img],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )

                result = json.loads(response.text)
                return result.get('has_time_clue', False), result.get('reason', '')
            except Exception as e:
                if attempt == 2:
                    print(f"Error calling Gemini API for {image_path}: {e}")
                    return False, "API Error"
                time.sleep(2) # 짧은 대기 후 재시도

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return False, "Image Load Error"

def main():
    print(f"데이터 로딩 중: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    # 결과를 저장할 리스트
    results = []

    print(f"총 {len(df)}개의 이미지 분석 시작...")

    # iterrows 대신 좀 더 효율적인 방식으로 접근, tqdm으로 진행률 표시
    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row['File Name']
        description = row['Description']
        full_path = row['Full Path']

        # 파일이 실제로 존재하는지 확인
        if pd.isna(full_path) or not os.path.exists(full_path):
            has_clue, reason = False, "File Not Found"
        else:
            has_clue, reason = analyze_image(full_path, description)

        results.append({
            'File Name': file_name,
            'Description': description,
            'DateTime': row.get('DateTime', ''),
            'Location': row.get('Location', ''),
            'Full Path': full_path,
            'Status': row.get('Status', ''),
            'Has_Time_Clue': has_clue,
            'Reason': reason
        })

        # 중간 저장 (선택 사항: 데이터가 많으므로 중간에 멈췄을 때를 대비)
        if (index + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV_PATH + ".temp", index=False, encoding='utf-8-sig')

    # 최종 결과 저장
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print(f"\n분석 완료! 결과가 저장되었습니다: {OUTPUT_CSV_PATH}")

    # 시간적 단서가 있는 데이터만 필터링하여 따로 저장 (옵션)
    filtered_df = output_df[output_df['Has_Time_Clue'] == True]
    filtered_csv_path = OUTPUT_CSV_PATH.replace('.csv', '_only_true.csv')
    filtered_df.to_csv(filtered_csv_path, index=False, encoding='utf-8-sig')
    print(f"시간적 단서가 있는 이미지만 따로 저장되었습니다: {filtered_csv_path}")
    print(f"총 {len(df)}개 중 {len(filtered_df)}개 선별됨.")

if __name__ == "__main__":
    main()
