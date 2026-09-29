import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
import io

import pandas as pd
from PIL import Image, ExifTags



MODEL_ID    = "gemini-2.5-flash"
PROJECT_ID  = os.environ["GOOGLE_CLOUD_PROJECT"]   # GCP project from gcp-key.json
LOCATION    = "global"

GCP_KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

IMAGE_DIR  = os.getenv("LUMINA_IMAGES_DIR", "runtime/images")
OUTPUT_CSV = os.getenv("LUMINA_DATASET_CSV", "runtime/dataset.csv")

COLUMNS = ["File Name", "Description", "DateTime", "Full Path",
           "Status", "latitude", "longitude", "Is_Temporal"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SYSTEM_PROMPT = """System Role: You are an expert image annotator creating elite-level training data in the ShareGPT4V style. Your objective is to provide an exhaustive, extremely detailed, and highly structured description of the provided image.

Task: Analyze the image and generate a dense caption (approx. 300-600 words). You must pay hyper-focused attention to the BACKGROUND details and TEMPORAL/ENVIRONMENTAL CUES. Do not hallucinate; base all descriptions on explicit visual evidence.

Structure your response exactly as follows:

1. Global Summary: Provide a 2-3 sentence overview of the entire scene, capturing the main subjects and the overall atmosphere.
2. Spatial & Background Analysis: > - Break down the scene into Foreground, Middleground, and Background.

Exhaustively detail the background elements: architecture, nature, textures, objects, and spatial relationships. Describe exactly where things are located relative to each other (e.g., "In the top right quadrant...", "Behind the main subject...").
3. Temporal & Era Cues (Crucial): > - Time of Day: Analyze the direction of light source, length/sharpness of shadows, and sky color to deduce the specific time of day. Explain your reasoning.

Season & Weather: Identify seasonal indicators (e.g., foliage, clothing styles, ground conditions like wetness or snow) and the current weather condition.

Era/Period: Point out any specific objects (e.g., vintage electronics, car models, architectural styles, fashion) that serve as chronological anchors.
4. Micro-details & Context: Describe any subtle interactions between the subjects and the background, text/signs visible in the background, or ambient elements that complete the semantic context of the scene."""

USER_PROMPT = "Describe this image following the exact structure specified in the system instructions."


_genai_types = None


def init_vertex(project_id=PROJECT_ID, location=LOCATION, gcp_key_path=GCP_KEY_PATH):
    """Initialize and return Vertex mode google-genai client. genai is lazy-imported here."""
    global _genai_types
    from google import genai
    from google.genai import types
    _genai_types = types

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_key_path
    client = genai.Client(vertexai=True, project=project_id, location=location)
    return client


_TAG_NAME = {v: k for k, v in ExifTags.TAGS.items()}
_GPS_NAME = {v: k for k, v in ExifTags.GPSTAGS.items()}


def _to_degrees(value):
    """Convert EXIF GPS (degrees, minutes, seconds) format to decimal degrees."""
    def _ratio(x):
        try:
            return x[0] / x[1]
        except (TypeError, IndexError):
            return float(x)
    d, m, s = value
    return _ratio(d) + _ratio(m) / 60.0 + _ratio(s) / 3600.0


def extract_metadata(img):
    """Extract (datetime_str, latitude, longitude) from PIL image. None if missing."""
    dt_str, lat, lon = None, None, None
    try:
        exif = img._getexif()
    except Exception:
        exif = None
    if not exif:
        return dt_str, lat, lon

    for tag_name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        tag_id = _TAG_NAME.get(tag_name)
        raw = exif.get(tag_id)
        if raw:
            raw = str(raw).strip()
            if len(raw) >= 19 and raw[4] == ":" and raw[7] == ":":
                raw = raw[:4] + "-" + raw[5:7] + "-" + raw[8:]
            try:
                dt_str = pd.to_datetime(raw).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_str = None
            if dt_str:
                break

    gps_id = _TAG_NAME.get("GPSInfo")
    gps = exif.get(gps_id)
    if gps:
        try:
            lat_ref = gps.get(_GPS_NAME["GPSLatitudeRef"])
            lat_val = gps.get(_GPS_NAME["GPSLatitude"])
            lon_ref = gps.get(_GPS_NAME["GPSLongitudeRef"])
            lon_val = gps.get(_GPS_NAME["GPSLongitude"])
            if lat_val and lon_val:
                lat = _to_degrees(lat_val)
                if lat_ref in ("S", b"S"):
                    lat = -lat
                lon = _to_degrees(lon_val)
                if lon_ref in ("W", b"W"):
                    lon = -lon
        except Exception:
            lat, lon = None, None

    return dt_str, lat, lon


def image_to_jpeg_bytes(img, max_side=1024):
    """Convert image to JPEG bytes (scale long side to max_side)."""
    im = img.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        im = im.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def generate_description(client, img, model_id=MODEL_ID,
                         system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT,
                         temperature=0.2, max_output_tokens=2048, max_side=1024):
    types = _genai_types
    image_part = types.Part.from_bytes(
        data=image_to_jpeg_bytes(img, max_side=max_side), mime_type="image/jpeg")
    resp = client.models.generate_content(
        model=model_id,
        contents=[user_prompt, image_part],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return resp.text.strip()


def collect_images(image_dir):
    files = []
    for root, _, names in os.walk(image_dir):
        for n in names:
            if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                files.append(os.path.join(root, n))
    return sorted(files)


def build_dataset(image_dir=IMAGE_DIR, output_csv=OUTPUT_CSV,
                  project_id=PROJECT_ID, location=LOCATION, gcp_key_path=GCP_KEY_PATH,
                  model_id=MODEL_ID, system_prompt=SYSTEM_PROMPT, user_prompt=USER_PROMPT,
                  temperature=0.2, max_output_tokens=2048, max_side=1024):
    client = init_vertex(project_id, location, gcp_key_path)

    processed = set()
    if os.path.exists(output_csv):
        try:
            done = pd.read_csv(output_csv)
            processed = set(done["File Name"].astype(str))
            print(f"Found {len(processed)} items in existing CSV, skipping")
        except Exception:
            processed = set()

    images = collect_images(image_dir)
    print(f"Total {len(images)} images found (target: {image_dir})")

    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0

    for i, path in enumerate(images, 1):
        file_name = os.path.basename(path)
        if file_name in processed:
            continue

        print(f"[{i}/{len(images)}] Processing: {file_name}")
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"Image open failed, skipping: {e}")
            continue

        dt_str, lat, lon = extract_metadata(img)
        status = 1 if dt_str else 0

        try:
            description = generate_description(
                client, img, model_id=model_id,
                system_prompt=system_prompt, user_prompt=user_prompt,
                temperature=temperature, max_output_tokens=max_output_tokens,
                max_side=max_side)
        except Exception as e:
            print(f"Description generation failed, skipping: {e}")
            continue

        row = {
            "File Name":   file_name,
            "Description": description,
            "DateTime":    dt_str if dt_str else "",
            "Full Path":   os.path.abspath(path),
            "Status":      status,
            "latitude":    lat if lat is not None else "",
            "longitude":   lon if lon is not None else "",
            "Is_Temporal": True,
        }

        pd.DataFrame([row], columns=COLUMNS).to_csv(
            output_csv, mode="a", header=write_header,
            index=False, encoding="utf-8-sig",
        )
        write_header = False
        print(f"  saved (Status={status}, "
              f"GPS={'O' if lat is not None else 'X'})")

    print(f"\nFinished! Results saved to: {output_csv}")


if __name__ == "__main__":
    build_dataset()
