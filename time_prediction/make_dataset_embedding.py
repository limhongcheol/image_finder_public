import os
import gc

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

import make_dataset as md

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID    = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
SAVE_DIR    = os.getenv("LUMINA_EMBEDDINGS_DIR", "runtime/embeddings")
CSV_PATH    = md.OUTPUT_CSV

TARGET_LAYERS = (9,)       # best_layer_9.pth 로 통일 (layer 9 만 추출)
IMAGE_SIDE  = 448
EMPTY_CACHE_EVERY = 50


def load_model(model_id=MODEL_ID):
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return processor, model


def build_worklist(csv_path=CSV_PATH, image_dir=None):
    """Create worklist from CSV if exists, else scan image_dir.

    Returns list of (file_name, full_path, description). The description (caption)
    is fed to the model together with the image; without it every image collapses
    to the same text-only embedding.
    """
    if image_dir is None:
        image_dir = md.IMAGE_DIR
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        items = []
        for _, row in df.iterrows():
            file_name = str(row["File Name"]).strip()
            full_path = str(row["Full Path"]).strip()
            if not full_path or full_path == "nan":
                full_path = os.path.join(image_dir, file_name)
            description = str(row.get("Description", "") or "").strip()
            if description == "nan":
                description = ""
            items.append((file_name, full_path, description))
        print(f"CSV-based worklist: {len(items)} items ({csv_path})")
        return items

    paths = md.collect_images(image_dir)
    print(f"No CSV found. Folder scan: {len(paths)} items ({image_dir})")
    return [(os.path.basename(p), p, "") for p in paths]


def embed_image(processor, model, full_path, caption="", image_side=IMAGE_SIDE, target_layers=TARGET_LAYERS):
    with Image.open(full_path) as img:
        image = img.convert("RGB")
        image.thumbnail((image_side, image_side))

    # 학습 입력과 동일하게 이미지 + 캡션(설명) 을 함께 넣는다.
    # 캡션이 없으면 이미지 토큰만 남아 모든 이미지가 동일한 임베딩이 된다.
    content = [{"type": "image"}]
    if caption:
        content.append({"type": "text", "text": str(caption)})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    embedding_dict = {}
    for i, layer_tensor in enumerate(outputs.hidden_states):
        if i in target_layers:
            embedding_dict[f"hidden_states{i}"] = layer_tensor.squeeze(0).to(torch.float16).cpu()

    del outputs, inputs, image
    return embedding_dict


def run_embedding(csv_path=CSV_PATH, save_dir=SAVE_DIR, model_id=MODEL_ID,
                  target_layers=TARGET_LAYERS, image_side=IMAGE_SIDE,
                  empty_cache_every=EMPTY_CACHE_EVERY, image_dir=None):
    os.makedirs(save_dir, exist_ok=True)
    processor, model = load_model(model_id)
    worklist = build_worklist(csv_path, image_dir)

    done = 0
    for count, (file_name, full_path, description) in enumerate(
        tqdm(worklist, desc="Extracting embeddings"), start=1
    ):
        pt_file_path = os.path.join(save_dir, f"{file_name}.pt")
        if os.path.exists(pt_file_path):
            continue

        try:
            embedding_dict = embed_image(processor, model, full_path, description, image_side, target_layers)
            torch.save(embedding_dict, pt_file_path)
            del embedding_dict
            done += 1
        except Exception as e:
            tqdm.write(f"Error ({file_name}): {e}")

        if count % empty_cache_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nComplete! {done} new items saved. Folder: {save_dir}")


if __name__ == "__main__":
    run_embedding()
