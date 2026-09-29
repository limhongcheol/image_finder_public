
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR       = os.getenv("LUMINA_IMAGES_DIR", os.path.join(BASE_DIR, "runtime", "images"))
DATASET_CSV     = os.getenv("LUMINA_DATASET_CSV", os.path.join(BASE_DIR, "runtime", "dataset.csv"))
PT_DIR          = os.getenv("LUMINA_EMBEDDINGS_DIR", os.path.join(BASE_DIR, "runtime", "embeddings"))
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "saved_models")
RESULT_DIR      = os.path.join(BASE_DIR, "analysis_result")
GCP_KEY_PATH    = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

PROJECT_ID      = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION        = "global"
GEMINI_MODEL    = "gemini-2.5-flash"
DESC_MAX_SIDE   = 1024
DESC_TEMP       = 0.2
DESC_MAX_TOKENS = 2048


EMBED_MODEL_ID    = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
EMBED_IMAGE_SIDE  = 448
EMPTY_CACHE_EVERY = 50

LAYERS  = [9]                               # 전 파이프라인 공통: layer 9 (best_layer_9.pth)
WEIGHTS = {9: 1.0}                           # 단일 레이어이므로 가중치 1.0
YEAR    = 2026

EPOCHS     = 10
BATCH_SIZE = 8
LR         = 1e-4
INPUT_DIM  = 4096

INFER_LAYER       = 9
INFER_INPUT_CSV   = DATASET_CSV
INFER_PT_DIR      = PT_DIR
BEST_MODEL_PATH   = os.path.join(MODEL_SAVE_DIR, f"best_layer_{INFER_LAYER}.pth")
INFER_OUTPUT_CSV  = os.path.join(RESULT_DIR, "inference_results.csv")

TRAIN_EDA_INPUT = os.path.join(RESULT_DIR, f"individual_layer_{INFER_LAYER}.csv")
INFER_EDA_INPUT = INFER_OUTPUT_CSV

STEPS = {
    "make_dataset":  True,
    "embedding":     True,
    "train":         True,
    "inference":     True,
    "train_eda":     True,
    "inference_eda": True,
}


def step_make_dataset():
    import make_dataset as md
    kwargs = dict(
        image_dir=IMAGE_DIR, output_csv=DATASET_CSV,
        project_id=PROJECT_ID, location=LOCATION, gcp_key_path=GCP_KEY_PATH,
        model_id=GEMINI_MODEL, temperature=DESC_TEMP,
        max_output_tokens=DESC_MAX_TOKENS, max_side=DESC_MAX_SIDE,
    )
    md.build_dataset(**kwargs)


def step_embedding():
    import make_dataset_embedding as emb
    emb.run_embedding(
        csv_path=DATASET_CSV, save_dir=PT_DIR, model_id=EMBED_MODEL_ID,
        target_layers=tuple(LAYERS), image_side=EMBED_IMAGE_SIDE,
        empty_cache_every=EMPTY_CACHE_EVERY, image_dir=IMAGE_DIR,
    )


def step_train():
    import train_code as tc
    tc.YEAR = YEAR
    tc.run_integrated_analysis(
        pt_dir=PT_DIR, csv_path=DATASET_CSV,
        layers=list(LAYERS), weights=WEIGHTS,
        result_dir=RESULT_DIR, model_save_dir=MODEL_SAVE_DIR,
        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, input_dim=INPUT_DIM,
    )


def step_inference():
    import inference_code as ic
    ic.YEAR = YEAR
    ic.run_inference(
        model_path=BEST_MODEL_PATH, new_csv_path=INFER_INPUT_CSV,
        pt_dir=INFER_PT_DIR, output_save_path=INFER_OUTPUT_CSV,
        layer_idx=INFER_LAYER,
    )


def step_train_eda():
    import train_eda
    train_eda.perform_detailed_eda(TRAIN_EDA_INPUT)


def step_inference_eda():
    import inference_eda
    inference_eda.perform_detailed_eda(INFER_EDA_INPUT)


_RUNNERS = [
    ("make_dataset",  step_make_dataset),
    ("embedding",     step_embedding),
    ("train",         step_train),
    ("inference",     step_inference),
    ("train_eda",     step_train_eda),
    ("inference_eda", step_inference_eda),
]


def main():
    for name, runner in _RUNNERS:
        if not STEPS.get(name):
            continue
        print(f"\n{'#'*60}\n# > STEP: {name}\n{'#'*60}")
        runner()
    print("\n Pipeline complete.")


if __name__ == "__main__":
    main()
