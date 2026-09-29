import os
import sys
import argparse
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

LAYER_IDX = 9
IMAGE_SIDE = 448
DEFAULT_MODEL = os.getenv("LUMINA_TIME_WEIGHTS", os.path.join(BASE_DIR, "saved_models", "best_layer_9.pth"))

_PROCESSOR = None
_QWEN = None
_PROBE = None
_DEVICE = None
_GEMINI = None
_md = None
_ic = None


def load_models(model_path=DEFAULT_MODEL, layer_idx=LAYER_IDX, use_caption=True):
    global _PROCESSOR, _QWEN, _PROBE, _DEVICE, _GEMINI, _md, _ic
    import torch
    import make_dataset as md
    import make_dataset_embedding as emb
    import inference_code as ic

    _md, _ic = md, ic
    ic.YEAR = 2026

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference_one_image] device = {_DEVICE}")

    print("[inference_one_image] Loading Qwen embedding model...")
    _PROCESSOR, _QWEN = emb.load_model()

    print(f"[inference_one_image] Loading Probe(SequenceProbeModel)... ({model_path})")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weight file not found: {model_path}")
    _PROBE = ic.SequenceProbeModel().to(_DEVICE)
    _PROBE.load_state_dict(torch.load(model_path, map_location=_DEVICE))
    _PROBE.eval()

    if use_caption:
        print("[inference_one_image] Loading Gemini caption client...")
        _GEMINI = md.init_vertex()
    print("[inference_one_image] All models loaded.")


def _embed_image_caption(pil_image, caption, layer_idx=LAYER_IDX):
    """Image + caption -> hidden_states from the selected layer (seq, 4096)."""
    import torch
    image = pil_image.convert("RGB")
    image.thumbnail((IMAGE_SIDE, IMAGE_SIDE))

    content = [{"type": "image"}]
    if caption:
        content.append({"type": "text", "text": str(caption)})
    messages = [{"role": "user", "content": content}]
    prompt = _PROCESSOR.apply_chat_template(messages, add_generation_prompt=True)
    inputs = _PROCESSOR(text=[prompt], images=[image], return_tensors="pt").to(_QWEN.device)

    with torch.no_grad():
        outputs = _QWEN(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer_idx].squeeze(0).to(torch.float32).cpu()
    del outputs, inputs
    return hidden


def predict_datetime(pil_image, caption=None, layer_idx=LAYER_IDX):
    import torch
    import pandas as pd

    actual, lat = None, None
    try:
        dt_str, lat, _ = _md.extract_metadata(pil_image)
        if dt_str:
            actual = dt_str
    except Exception:
        actual, lat = None, None

    if caption is None and _GEMINI is not None:
        try:
            caption = _md.generate_description(_GEMINI, pil_image)
        except Exception as e:
            print(f"[inference_one_image] Caption generation failed; using image only: {e}")
            caption = ""

    hidden = _embed_image_caption(pil_image, caption, layer_idx)
    h = hidden.unsqueeze(0).to(_DEVICE)
    mask = torch.ones(h.shape[:2], dtype=torch.bool, device=_DEVICE)

    with torch.no_grad():
        pred = _PROBE(h, mask).cpu().numpy()[0]

    pred_dt = _ic.inverse_angular_time(float(pred[0]), float(pred[1]))

    try:
        if lat is not None and float(lat) < -5.0:
            pred_dt = (pred_dt - pd.DateOffset(months=6)).to_pydatetime()
    except (TypeError, ValueError):
        pass

    return pred_dt, actual, caption


def _load_font(size):
    for name in ("arial.ttf", "malgun.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_bottom_right(image, lines):
    draw = ImageDraw.Draw(image)
    w, h = image.size

    font_size = max(16, int(min(w, h) * 0.045))
    font = _load_font(font_size)
    margin = max(8, int(font_size * 0.5))
    line_gap = int(font_size * 0.3)

    sizes = []
    for text in lines:
        box = draw.textbbox((0, 0), text, font=font)
        sizes.append((box[2] - box[0], box[3] - box[1]))

    block_w = max(s[0] for s in sizes)
    block_h = sum(s[1] for s in sizes) + line_gap * (len(lines) - 1)

    pad = margin
    box_x1 = w - block_w - margin - pad
    box_y1 = h - block_h - margin - pad
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle(
        [box_x1, box_y1, w - margin + pad, h - margin + pad],
        fill=(0, 0, 0, 140),
    )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    y = h - block_h - margin
    for text, (tw, th) in zip(lines, sizes):
        x = w - tw - margin
        draw.text(
            (x, y), text, font=font, fill=(255, 255, 255),
            stroke_width=max(1, font_size // 12), stroke_fill=(0, 0, 0),
        )
        y += th + line_gap

    return image


def _fmt_actual(actual):
    if not actual:
        return "Actual: N/A"
    try:
        dt = datetime.strptime(actual, "%Y-%m-%d %H:%M:%S")
        return "Actual: " + dt.strftime("%m-%d %H:%M")
    except ValueError:
        return "Actual: " + str(actual)


def infer_and_annotate(image_path, output_path=None, caption=None, layer_idx=LAYER_IDX):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    pil = Image.open(image_path)
    pred_dt, actual, _ = predict_datetime(pil, caption=caption, layer_idx=layer_idx)

    pred_line = "Pred:   " + pred_dt.strftime("%m-%d %H:%M")
    actual_line = _fmt_actual(actual)

    annotated = _draw_bottom_right(pil.convert("RGB"), [pred_line, actual_line])

    if output_path is None:
        root, _ = os.path.splitext(image_path)
        output_path = root + "_pred.jpg"
    annotated.save(output_path, quality=95)

    print(f"[inference_one_image] Predicted time (without year): {pred_dt.strftime('%m-%d %H:%M')}")
    print(f"[inference_one_image] Actual time: {actual if actual else 'N/A'}")
    print(f"[inference_one_image] Saved result: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Infer and annotate the capture time for a single image")
    parser.add_argument("image", help="Path to the image to infer")
    parser.add_argument("-o", "--output", default=None, help="Path to save the result")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to probe weights")
    parser.add_argument("--layer", type=int, default=LAYER_IDX, help="Qwen hidden layer index")
    parser.add_argument("--caption", default=None, help="Provide a caption directly; Gemini is used when omitted")
    parser.add_argument(
        "--no-caption",
        action="store_true",
        help="Skip caption generation; not recommended because predictions may collapse to a constant",
    )
    args = parser.parse_args()

    print("[inference_one_image] Loading models...")
    use_caption = (args.caption is None) and (not args.no_caption)
    load_models(model_path=args.model, layer_idx=args.layer, use_caption=use_caption)

    caption = "" if args.no_caption else args.caption
    infer_and_annotate(args.image, args.output, caption=caption, layer_idx=args.layer)


if __name__ == "__main__":
    main()
