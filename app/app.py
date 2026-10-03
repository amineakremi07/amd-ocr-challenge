"""OCR inference entrypoint for AMD AI Academy Mini Challenge 2.

Usage:
    python3 /app/app.py --input-image /path/to/image_01.png

Writes /app/output/image_01_output.json containing:
    {"text": "<extracted_text>", "confidence": 0.95}
"""

import argparse
import json
import math
import os
import sys

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Placeholder model; swap via the OCR_MODEL env var or edit here.
MODEL_ID = os.environ.get("OCR_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# Output directory (overridable so test_runner.py can run outside the container).
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")

# Confidence reported if it cannot be computed from token probabilities.
FALLBACK_CONFIDENCE = 0.5

OCR_PROMPT = (
    "You are an OCR engine. Read the text in this image and output ONLY the "
    "extracted text, with no explanation, quotes or extra words.\n"
    "Rules:\n"
    "1. US license plates: output ONLY the plate number. Ignore the state "
    "name, slogans, stickers and any other decoration.\n"
    "2. Chinese license plates: output the full plate including the province "
    "character at the beginning (e.g. the first Chinese character).\n"
    "3. Multi-line text or signs: read from top to bottom, joining the lines "
    "with a single space.\n"
    "4. Preserve the original characters and case exactly as shown."
)


def load_model():
    """Load the VLM and its processor onto the GPU (CUDA or ROCm/HIP)."""
    if torch.cuda.is_available():  # ROCm builds of torch also report via torch.cuda
        device = "cuda"
        # bfloat16 is supported on MI-series GPUs; fall back to float16 otherwise.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        print("WARNING: no GPU found, running on CPU (slow).", file=sys.stderr)
        device, dtype = "cpu", torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def run_ocr(model, processor, image_path):
    """Run the model on one image. Returns (text, confidence)."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,  # deterministic output
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Keep only newly generated tokens (drop the prompt).
    new_tokens = out.sequences[:, inputs.input_ids.shape[1]:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    # Confidence = geometric mean of the per-token probabilities.
    try:
        scores = model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]
        confidence = math.exp(scores.float().mean().item()) if scores.numel() else 0.0
    except Exception as exc:  # never fail the run because of confidence
        print(f"WARNING: could not compute confidence: {exc}", file=sys.stderr)
        confidence = FALLBACK_CONFIDENCE

    return text, round(max(0.0, min(1.0, confidence)), 4)


def save_result(image_path, text, confidence):
    """Write {image_name}_output.json (extension stripped) to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{image_name}_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False keeps Chinese province characters readable.
        json.dump({"text": text, "confidence": confidence}, f, ensure_ascii=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="VLM-based OCR")
    parser.add_argument("--input-image", required=True, help="Path to the input image")
    args = parser.parse_args()

    if not os.path.isfile(args.input_image):
        sys.exit(f"Input image not found: {args.input_image}")

    model, processor = load_model()
    text, confidence = run_ocr(model, processor, args.input_image)
    out_path = save_result(args.input_image, text, confidence)
    print(f"Saved {out_path}: {text!r} (confidence={confidence})")


if __name__ == "__main__":
    main()
