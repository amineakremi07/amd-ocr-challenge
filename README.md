# Mini Challenge 2 – OCR (AMD AI Academy Challenge)

VLM-based OCR using `Qwen/Qwen2.5-VL-7B-Instruct` (placeholder model) on ROCm.

## Layout
- `app/app.py` – takes `--input-image`, writes `/app/output/{image_name}_output.json`
- `app/requirements.txt` – Python dependencies
- `Dockerfile` – ROCm base image; model weights are pre-downloaded at build time
- `test_runner.py` – local helper (reads `test_input/`, writes `test_output/`)

## Output format
```json
{"text": "<extracted_text>", "confidence": 0.95}
```
`confidence` is the geometric-mean token probability of the generated text.

## Build & run
```bash
docker build -t ocr-challenge .
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$PWD/test_input:/input" -v "$PWD/test_output:/app/output" \
  ocr-challenge python3 /app/app.py --input-image /input/image_01.png
```

## Local test (no Docker)
```bash
pip install -r app/requirements.txt
mkdir test_input   # add sample images, e.g. image_01.png
python test_runner.py
```
`OUTPUT_DIR` env var overrides the output folder (defaults to `/app/output`).

## Notes
- The weights are baked into the image (`HF_HOME=/app/.cache/huggingface`) and
  `HF_HUB_OFFLINE=1` is set, so no download happens at startup.
- To change models, edit `OCR_MODEL` in the Dockerfile; the warm-up layer follows it.
- `Qwen2_5_VLForConditionalGeneration` is Qwen2.5-VL specific; another model family needs a different loader class.
