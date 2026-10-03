FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Keep model weights inside the image so no download happens at run time.
ENV HF_HOME=/app/.cache/huggingface \
    OCR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# Install dependencies first so this layer is cached when only app code changes.
# (torch is already in the base image, so pip keeps the ROCm build.)
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Warm-up layer: download model weights + processor at build time
# to keep container startup well under the 10-minute limit.
RUN python3 -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['OCR_MODEL'])"

# Weights are baked in; avoid any network access at run time.
ENV HF_HUB_OFFLINE=1

# Copy application code.
COPY app/ /app/

CMD ["python3", "/app/app.py"]
