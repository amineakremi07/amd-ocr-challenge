"""Local testing helper: simulates `python3 /app/app.py --input-image <path>`.

Runs app/app.py on every image in test_input/ (or one given on the command
line), writes results to test_output/, and validates the JSON produced.

Usage:
    python test_runner.py                      # all images in test_input/
    python test_runner.py test_input/a.png     # a single image
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(ROOT, "app", "app.py")
INPUT_DIR = os.path.join(ROOT, "test_input")
OUTPUT_DIR = os.path.join(ROOT, "test_output")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def check_output(image_path):
    """Return an error string, or None if the output JSON is valid."""
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}_output.json")
    if not os.path.isfile(out_path):
        return f"missing output file {out_path}"
    try:
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return f"invalid JSON: {exc}"
    if set(data) != {"text", "confidence"}:
        return f"unexpected keys: {sorted(data)}"
    if not isinstance(data["text"], str):
        return "'text' must be a string"
    if isinstance(data["confidence"], bool) or not isinstance(data["confidence"], (int, float)):
        return "'confidence' must be a number"
    if not 0.0 <= data["confidence"] <= 1.0:
        return "'confidence' must be between 0 and 1"
    print(f"  -> {data}")
    return None


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        images = sys.argv[1:]
    else:
        images = [
            os.path.join(INPUT_DIR, f)
            for f in sorted(os.listdir(INPUT_DIR))
            if f.lower().endswith(IMAGE_EXTS)
        ]
    if not images:
        sys.exit(f"No images found. Put sample images in {INPUT_DIR}")

    # Point the app at test_output/ instead of /app/output.
    env = dict(os.environ, OUTPUT_DIR=OUTPUT_DIR)
    failures = 0
    for image in images:
        print(f"[RUN] {image}")
        result = subprocess.run([sys.executable, APP, "--input-image", image], env=env)
        error = f"app exited with code {result.returncode}" if result.returncode else check_output(image)
        if error:
            failures += 1
            print(f"[FAIL] {image}: {error}")
        else:
            print(f"[PASS] {image}")

    print(f"\n{len(images) - failures}/{len(images)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
