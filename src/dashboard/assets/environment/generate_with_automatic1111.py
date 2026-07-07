"""
Script helper to generate the 8K variants using an Automatic1111-compatible API.
Usage:
  - Ensure Automatic1111 is running or set `--base_url` to your instance.
  - Install: `pip install requests pillow`
  - Run: `python generate_with_automatic1111.py --base_url http://127.0.0.1:7860 --api_key ''`

Notes:
  - This script will request 7680x3240 images; ensure your SD backend supports large sizes.
  - If your instance cannot generate at 8k, consider tiled generation or use an external service.
"""

import argparse
import json
import base64
import os
import requests
from PIL import Image
from io import BytesIO


def save_base64_image(b64, out_path):
    data = base64.b64decode(b64)
    img = Image.open(BytesIO(data))
    img.save(out_path, format="WEBP", quality=90)
    return out_path


def make_preview(in_path, out_path, width=1920):
    img = Image.open(in_path)
    w, h = img.size
    new_h = int(width * h / w)
    img = img.resize((width, new_h), Image.LANCZOS)
    img.save(out_path, format="WEBP", quality=90)
    return out_path


def main(args):
    base = args.base_url.rstrip("/")
    manifest = json.load(
        open("assets/environment/generation_manifest.json", "r", encoding="utf-8")
    )
    prompt = open("assets/environment/prompt.txt", "r", encoding="utf-8").read().strip()
    negative = (
        open("assets/environment/negative.txt", "r", encoding="utf-8").read().strip()
    )

    headers = {}
    # AUTOMATIC1111 uses no API key by default; some deployments add one
    if args.api_key:
        headers["api-key"] = args.api_key

    os.makedirs("assets/environment/out", exist_ok=True)

    for v in manifest["variants"]:
        filename = os.path.join("assets/environment/out", v["filename"])
        print("Requesting variant", v["id"], "->", filename)
        payload = {
            "prompt": prompt,
            "negative_prompt": negative,
            "width": manifest["parameters"]["width"],
            "height": manifest["parameters"]["height"],
            "sampler_name": manifest["parameters"]["sampler"],
            "steps": manifest["parameters"]["steps"],
            "cfg_scale": manifest["parameters"]["cfg_scale"],
            "seed": v.get("seed", -1),
            "n_samples": manifest["parameters"].get("samples", 1),
        }
        try:
            r = requests.post(
                base + "/sdapi/v1/txt2img", json=payload, headers=headers, timeout=600
            )
            r.raise_for_status()
            data = r.json()
            # Automatic1111 returns images as base64 list
            b64 = data["images"][0]
            save_base64_image(b64, filename)
        except Exception as e:
            print("Generation failed for", v["id"], e)
            continue

    # Create preview from v01 (or first successful file)
    out_dir = "assets/environment/out"
    final_manifest = os.path.join(out_dir, manifest["final"])
    preview_path = os.path.join(out_dir, manifest["preview"])
    # pick first existing variant
    for v in manifest["variants"]:
        p = os.path.join(out_dir, v["filename"])
        if os.path.exists(p):
            # also copy the chosen final (could be refined later)
            img = Image.open(p)
            img.save(final_manifest, format="WEBP", quality=90)
            make_preview(p, preview_path, width=1920)
            print("Saved final and preview to", final_manifest, preview_path)
            break
    else:
        print("No generated files found. Check API or reduce resolution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url", default="http://127.0.0.1:7860", help="Automatic1111 base URL"
    )
    parser.add_argument(
        "--api_key", default="", help="API key if your instance requires one"
    )
    args = parser.parse_args()
    main(args)
