Generation workflow (options)

1) Automatic1111 local/remote (recommended if you control a powerful GPU)
   - Start Automatic1111 server and ensure it can render 7680x3240 (consider tiled/patch mode if not).
   - Run the helper script:

```bash
pip install -r requirements.txt  # requests pillow
python assets/environment/generate_with_automatic1111.py --base_url http://127.0.0.1:7860 --api_key ''
```

2) Stable Horde / Cloud services
   - Use the `prompt.txt` and `negative.txt` files, pass parameters from `generation_manifest.json`.
   - Request 8 variants, set seeds from the manifest to keep reproducibility.

3) Midjourney / Flux Kontext
   - Use the master prompt and negative prompt, generate 8 images at 21:9 ratio.
   - Download full-resolution PNG/TIFF, then convert to WebP 8k in Photoshop/Imagemagick.

Postprocessing
- Select top 3 images by the criteria in `generation_manifest.json` (composition, lighting, material realism, center clearance).
- Produce a final optimized `assay_office_final.webp` and a development `preview_1920.webp`.

Deliverables
- assets/environment/out/assay_office_v01.webp ... v08.webp
- assets/environment/out/assay_office_final.webp
- assets/environment/out/preview_1920.webp

Notes
- If your backend cannot generate 7680x3240 directly, generate at a lower multiple and upscale with a high-quality upscaler.
- Keep original PNG/TIFF/EXR masters for archival.
