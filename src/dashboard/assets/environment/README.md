# Assay Office 8K asset instructions

This folder should contain the final photoreal background and auxiliary passes.

Preferred pipeline:
1. Generate a photoreal 8K PNG (7680x3240) using Midjourney / SDXL / Flux Kontext with the provided master prompt.
2. Produce a depth pass (`assay_office_depth.png`) and a lighting pass (`lighting_pass.png`) for compositing and parallax.
3. Postprocess in Photoshop/Lightroom: lens grain, filmic color grade, highlight rolloff, denoise, export WebP 90.

Filenames expected by project:
- `assay_office_8k.webp` — final background (WebP, 8k, sRGB, 16-bit if supported)
- `assay_office_depth.png` — linear depth map for parallax/DOF
- `lighting_pass.png` — lighting-only pass for bloom/glow

Notes:
- Keep the center 65% of the canvas visually clean for the UI overlay.
- Save working PSD/EXR in a separate archive if needed.
- Use the master prompt and negative prompt exactly as documented in the product brief.
