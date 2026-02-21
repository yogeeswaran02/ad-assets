"""Image generation using Nano Banana Pro (gemini-3-pro-image-preview)."""

from __future__ import annotations

import base64
import logging

from google import genai
from google.genai import types

from app.config import settings
from app.generators.creative import generate_creative_prompt
from app.jobs import get_assets_dir
from app.models import Job

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash-exp"  # Fallback; prefer gemini-3-pro-image-preview when available


async def generate_image(job: Job) -> None:
    """Generate an ad image using Nano Banana Pro."""
    job.progress = 20

    # Step 1: Generate creative prompt via LLM
    creative_prompt = await generate_creative_prompt(job.request)
    job.progress = 40

    # Step 2: Generate image via Gemini
    client = genai.Client(api_key=settings.gemini_api_key)

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=creative_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )
    job.progress = 80

    # Step 3: Save the generated image
    image_data = None
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            image_data = part.inline_data.data
            mime_type = part.inline_data.mime_type
            break

    if image_data is None:
        raise RuntimeError("No image returned by the model")

    ext = "png" if "png" in mime_type else "jpg"
    filename = f"{job.job_id}.{ext}"
    filepath = get_assets_dir() / filename

    if isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data

    filepath.write_bytes(image_bytes)

    job.result_filename = filename
    job.result_url = f"{settings.base_url}/assets/{filename}"
    job.progress = 100
