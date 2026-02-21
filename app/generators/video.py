"""Video generation using Google Veo 3.1."""

from __future__ import annotations

import asyncio
import logging

from google import genai
from google.genai import types

from app.config import settings
from app.generators.creative import generate_creative_prompt
from app.jobs import get_assets_dir
from app.models import Job

logger = logging.getLogger(__name__)

MODEL = "veo-3.0-generate-preview"  # Use veo-3.1-generate-preview when available


async def generate_video(job: Job) -> None:
    """Generate an ad video using Veo 3.1."""
    job.progress = 10

    # Step 1: Generate creative prompt via LLM
    creative_prompt = await generate_creative_prompt(job.request)
    job.progress = 20

    # Step 2: Submit video generation (async operation)
    client = genai.Client(api_key=settings.gemini_api_key)

    operation = client.models.generate_videos(
        model=MODEL,
        prompt=creative_prompt,
        config=types.GenerateVideosConfig(
            number_of_videos=1,
            duration_seconds=8,
            aspect_ratio=_map_aspect_ratio(job.request.aspect_ratio.value),
            negative_prompt="blurry, low quality, distorted, watermark, text overlay, cartoon",
        ),
    )
    job.progress = 30

    # Step 3: Poll for completion
    poll_interval = 15  # seconds
    max_polls = 30  # up to ~7.5 minutes
    for i in range(max_polls):
        if operation.done:
            break
        await asyncio.sleep(poll_interval)
        operation = client.operations.get(operation)
        # Estimate progress: 30% -> 90% over the polling period
        job.progress = min(30 + int(60 * (i + 1) / max_polls), 90)

    if not operation.done:
        raise RuntimeError("Video generation timed out after polling")

    if not operation.response or not operation.response.generated_videos:
        raise RuntimeError("No video returned by the model")

    # Step 4: Save the video
    video = operation.response.generated_videos[0]
    filename = f"{job.job_id}.mp4"
    filepath = get_assets_dir() / filename
    video.video.save(str(filepath))

    job.result_filename = filename
    job.result_url = f"{settings.base_url}/assets/{filename}"
    job.progress = 100


def _map_aspect_ratio(ratio: str) -> str:
    """Map our aspect ratio values to Veo's expected format."""
    mapping = {"16:9": "16:9", "9:16": "9:16", "1:1": "16:9"}
    return mapping.get(ratio, "16:9")
