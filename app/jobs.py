from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from app.config import settings
from app.models import AssetType, GenerateRequest, Job, JobStatus

logger = logging.getLogger(__name__)

# In-memory job store (use Redis for production)
_jobs: dict[str, Job] = {}


def create_job(request: GenerateRequest) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id=job_id, asset_type=request.type, request=request)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def get_assets_dir() -> Path:
    path = Path(settings.assets_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


async def run_job(job: Job) -> None:
    """Execute asset generation in the background."""
    job.status = JobStatus.processing
    job.progress = 10

    try:
        if job.asset_type == AssetType.image:
            from app.generators.image import generate_image

            await generate_image(job)
        elif job.asset_type == AssetType.video:
            from app.generators.video import generate_video

            await generate_video(job)
        elif job.asset_type == AssetType.audio:
            from app.generators.audio import generate_audio

            await generate_audio(job)
        else:
            raise ValueError(f"Unknown asset type: {job.asset_type}")

        job.status = JobStatus.completed
        job.progress = 100
        logger.info("Job %s completed: %s", job.job_id, job.result_url)

    except Exception:
        job.status = JobStatus.failed
        job.progress = 0
        logger.exception("Job %s failed", job.job_id)
        job.error_message = "Asset generation failed. Please try again."
