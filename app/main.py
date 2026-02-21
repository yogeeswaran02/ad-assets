"""Ad Assets API — generates image, video, and audio ad assets for Salesforce AgentForce."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader

from app.config import settings
from app.jobs import create_job, get_assets_dir, get_job, run_job
from app.models import (
    ErrorResponse,
    GenerateRequest,
    HealthResponse,
    JobResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ad Assets API",
    description="Generate AI-powered ad assets (images, videos, audio) for Salesforce AgentForce.",
    version="0.1.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    if not settings.api_key:
        return "no-auth"
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the service health status and version. Use this to verify the API is running.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", version="0.1.0")


@app.post(
    "/generate",
    response_model=JobResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
    summary="Generate an ad asset",
    description=(
        "Submit a request to generate an ad asset. Returns a job ID that can be used "
        "to check the generation status. Supports three asset types: "
        "image (product photography via AI), video (8-second cinematic ad clip), "
        "and audio (30-second conversational podcast-style ad). "
        "The API generates an optimized creative prompt from the product information "
        "and then produces the asset using specialized AI models."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def generate(request: GenerateRequest) -> JobResponse:
    job = create_job(request)
    asyncio.create_task(run_job(job))
    logger.info("Created job %s for %s: %s", job.job_id, request.type.value, request.product_name)
    return job.to_response()


@app.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    responses={404: {"model": ErrorResponse}, 401: {"model": ErrorResponse}},
    summary="Check asset generation status",
    description=(
        "Check the status of an asset generation job. Returns the current status "
        "(queued, processing, completed, or failed), progress percentage, "
        "and the result URL when the asset is ready for download."
    ),
    dependencies=[Depends(verify_api_key)],
)
async def get_job_status(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_response()


@app.get(
    "/assets/{filename}",
    summary="Download a generated asset",
    description="Download a completed ad asset file by its filename.",
    dependencies=[Depends(verify_api_key)],
)
async def get_asset(filename: str) -> FileResponse:
    # Prevent path traversal
    safe_name = Path(filename).name
    filepath = get_assets_dir() / safe_name
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Asset {filename} not found")

    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
    }
    ext = filepath.suffix.lower()
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(filepath, media_type=media_type, filename=safe_name)


def run() -> None:
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)


if __name__ == "__main__":
    run()
