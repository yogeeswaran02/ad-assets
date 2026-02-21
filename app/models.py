from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class AssetType(str, enum.Enum):
    image = "image"
    video = "video"
    audio = "audio"


class Tone(str, enum.Enum):
    professional = "professional"
    casual = "casual"
    energetic = "energetic"
    luxury = "luxury"
    humorous = "humorous"
    inspirational = "inspirational"


class AspectRatio(str, enum.Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"


class JobStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class GenerateRequest(BaseModel):
    """Request to generate an ad asset."""

    type: AssetType = Field(description="Type of ad asset to generate: image, video, or audio")
    product_name: str = Field(description="Name of the product or brand being advertised")
    product_description: str = Field(
        description="Brief description of the product, its key features and benefits"
    )
    target_audience: str = Field(
        default="general consumers",
        description="Target audience for the ad (e.g. 'young professionals aged 25-35')",
    )
    tone: Tone = Field(
        default=Tone.professional,
        description="Desired tone/mood for the ad asset",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.landscape,
        description="Aspect ratio for image or video assets",
    )
    additional_instructions: str = Field(
        default="",
        description="Any additional creative direction or specific requirements",
    )


class JobResponse(BaseModel):
    """Response representing a generation job."""

    job_id: str = Field(description="Unique identifier for the generation job")
    status: JobStatus = Field(description="Current status of the job")
    type: AssetType = Field(description="Type of asset being generated")
    progress: int = Field(
        default=0,
        description="Estimated progress percentage (0-100)",
    )
    result_url: str | None = Field(
        default=None,
        description="URL to download the generated asset, available when status is completed",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if the job failed",
    )
    created_at: str = Field(description="ISO 8601 timestamp when the job was created")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service health status")
    version: str = Field(description="API version")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message describing what went wrong")


class Job:
    """Internal job state."""

    def __init__(self, job_id: str, asset_type: AssetType, request: GenerateRequest) -> None:
        self.job_id = job_id
        self.asset_type = asset_type
        self.request = request
        self.status = JobStatus.queued
        self.progress = 0
        self.result_url: str | None = None
        self.result_filename: str | None = None
        self.error_message: str | None = None
        self.created_at = datetime.utcnow()

    def to_response(self) -> JobResponse:
        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            type=self.asset_type,
            progress=self.progress,
            result_url=self.result_url,
            error_message=self.error_message,
            created_at=self.created_at.isoformat() + "Z",
        )
