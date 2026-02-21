"""Audio ad generation using ElevenLabs Text-to-Dialogue."""

from __future__ import annotations

import logging

import httpx

from app.config import settings
from app.generators.creative import generate_creative_prompt
from app.jobs import get_assets_dir
from app.models import Job

logger = logging.getLogger(__name__)

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"

# Default voice IDs (ElevenLabs pre-made voices)
SPEAKER_1_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel
SPEAKER_2_VOICE = "29vD33N1CtxCmqQRPOHJ"  # Drew


async def generate_audio(job: Job) -> None:
    """Generate a conversational audio ad using ElevenLabs."""
    job.progress = 10

    # Step 1: Generate conversational script via LLM creative director
    script = await generate_creative_prompt(job.request)
    job.progress = 30

    # Step 2: Parse script into speaker segments
    segments = _parse_script(script)
    if not segments:
        raise RuntimeError("Failed to parse audio script into speaker segments")

    job.progress = 40

    # Step 3: Generate audio for each segment and concatenate
    audio_chunks: list[bytes] = []
    total_segments = len(segments)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, (speaker, text) in enumerate(segments):
            voice_id = SPEAKER_1_VOICE if speaker == 1 else SPEAKER_2_VOICE
            chunk = await _synthesize_speech(client, voice_id, text)
            audio_chunks.append(chunk)
            job.progress = 40 + int(50 * (i + 1) / total_segments)

    # Step 4: Concatenate audio chunks (simple MP3 concatenation)
    filename = f"{job.job_id}.mp3"
    filepath = get_assets_dir() / filename

    with open(filepath, "wb") as f:
        for chunk in audio_chunks:
            f.write(chunk)

    job.result_filename = filename
    job.result_url = f"{settings.base_url}/assets/{filename}"
    job.progress = 100


def _parse_script(script: str) -> list[tuple[int, str]]:
    """Parse a script into (speaker_number, text) tuples."""
    segments: list[tuple[int, str]] = []
    for line in script.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("speaker 1:") or lower.startswith("speaker1:"):
            text = line.split(":", 1)[1].strip()
            segments.append((1, text))
        elif lower.startswith("speaker 2:") or lower.startswith("speaker2:"):
            text = line.split(":", 1)[1].strip()
            segments.append((2, text))
        elif lower.startswith("s1:"):
            text = line.split(":", 1)[1].strip()
            segments.append((1, text))
        elif lower.startswith("s2:"):
            text = line.split(":", 1)[1].strip()
            segments.append((2, text))
    return segments


async def _synthesize_speech(client: httpx.AsyncClient, voice_id: str, text: str) -> bytes:
    """Synthesize speech for a single text segment using ElevenLabs."""
    response = await client.post(
        f"{ELEVENLABS_TTS_URL}/{voice_id}",
        headers={
            "xi-api-key": settings.elevenlabs_api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_turbo_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.3,
            },
        },
    )
    response.raise_for_status()
    return response.content
