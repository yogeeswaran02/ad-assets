"""LLM creative director — transforms product info into optimized generation prompts.

Supports both Gemini and Claude as the underlying LLM. Configured via CREATIVE_LLM
env var ("gemini" or "claude"). Defaults to Gemini since it shares the same API key
used for image/video generation.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.models import AssetType, GenerateRequest

logger = logging.getLogger(__name__)

IMAGE_SYSTEM = """You are an expert advertising creative director specializing in visual ad design.
Given product information, generate a single detailed image generation prompt optimized for AI image generation (Google Gemini / Nano Banana Pro).

Your prompt should include:
- Subject and composition (product placement, angles)
- Setting/background that resonates with the target audience
- Lighting style (studio, natural, dramatic, etc.)
- Visual style (product photography, lifestyle, flat lay, etc.)
- Color palette hints that match the brand tone
- Empty space for text overlay if appropriate

Keep the prompt to 2-3 sentences. Do NOT include any text/copy in the image — just the visual scene.
Output ONLY the prompt text, nothing else."""

VIDEO_SYSTEM = """You are an expert advertising creative director specializing in video ad production.
Given product information, generate a single detailed video generation prompt optimized for Google Veo 3.1.

Your prompt should include:
- Camera movement (tracking, dolly, crane, static, slow zoom, etc.)
- Subject action and motion
- Setting and atmosphere
- Lighting description
- Audio direction (ambient sounds, music mood, optional dialogue)
- Pacing notes

The video will be 8 seconds long. Design the prompt for a single continuous shot.
Keep the prompt to 3-4 sentences.
Output ONLY the prompt text, nothing else."""

AUDIO_SYSTEM = """You are an expert advertising creative director specializing in podcast-style audio ads.
Given product information, write a natural conversational script between two speakers (Speaker 1 and Speaker 2) for a 30-second audio ad.

Requirements:
- 60-80 words total (30 seconds of dialogue)
- Natural, conversational tone — like two friends chatting on a podcast
- Speaker 1 shares their experience with the product
- Speaker 2 asks curious questions and reacts naturally
- Include a clear call-to-action near the end
- Add occasional non-verbal cues in brackets: [laughs], [pauses], [excited]
- Match the requested tone (professional, casual, energetic, etc.)

Format each line as:
Speaker 1: dialogue here
Speaker 2: dialogue here

Output ONLY the script, nothing else."""


def _get_system_prompt(asset_type: AssetType) -> str:
    if asset_type == AssetType.image:
        return IMAGE_SYSTEM
    elif asset_type == AssetType.video:
        return VIDEO_SYSTEM
    elif asset_type == AssetType.audio:
        return AUDIO_SYSTEM
    raise ValueError(f"Unknown asset type: {asset_type}")


def _build_user_message(request: GenerateRequest) -> str:
    parts = [
        f"Product: {request.product_name}",
        f"Description: {request.product_description}",
        f"Target audience: {request.target_audience}",
        f"Tone: {request.tone.value}",
    ]
    if request.aspect_ratio:
        parts.append(f"Aspect ratio: {request.aspect_ratio.value}")
    if request.additional_instructions:
        parts.append(f"Additional direction: {request.additional_instructions}")
    return "\n".join(parts)


async def _generate_with_gemini(system: str, user_message: str) -> str:
    """Generate creative prompt using Google Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.gemini_api_key)

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=500,
            temperature=0.9,
        ),
    )
    return response.text.strip()


async def _generate_with_claude(system: str, user_message: str) -> str:
    """Generate creative prompt using Anthropic Claude."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text.strip()


async def generate_creative_prompt(request: GenerateRequest) -> str:
    """Use an LLM as creative director to generate an optimized asset prompt.

    Provider is controlled by the CREATIVE_LLM setting ("gemini" or "claude").
    Falls back to the other provider if the primary one fails.
    """
    system = _get_system_prompt(request.type)
    user_message = _build_user_message(request)
    primary = settings.creative_llm.lower()

    providers = {
        "gemini": _generate_with_gemini,
        "claude": _generate_with_claude,
    }

    # Try primary provider, fall back to the other
    fallback = "claude" if primary == "gemini" else "gemini"
    for provider_name in [primary, fallback]:
        fn = providers.get(provider_name)
        if fn is None:
            continue
        try:
            prompt = await fn(system, user_message)
            logger.info(
                "Creative prompt via %s for %s [%s]: %s",
                provider_name,
                request.type.value,
                request.product_name,
                prompt,
            )
            return prompt
        except Exception:
            logger.warning("Creative director failed with %s, trying fallback", provider_name, exc_info=True)

    raise RuntimeError("All creative director LLM providers failed")
