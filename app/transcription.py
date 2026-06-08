from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import groq
import requests

from app.audio import AudioChunk
from app.config import settings


def _request_with_retry(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, timeout=120, **kwargs)
            if response.status_code not in (429, 500, 502, 503, 504):
                return response
            last_error = RuntimeError(
                f"Provider returned {response.status_code}: {response.text[:300]}"
            )
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"Transcription provider failed: {last_error}")


def transcribe_groq(chunk: Path) -> dict[str, Any]:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = groq.Groq(api_key=settings.groq_api_key)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with chunk.open("rb") as audio:
                text = client.audio.transcriptions.create(
                    file=(chunk.name, audio.read()),
                    model="whisper-large-v3",
                    response_format="text",
                )
            if not str(text or "").strip():
                raise RuntimeError("Groq returned an empty transcript.")
            return {
                "text": str(text or "").strip(),
                "segments": [],
                "provider": "Groq Whisper (whisper-large-v3)",
            }
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Groq transcription failed: {last_error}")


def transcribe_whisper(chunk: Path) -> dict[str, Any]:
    result = transcribe_groq(chunk)
    return {
        "text": result["text"],
        "segments": [],
        "provider": result["provider"],
    }


def transcribe_sarvam(chunk: Path) -> dict[str, Any]:
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is not configured.")
    with chunk.open("rb") as audio:
        response = _request_with_retry(
            "POST",
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": settings.sarvam_api_key},
            files={"file": (chunk.name, audio, "audio/wav")},
            data={
                "model": "saaras:v3",
                "mode": "codemix",
                "language_code": "hi-IN",
                "with_timestamps": "true",
            },
        )
    if not response.ok:
        raise RuntimeError(f"Sarvam transcription failed: {response.text[:500]}")
    data = response.json()
    timestamps = data.get("timestamps") or {}
    words = timestamps.get("words") or []
    starts = timestamps.get("start_time_seconds") or []
    ends = timestamps.get("end_time_seconds") or []
    segments = [
        {"start": starts[i], "end": ends[i], "text": word}
        for i, word in enumerate(words)
        if i < len(starts) and i < len(ends)
    ]
    return {
        "text": (data.get("transcript") or "").strip(),
        "segments": segments,
        "provider": "Sarvam Saaras v3",
    }


def transcribe_chunks(
    chunks: list[AudioChunk],
    language_mode: str,
    log: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    merged_segments: list[dict[str, Any]] = []
    texts: list[str] = []
    providers: list[str] = []
    fallback_used = False

    for index, chunk_info in enumerate(chunks, start=1):
        chunk = chunk_info.path
        offset = chunk_info.offset
        if chunk.stat().st_size <= 1024 or chunk_info.duration < 0.2:
            if log:
                log(f"Skipping empty chunk {index}: {chunk.name}", "warning")
            continue
        if language_mode in {"english", "hinglish"}:
            if log:
                log(f"Chunk {index}/{len(chunks)}: sending to Groq Whisper")
            result = transcribe_whisper(chunk)
        else:
            if log:
                log(f"Chunk {index}/{len(chunks)}: sending to Sarvam")
            result = transcribe_sarvam(chunk)

        providers.append(result["provider"])
        if result["text"]:
            texts.append(result["text"])
            if log:
                log(f"Chunk {index}: received {len(result['text'])} characters from {result['provider']}")
        for segment in result["segments"]:
            text = str(segment.get("text", "")).strip()
            if text:
                merged_segments.append(
                    {
                        "start": round(float(segment.get("start", 0)) + offset, 2),
                        "end": round(float(segment.get("end", 0)) + offset, 2),
                        "text": text,
                    }
                )

    text = " ".join(texts).strip()
    if not text:
        raise RuntimeError(
            "Transcription finished but no text was returned. Check that the "
            "audio contains speech and that the selected language/provider is correct."
        )

    return {
        "text": text,
        "segments": merged_segments,
        "provider": " + ".join(dict.fromkeys(providers)),
        "fallback_used": fallback_used,
    }
