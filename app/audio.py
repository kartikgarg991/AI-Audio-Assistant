from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import imageio_ffmpeg
import yt_dlp

from app.config import settings

# Root of the project (one level above this file: app/audio.py → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AudioChunk(NamedTuple):
    path: Path
    offset: float
    duration: float


SAFE_UPLOAD_SUFFIXES = {
    ".aac",
    ".aiff",
    ".bin",
    ".flac",
    ".m4a",
    ".m4b",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


def ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _has_cookie_rows(path: Path) -> bool:
    """Return True only if the cookie file has at least one non-comment data row."""
    try:
        return any(
            line.strip() and not line.startswith("#")
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        )
    except OSError:
        return False


def _resolve_cookies(workdir: Path) -> str | None:
    """
    Priority order (mirrors Sakshi's working approach):
    1. YT_COOKIES_CONTENT env var  → write to workdir/_yt_cookies.txt (Render)
    2. YTDLP_COOKIES_FILE env var  → resolve as absolute path (local dev)
    3. <project_root>/cookies.txt  → absolute fallback for local dev
    Always normalizes \\n → \n and validates the file has real cookie rows.
    """
    # --- Priority 1: env var content (Render deployment) ---
    content = os.getenv("YT_COOKIES_CONTENT", "").strip()
    if content:
        # Always normalize escaped newlines unconditionally
        content = content.replace("\\n", "\n")
        if not content.startswith("# Netscape HTTP Cookie File"):
            content = "# Netscape HTTP Cookie File\n" + content
        cookie_path = workdir / "_yt_cookies.txt"
        cookie_path.write_text(content, encoding="utf-8")
        if _has_cookie_rows(cookie_path):
            print(f"[audio] Using YT_COOKIES_CONTENT env var ({len(content)} bytes)")
            return str(cookie_path)
        print("[audio] YT_COOKIES_CONTENT is set but has no valid cookie rows — skipping")

    # --- Priority 2: explicit file path env var ---
    env_path = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    if env_path:
        # Resolve relative paths against project root, not cwd
        candidate = Path(env_path)
        if not candidate.is_absolute():
            candidate = _PROJECT_ROOT / env_path
        if candidate.exists() and _has_cookie_rows(candidate):
            print(f"[audio] Using cookies file: {candidate}")
            return str(candidate)

    # --- Priority 3: auto-detect cookies.txt at project root ---
    fallback = _PROJECT_ROOT / "cookies.txt"
    if fallback.exists() and _has_cookie_rows(fallback):
        print(f"[audio] Using project-root cookies.txt: {fallback}")
        return str(fallback)

    print("[audio] No valid YouTube cookies found — proceeding without authentication")
    return None


def download_youtube_audio(
    url: str,
    workdir: Path,
    cookies_file: str | None = None,
) -> Path:
    output_template = str(workdir / "youtube.%(ext)s")
    options = {
        "format": "bestaudio[ext=m4a]/bestaudio/best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": True,
        "socket_timeout": 600,
        "retries": 3,
        "fragment_retries": 3,
        "ffmpeg_location": str(Path(ffmpeg_path()).parent),
        "js_runtimes": {"node": {}},
    }
    # Resolve cookies fresh on every call (not just at startup)
    resolved = cookies_file or _resolve_cookies(workdir)
    if resolved:
        options["cookiefile"] = resolved
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=True)
        downloaded = Path(downloader.prepare_filename(info))
    if not downloaded.exists():
        candidates = list(workdir.glob("youtube.*"))
        if not candidates:
            raise RuntimeError("YouTube audio download did not produce a file.")
        downloaded = candidates[0]
    return downloaded


def _probe_duration(path: Path) -> float:
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-i",
        str(path),
        "-f",
        "null",
        "-",
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    marker = "Duration: "
    if marker not in process.stderr:
        return 0.0
    raw = process.stderr.split(marker, 1)[1].split(",", 1)[0].strip()
    parts = raw.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0.0


def split_audio(
    source: Path,
    output_dir: Path,
    chunk_seconds: int | None = None,
) -> list[AudioChunk]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "chunk_%04d.wav"
    segment_seconds = chunk_seconds or settings.sarvam_transcript_chunk_seconds
    command = [
        ffmpeg_path(),
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        "-c:a",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-reset_timestamps",
        "1",
        str(pattern),
    ]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode != 0:
        detail = process.stderr[-500:].strip() or "FFmpeg could not read this file."
        raise RuntimeError(f"Audio conversion failed. Upload a valid audio file that FFmpeg can decode. Details: {detail}")
    chunks: list[AudioChunk] = []
    offset = 0.0
    for chunk in sorted(output_dir.glob("chunk_*.wav")):
        duration = _probe_duration(chunk)
        if chunk.stat().st_size > 1024 and duration >= 0.2:
            chunks.append(AudioChunk(chunk, offset, duration))
        offset += duration or segment_seconds

    if not chunks:
        raise RuntimeError("No audio chunks were created.")
    return chunks


def copy_upload_to_temp(contents: bytes, suffix: str) -> tuple[Path, Path]:
    workdir = Path(tempfile.mkdtemp(prefix="ai-video-assistant-"))
    safe_suffix = suffix.lower() if suffix and suffix.lower() in SAFE_UPLOAD_SUFFIXES else ".bin"
    source = workdir / f"source{safe_suffix}"
    source.write_bytes(contents)
    return workdir, source


def cleanup_workdir(workdir: Path) -> None:
    shutil.rmtree(workdir, ignore_errors=True)
