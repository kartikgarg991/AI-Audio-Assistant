from __future__ import annotations

import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai import (
    answer_question,
    delete_namespace,
    index_transcript,
    summarize_transcript,
)
from app.audio import cleanup_workdir, copy_upload_to_temp, download_youtube_audio, split_audio
from app.config import settings
from app.session_store import (
    append_log,
    clear_session,
    clear_logs,
    get_expired_sessions,
    get_json,
    get_logs,
    mark_cleanup_if_due,
    set_json,
    touch_session,
)
from app.transcription import transcribe_chunks


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
app = FastAPI(title="AI Video Assistant")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# @app.on_event("startup")
# def _materialize_yt_cookies() -> None:
#     if settings.yt_cookies_content:
#         cookie_path = Path("/tmp/yt_cookies.txt")
#         cookie_path.write_text(settings.yt_cookies_content)
#         object.__setattr__(settings, "yt_cookies_path", str(cookie_path))

@app.on_event("startup")
def _materialize_yt_cookies() -> None:
    if settings.yt_cookies_path and Path(settings.yt_cookies_path).exists():
        return
    if settings.yt_cookies_content:
        cookie_path = Path(tempfile.gettempdir()) / "yt_cookies.txt"
        content = settings.yt_cookies_content.strip()
        if "\\n" in content and "\n" not in content:
            content = content.replace("\\n", "\n")
        if not content.startswith("# Netscape HTTP Cookie File"):
            content = "# Netscape HTTP Cookie File\n" + content
        cookie_path.write_text(content, encoding="utf-8")
        object.__setattr__(settings, "yt_cookies_path", str(cookie_path))


@app.middleware("http")
async def cleanup_middleware(request, call_next):
    schedule_cleanup()
    return await call_next(request)


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=100)


class YouTubeRequest(SessionRequest):
    url: str
    language_mode: str


class TranscriptUpdate(SessionRequest):
    audio_id: str
    transcript: str = Field(min_length=1)


class CreateChatRequest(TranscriptUpdate):
    pass


class AskRequest(SessionRequest):
    audio_id: str
    question: str = Field(min_length=1, max_length=4000)


def _validate_language(language_mode: str) -> None:
    if language_mode not in {"english", "hinglish", "hindi"}:
        raise HTTPException(400, "Choose English, Hinglish, or Hindi.")


def _process_audio(
    source: Path,
    workdir: Path,
    session_id: str,
    audio_id: str,
    source_type: str,
    language_mode: str,
) -> dict:
    try:
        append_log(session_id, f"Starting {source_type} processing")
        chunk_seconds = (
            settings.sarvam_transcript_chunk_seconds
            if language_mode == "hindi"
            else settings.groq_transcript_chunk_seconds
        )
        append_log(session_id, f"Normalizing audio and creating {chunk_seconds}-second chunks")
        chunks = split_audio(source, workdir / "chunks", chunk_seconds=chunk_seconds)
        append_log(session_id, f"Created {len(chunks)} audio chunk(s)")
        result = transcribe_chunks(
            chunks,
            language_mode,
            log=lambda message, level="info": append_log(session_id, message, level),
        )
        append_log(session_id, "Transcript merge complete")
        payload = {
            "audio_id": audio_id,
            "source_type": source_type,
            "language_mode": language_mode,
            **result,
        }
        set_json("transcript", session_id, payload)
        return payload
    finally:
        append_log(session_id, "Cleaning temporary audio files")
        cleanup_workdir(workdir)


def cleanup_expired_sessions() -> None:
    for session_id in get_expired_sessions():
        try:
            delete_namespace(session_id)
        except Exception as exc:
            print(f"Cleanup failed for {session_id}: {exc}")
        finally:
            clear_session(session_id)


def schedule_cleanup() -> None:
    if mark_cleanup_if_due():
        threading.Thread(target=cleanup_expired_sessions, daemon=True).start()


@app.get("/")
def home():
    schedule_cleanup()
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/session/touch")
def session_touch(payload: SessionRequest):
    return {"expires_at": touch_session(payload.session_id)}


@app.get("/api/session/{session_id}/logs")
def session_logs(session_id: str):
    return {"logs": get_logs(session_id)}


@app.post("/api/transcribe/upload")
def transcribe_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    language_mode: str = Form(...),
    source_type: str = Form("upload"),
):
    _validate_language(language_mode)
    if source_type not in {"upload", "microphone"}:
        raise HTTPException(400, "Invalid source type.")
    clear_logs(session_id)
    append_log(session_id, f"Received {source_type} file: {file.filename or 'recording'}")
    contents = file.file.read()
    if not contents:
        append_log(session_id, "Uploaded audio was empty", "error")
        raise HTTPException(400, "The uploaded audio is empty.")
    append_log(session_id, f"Upload size: {len(contents)} bytes")

    audio_id = str(uuid.uuid4())
    suffix = Path(file.filename or "audio.webm").suffix
    workdir, source = copy_upload_to_temp(contents, suffix)
    touch_session(session_id)
    try:
        return _process_audio(
            source,
            workdir,
            session_id,
            audio_id,
            source_type,
            language_mode,
        )
    except Exception as exc:
        append_log(session_id, f"Failed: {exc}", "error")
        cleanup_workdir(workdir)
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/transcribe/youtube")
def transcribe_youtube(payload: YouTubeRequest):
    _validate_language(payload.language_mode)
    clear_logs(payload.session_id)
    append_log(payload.session_id, f"Received YouTube URL: {payload.url}")
    audio_id = str(uuid.uuid4())
    workdir = Path(tempfile.mkdtemp(prefix="ai-video-assistant-youtube-"))
    touch_session(payload.session_id)
    try:
        append_log(payload.session_id, "Downloading YouTube audio")
        source = download_youtube_audio(
            payload.url,
            workdir,
            cookies_file=settings.yt_cookies_path,
        )
        append_log(payload.session_id, f"YouTube audio downloaded: {source.name}")
        return _process_audio(
            source,
            workdir,
            payload.session_id,
            audio_id,
            "youtube",
            payload.language_mode,
        )
    except Exception as exc:
        append_log(payload.session_id, f"Failed: {exc}", "error")
        cleanup_workdir(workdir)
        raise HTTPException(502, str(exc)) from exc


@app.put("/api/transcript")
def save_transcript(payload: TranscriptUpdate):
    existing = get_json("transcript", payload.session_id)
    if not existing or existing.get("audio_id") != payload.audio_id:
        raise HTTPException(404, "Transcript session was not found or expired.")
    existing["text"] = payload.transcript.strip()
    existing["edited"] = True
    set_json("transcript", payload.session_id, existing)
    return {"saved": True}


@app.post("/api/chat/create")
def create_chat(payload: CreateChatRequest):
    existing = get_json("transcript", payload.session_id)
    if not existing or existing.get("audio_id") != payload.audio_id:
        raise HTTPException(404, "Transcript session was not found or expired.")

    transcript = payload.transcript.strip()
    existing["text"] = transcript
    set_json("transcript", payload.session_id, existing)
    append_log(payload.session_id, "Generating title and summary with Mistral")
    metadata = summarize_transcript(transcript)
    append_log(payload.session_id, "Creating Mistral embeddings and Pinecone vectors")
    chunk_count = index_transcript(
        payload.session_id,
        payload.audio_id,
        transcript,
    )
    result = {**metadata, "audio_id": payload.audio_id, "chunk_count": chunk_count}
    set_json("result", payload.session_id, result)
    set_json("chat", payload.session_id, [])
    append_log(payload.session_id, f"Chat index ready with {chunk_count} chunk(s)")
    return result


@app.post("/api/chat/ask")
def ask(payload: AskRequest):
    result = get_json("result", payload.session_id)
    if not result or result.get("audio_id") != payload.audio_id:
        raise HTTPException(409, "Create the chat index before asking questions.")
    history = get_json("chat", payload.session_id, [])
    answer = answer_question(
        payload.session_id,
        payload.audio_id,
        payload.question,
        history,
    )
    history.append({"user": payload.question, "assistant": answer})
    set_json("chat", payload.session_id, history[-5:])
    return {"answer": answer}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(delete_namespace, session_id)
    clear_session(session_id)
    return {"deleted": True}
