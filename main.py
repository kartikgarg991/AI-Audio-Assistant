import tempfile
import threading
import uuid
from pathlib import Path

from datetime import datetime, timedelta
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Depends, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Request # Add Request to your existing fastapi imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.database import users_collection



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

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Create a Password Hashing Context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Define where FastAPI should look for the token (the login endpoint)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# Helper to check if a plain password matches the hashed password in the DB
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Helper to hash a plain password before saving it to the DB
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Function to generate a signed JWT
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    
    # 1. Calculate the expiration time (current time + 30 minutes)
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    # 2. Add the expiration timestamp to the payload
    to_encode.update({"exp": expire})
    
    # 3. Sign and encode the JWT using settings
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    
    return encoded_jwt

# User data is now stored permanently in MongoDB (see app/database.py)

# 2. Pydantic model for incoming registration data
class UserRegister(BaseModel):
    username: str
    password: str
    full_name: str

# 3. The Registration Endpoint
@app.post("/api/register")
async def register_user(user_data: UserRegister):
    # Check if the user already exists in MongoDB
    existing = await users_collection.find_one({"username": user_data.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash their plain text password securely
    hashed_password = get_password_hash(user_data.password)
    
    # Save the new user permanently to MongoDB
    new_user = {
        "username": user_data.username,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
    }
    await users_collection.insert_one(new_user)
    
    return {"message": "User created successfully! You can now log in."}

@app.post("/api/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Fetch the user from MongoDB
    user = await users_collection.find_one({"username": form_data.username})
    
    if not user:
        raise credentials_exception
    
    # Verify the password
    if not verify_password(form_data.password, user["hashed_password"]):
        raise credentials_exception
    
    # Create the JWT token
    access_token = create_access_token(data={"sub": user["username"]})
    
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "user": {"username": user["username"], "full_name": user["full_name"]}
    }

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the token using our secret key from settings
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        # Catch any errors (like an expired token or tampered signature)
        raise credentials_exception
        
    # Find the user in MongoDB
    user = await users_collection.find_one({"username": username})
    if user is None:
        raise credentials_exception
        
    # If everything is good, let them through and give the endpoint their info!
    return user


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
    for session_id in get_expired_sessions(limit=1000):
        try:
            delete_namespace(session_id)
            clear_session(session_id)
        except Exception as exc:
            error_msg = str(exc).lower()
            if "not exist" in error_msg or "not found" in error_msg:
                clear_session(session_id)
            else:
                print(f"Cleanup failed for {session_id}, will retry later: {exc}")


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
    audio_id = str(uuid.uuid4())
    suffix = Path(file.filename or "audio.webm").suffix
    
    # Stream the file directly to disk to avoid Out-Of-Memory errors
    workdir, source = copy_upload_to_temp(file.file, suffix)
    
    file_size = source.stat().st_size
    if file_size == 0:
        cleanup_workdir(workdir)
        append_log(session_id, "Uploaded audio was empty", "error")
        raise HTTPException(400, "The uploaded audio is empty.")
        
    append_log(session_id, f"Upload saved to disk: {file_size} bytes")
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
            # cookies resolved automatically by _resolve_cookies() in audio.py
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
@limiter.limit("5/minute")  # 👈 This allows 5 requests per minute!
def ask(request: Request, payload: AskRequest, current_user: dict = Depends(get_current_user)):
    # Let's add a quick print statement to prove we know who is asking!
    print(f"🔒 Secure Route Accessed by: {current_user['full_name']}")
    
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
