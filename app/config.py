from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv(
        "PINECONE_INDEX_NAME", "ai-video-assistant"
    )
    redis_url: str = os.getenv("REDIS_URL", "")
    yt_cookies_path: str | None = os.getenv("YTDLP_COOKIES_FILE") or None
    yt_cookies_content: str | None = os.getenv("YT_COOKIES_CONTENT") or None
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "7200"))
    cleanup_cooldown_seconds: int = int(
        os.getenv("CLEANUP_COOLDOWN_SECONDS", "3600")
    )
    mistral_chat_model: str = os.getenv(
        "MISTRAL_CHAT_MODEL", "mistral-small-latest"
    )
    mistral_embed_model: str = os.getenv(
        "MISTRAL_EMBED_MODEL", "mistral-embed"
    )
    groq_transcript_chunk_seconds: int = int(
        os.getenv("GROQ_TRANSCRIPT_CHUNK_SECONDS", "600")
    )
    sarvam_transcript_chunk_seconds: int = int(
        os.getenv("SARVAM_TRANSCRIPT_CHUNK_SECONDS", "24")
    )
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_access_token_expire_minutes: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    mongodb_url: str = os.getenv("MONGODB_URL", "")  


settings = Settings()
