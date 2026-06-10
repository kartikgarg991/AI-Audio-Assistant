from __future__ import annotations

import json
import re
from typing import Any

import requests
from pinecone import Pinecone

from app.config import settings


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def _mistral_headers() -> dict[str, str]:
    if not settings.mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY is not configured.")
    return {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json",
    }


def chat_completion(messages: list[dict[str, str]]) -> str:
    response = requests.post(
        f"{MISTRAL_BASE_URL}/chat/completions",
        headers=_mistral_headers(),
        json={
            "model": settings.mistral_chat_model,
            "messages": messages,
            "temperature": 0.2,
        },
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"Mistral chat failed: {response.text[:500]}")
    return response.json()["choices"][0]["message"]["content"].strip()


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = requests.post(
        f"{MISTRAL_BASE_URL}/embeddings",
        headers=_mistral_headers(),
        json={"model": settings.mistral_embed_model, "input": texts},
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(f"Mistral embeddings failed: {response.text[:500]}")
    return [item["embedding"] for item in response.json()["data"]]


def summarize_transcript(transcript: str) -> dict[str, str]:
    transcript_parts = split_text(transcript, size=9000, overlap=0)
    if len(transcript_parts) > 1:
        partial_summaries = [
            chat_completion(
                [
                    {
                        "role": "system",
                        "content": "Summarize transcript sections faithfully and concisely.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Summarize this transcript section using only its content:\n\n"
                            f"{part}"
                        ),
                    },
                ]
            )
            for part in transcript_parts
        ]
        source_text = "\n\n".join(partial_summaries)
    else:
        source_text = transcript

    prompt = (
        "Return strict JSON with keys title and summary. Create a concise title "
        "and a faithful summary using only this transcript. Do not add facts.\n\n"
        f"TRANSCRIPT OR SECTION SUMMARIES:\n{source_text}"
    )
    raw = chat_completion(
        [
            {"role": "system", "content": "You produce accurate transcript metadata."},
            {"role": "user", "content": prompt},
        ]
    )
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "title": str(data.get("title", "Untitled recording")),
                "summary": str(data.get("summary", "")),
            }
        except json.JSONDecodeError:
            pass
    return {"title": "Untitled recording", "summary": raw}


def split_text(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        if current and current_length + len(word) + 1 > size:
            chunk = " ".join(current)
            chunks.append(chunk)
            overlap_words: list[str] = []
            overlap_length = 0
            for old_word in reversed(current):
                if overlap_length + len(old_word) + 1 > overlap:
                    break
                overlap_words.insert(0, old_word)
                overlap_length += len(old_word) + 1
            current = overlap_words
            current_length = overlap_length
        current.append(word)
        current_length += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def pinecone_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")
    return Pinecone(api_key=settings.pinecone_api_key).Index(
        settings.pinecone_index_name
    )


def index_transcript(
    session_id: str,
    audio_id: str,
    transcript: str,
) -> int:
    chunks = split_text(transcript)
    total_vectors = 0
    batch_size = 50
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = embed_texts(batch_chunks)
        
        payload = [
            {
                "id": f"{audio_id}:{i + index}",
                "values": vector,
                "metadata": {
                    "audio_id": audio_id,
                    "chunk_id": i + index,
                    "text": batch_chunks[index],
                },
            }
            for index, vector in enumerate(batch_vectors)
        ]
        pinecone_index().upsert(vectors=payload, namespace=session_id)
        total_vectors += len(payload)
        
    return total_vectors


def answer_question(
    session_id: str,
    audio_id: str,
    question: str,
    history: list[dict[str, str]],
) -> str:
    vector = embed_texts([question])[0]
    results = pinecone_index().query(
        namespace=session_id,
        vector=vector,
        top_k=6,
        include_metadata=True,
        filter={"audio_id": {"$eq": audio_id}},
    )
    context = "\n\n---\n\n".join(
        match["metadata"]["text"]
        for match in results.get("matches", [])
        if match.get("metadata", {}).get("text")
    )
    if not context:
        return "I could not find relevant information in this transcript."

    previous = "\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in history[-5:]
    )
    return chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied transcript context. If the "
                    "answer is absent, say that clearly. Never invent details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous conversation:\n{previous}\n\n"
                    f"Question: {question}\n\nTranscript context:\n{context}"
                ),
            },
        ]
    )


def delete_namespace(session_id: str) -> None:
    if not settings.pinecone_api_key:
        return
    pinecone_index().delete(delete_all=True, namespace=session_id)
