# AudioLens AI

An AI video and audio assistant built with plain HTML, CSS, JavaScript, and FastAPI.

## Workflow

1. Import a YouTube video, upload audio, or record with the microphone.
2. Select English/Hinglish or Hindi manually.
3. English/Hinglish uses Groq Whisper with `whisper-large-v3`.
4. Hindi uses Sarvam Saaras v3 in `codemix` mode.
5. Review and edit the merged transcript.
6. Save changes, then explicitly create the chat workspace.
7. Mistral generates the title, summary, embeddings, and grounded answers.
8. Pinecone stores vectors under the browser-generated session namespace.
9. Redis applies a two-hour sliding TTL and expired namespaces are removed.

## Local setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`.

## Required services

- Groq API key
- Sarvam API key
- Mistral API key
- Pinecone Starter index
- Redis URL, such as an Upstash free database

The Pinecone index dimension must match the output dimension of `mistral-embed`
(normally 1024) and use cosine similarity.

## Render

Create a Render Blueprint from `render.yaml`, then add all secret environment
variables from `.env.example` in the Render dashboard.

Render has an ephemeral filesystem. Audio is therefore kept only in a temporary
directory and removed immediately after transcription.

## Notes

- YouTube availability depends on YouTube and `yt-dlp`; some videos may require
  cookies, which are intentionally not part of this first version.
- Free services can sleep, throttle, or change quotas.
- English/Hinglish uses longer Groq chunks; Hindi uses shorter Sarvam-safe chunks.
- In-memory session storage is available for local development only. Use Redis
  in deployment so sessions survive across requests and cleanup remains reliable.
