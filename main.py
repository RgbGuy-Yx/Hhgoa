import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from agent import run_agent
from stt import transcribe_audio, text_to_speech, text_to_speech_b64

app = FastAPI(title="Hacker House Goa 2026 - Voice RAG")


@app.on_event("startup")
async def startup():
    print("\n✅  HHGoa backend running at http://0.0.0.0:8000")
    print("   Health check → http://localhost:8000/health\n")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",   # in case Next.js picks a different port
        "http://127.0.0.1:3001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# ── voice endpoint (audio upload → STT → RAG → TTS) ─────────────────────────
@app.post("/ask")
async def ask(
    audio: UploadFile = File(...),
    tts: bool = Query(default=True, description="Include base64 TTS audio in response"),
):
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
    finally:
        os.remove(tmp_path)

    result = run_agent(transcript)
    answer = result.get("answer", "")

    audio_b64 = None
    if tts and answer:
        try:
            audio_b64 = text_to_speech_b64(answer)
        except Exception:
            pass  # TTS failure is non-fatal - client falls back to text

    return {"transcript": transcript, **result, "audio_b64": audio_b64}


# ── text endpoint (typed question → RAG → TTS) ───────────────────────────────
class TextQuery(BaseModel):
    question: str
    tts: bool = True  # whether to include TTS audio in response


@app.post("/ask-text")
async def ask_text(body: TextQuery):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = run_agent(body.question.strip())
    answer = result.get("answer", "")

    audio_b64 = None
    if body.tts and answer:
        try:
            audio_b64 = text_to_speech_b64(answer)
        except Exception:
            pass

    return {"transcript": body.question.strip(), **result, "audio_b64": audio_b64}


# ── standalone TTS endpoint (for custom text) ────────────────────────────────
class TTSRequest(BaseModel):
    text: str


@app.post("/speak")
async def speak(body: TTSRequest):
    """Returns raw WAV bytes - browser can play with new Audio(url)."""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    try:
        wav_bytes = text_to_speech(body.text.strip())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS error: {e}")
    return Response(content=wav_bytes, media_type="audio/wav")
