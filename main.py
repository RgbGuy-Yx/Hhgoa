import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile

from agent import run_agent
from stt import transcribe_audio

app = FastAPI(title="Hacker House Goa 2026 - Voice RAG (Phase 1)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
    finally:
        os.remove(tmp_path)

    result = run_agent(transcript)
    return {"transcript": transcript, **result}
