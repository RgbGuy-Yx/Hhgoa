# Hhgoa

A FastAPI-based voice RAG application for querying the project corpus using audio input and retrieval-augmented generation.

## Requirements

- Python 3.10+
- A Mistral API key
- A Sarvam AI API key

## 1) Create and activate a virtual environment

On Windows PowerShell:

```powershell
cd D:\hhgoa
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## 3) Configure environment variables

Create a `.env` file in the project root with the following values:

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key

CORPUS_DIR=./corpus
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=hhgoa_docs
MISTRAL_EMBED_MODEL=mistral-embed
MISTRAL_CHAT_MODEL=mistral-small-latest
SARVAM_STT_MODEL=saaras:v3
SARVAM_LANGUAGE_CODE=en-IN
WINDOW_SIZE=2
TOP_K=3
```

## 4) Build the vector database

This loads the Markdown corpus and stores chunk embeddings in Chroma:

```powershell
python ingest.py
```

## 5) Start the application

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## 6) Test the endpoint

You can send an audio file to the `/ask` endpoint:

```powershell
curl -X POST "http://127.0.0.1:8000/ask" -F "audio=@sample.wav"
```

## Notes

- The app expects audio input in a format supported by the STT provider.
- The corpus is read from the `corpus/` directory by default.
- If you want to replay the ingestion step after modifying files, run `python ingest.py` again.
