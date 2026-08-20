import os

from dotenv import load_dotenv

load_dotenv()

CORPUS_DIR = os.getenv("CORPUS_DIR", "./corpus")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hhgoa_docs")

MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# STT - saaras:v3 with "unknown" triggers automatic language detection
# across 23 languages (22 Indian + English)
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_LANGUAGE_CODE = os.getenv("SARVAM_LANGUAGE_CODE", "unknown")

# TTS - bulbul:v3, natural Indian-accent voices
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "priya")   # valid for bulbul:v3
SARVAM_TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "2"))
TOP_K = int(os.getenv("TOP_K", "3"))
