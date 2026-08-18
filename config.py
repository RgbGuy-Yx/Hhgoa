import os

from dotenv import load_dotenv

load_dotenv()

CORPUS_DIR = os.getenv("CORPUS_DIR", "./corpus")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "hhgoa_docs")

MISTRAL_EMBED_MODEL = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
MISTRAL_CHAT_MODEL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_LANGUAGE_CODE = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN")

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "2"))
TOP_K = int(os.getenv("TOP_K", "3"))
