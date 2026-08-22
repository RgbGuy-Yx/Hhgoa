"""
stt.py — Sarvam AI speech utilities

STT : saaras:v3  — language_code="unknown" auto-detects 23 languages
TTS : bulbul:v3  — returns base64 WAV via response.audios[0]

SDK call signatures:
  client.speech_to_text.transcribe(file, model, language_code)
  client.text_to_speech.convert(text, language_code, speaker, model)
"""

import base64
import re
from typing import Optional
from sarvamai import SarvamAI

import config

_sarvam_client = None


def get_sarvam_client() -> SarvamAI:
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    return _sarvam_client


# ── Text Cleaning for Natural Speech ──────────────────────────────────────────

def clean_for_speech(text: str, max_chars: int = 420) -> str:
    """
    Clean markdown symbols, hashtags, URLs, and formatting tags
    so TTS speaks natural, fluid human sentences instead of
    pronouncing markdown characters (#, *, `, _, [, ], etc.).
    Also keeps spoken length crisp for sub-300ms TTS synthesis.
    """
    if not text:
        return ""

    # 1. Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", text)
    # 2. Remove inline code backticks
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # 3. Replace markdown links [text](url) with just text
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    # 4. Remove raw URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # 5. Remove markdown headers (# Title, ## Subtitle, etc.)
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    # 6. Remove bold & italics (**bold**, *italic*, __bold__, _italic_)
    cleaned = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", cleaned)
    # 7. Convert hashtags like #RAGInGoa -> RAG in Goa or plain text
    cleaned = re.sub(r"#RAGInGoa\b", "RAG In Goa", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"#HHGoa\b", "Hacker House Goa", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"#([a-zA-Z0-9_]+)", r"\1", cleaned)
    # 8. Clean list bullets (- item, * item, 1. item) into natural pauses
    cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+\.\s+", "", cleaned, flags=re.MULTILINE)
    # 9. Remove blockquotes and horizontal rules
    cleaned = re.sub(r"^\s*>\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[-*_]{3,}", "", cleaned)
    # 10. Remove extra brackets, pipes, special syntax characters
    cleaned = re.sub(r"[|~]", " ", cleaned)
    # 11. Normalize whitespace & linebreaks into natural spoken pauses
    cleaned = re.sub(r"\n+", ". ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # 12. Fix multiple dots or punctuation glitches
    cleaned = re.sub(r"\.+", ".", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)

    # Trim to natural sentence boundary within max_chars for fast synthesis
    if len(cleaned) > max_chars:
        cutoff = cleaned[:max_chars].rfind(".")
        if cutoff > 120:
            cleaned = cleaned[:cutoff + 1]
        else:
            cleaned = cleaned[:max_chars].rstrip() + "."

    return cleaned


# ── Speech-to-Text ────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using Sarvam saaras:v3.
    language_code="unknown" triggers automatic language detection across
    22 Indian languages + English.
    """
    client = get_sarvam_client()
    with open(audio_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model=config.SARVAM_STT_MODEL,
            language_code=config.SARVAM_LANGUAGE_CODE,
        )
    return response.transcript


# ── Language detection helper ─────────────────────────────────────────────────

_SCRIPT_LANG_MAP = [
    (re.compile(r"[\u0900-\u097F]"), "hi-IN"),   # Devanagari → Hindi
    (re.compile(r"[\u0980-\u09FF]"), "bn-IN"),   # Bengali
    (re.compile(r"[\u0A80-\u0AFF]"), "gu-IN"),   # Gujarati
    (re.compile(r"[\u0C80-\u0CFF]"), "kn-IN"),   # Kannada
    (re.compile(r"[\u0D00-\u0D7F]"), "ml-IN"),   # Malayalam
    (re.compile(r"[\u0900-\u097F]"), "mr-IN"),   # Marathi
    (re.compile(r"[\u0B00-\u0B7F]"), "od-IN"),   # Odia
    (re.compile(r"[\u0A00-\u0A7F]"), "pa-IN"),   # Gurmukhi → Punjabi
    (re.compile(r"[\u0B80-\u0BFF]"), "ta-IN"),   # Tamil
    (re.compile(r"[\u0C00-\u0C7F]"), "te-IN"),   # Telugu
]


def detect_tts_language(text: str) -> str:
    """
    Detect the BCP-47 language code for TTS from the text content.
    Falls back to config.SARVAM_TTS_LANGUAGE (en-IN by default).
    """
    for pattern, lang_code in _SCRIPT_LANG_MAP:
        if pattern.search(text):
            return lang_code
    return config.SARVAM_TTS_LANGUAGE


# ── Text-to-Speech ────────────────────────────────────────────────────────────

def text_to_speech_b64(text: str, language_code: Optional[str] = None) -> str:
    """
    Convert text → Bulbul v3 audio.
    Cleans markdown/hashtags to ensure clean pronunciation.
    Returns the base64-encoded WAV string directly from response.audios[0].
    """
    spoken_text = clean_for_speech(text)
    if not spoken_text:
        spoken_text = text[:150]

    lang = language_code or detect_tts_language(spoken_text)
    client = get_sarvam_client()
    response = client.text_to_speech.convert(
        text=spoken_text,
        language_code=lang,
        speaker=config.SARVAM_TTS_SPEAKER,
        model=config.SARVAM_TTS_MODEL,
    )
    return response.audios[0]


def text_to_speech(text: str, language_code: Optional[str] = None) -> bytes:
    """
    Same as text_to_speech_b64() but decodes and returns raw WAV bytes.
    """
    b64 = text_to_speech_b64(text, language_code)
    return base64.b64decode(b64)
