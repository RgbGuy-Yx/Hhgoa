"""
stt.py — Sarvam AI speech utilities

STT : saaras:v3  — language_code="unknown" auto-detects 23 languages
TTS : bulbul:v3  — returns base64 WAV via response.audios[0]

SDK call signatures (verified against installed sarvamai package):
  client.speech_to_text.transcribe(file, model, language_code)
  client.text_to_speech.convert(text, language_code, speaker, model)
"""

import re
from sarvamai import SarvamAI

import config

_sarvam_client = None


def get_sarvam_client() -> SarvamAI:
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    return _sarvam_client


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

# Maps Unicode script ranges to BCP-47 language codes supported by Bulbul v3
_SCRIPT_LANG_MAP = [
    (re.compile(r"[\u0900-\u097F]"), "hi-IN"),   # Devanagari → Hindi
    (re.compile(r"[\u0980-\u09FF]"), "bn-IN"),   # Bengali
    (re.compile(r"[\u0A80-\u0AFF]"), "gu-IN"),   # Gujarati
    (re.compile(r"[\u0C80-\u0CFF]"), "kn-IN"),   # Kannada
    (re.compile(r"[\u0D00-\u0D7F]"), "ml-IN"),   # Malayalam
    (re.compile(r"[\u0900-\u097F]"), "mr-IN"),   # Marathi (also Devanagari — hi-IN wins for now)
    (re.compile(r"[\u0B00-\u0B7F]"), "od-IN"),   # Odia
    (re.compile(r"[\u0A00-\u0A7F]"), "pa-IN"),   # Gurmukhi → Punjabi
    (re.compile(r"[\u0B80-\u0BFF]"), "ta-IN"),   # Tamil
    (re.compile(r"[\u0C00-\u0C7F]"), "te-IN"),   # Telugu
]

_BULBUL_LANG_CODES = {
    "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN",
    "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN",
}


def detect_tts_language(text: str) -> str:
    """
    Detect the BCP-47 language code for TTS from the text content.
    Falls back to config.SARVAM_TTS_LANGUAGE (en-IN by default).
    Only returns codes supported by bulbul:v3.
    """
    for pattern, lang_code in _SCRIPT_LANG_MAP:
        if pattern.search(text):
            return lang_code
    return config.SARVAM_TTS_LANGUAGE  # default: en-IN


# ── Text-to-Speech ────────────────────────────────────────────────────────────

def text_to_speech_b64(text: str, language_code: str | None = None) -> str:
    """
    Convert text → Bulbul v3 audio.
    Returns the base64-encoded WAV string directly from response.audios[0].
    Auto-detects language from script if language_code is not provided.
    """
    lang = language_code or detect_tts_language(text)
    client = get_sarvam_client()
    response = client.text_to_speech.convert(
        text=text,
        language_code=lang,
        speaker=config.SARVAM_TTS_SPEAKER,
        model=config.SARVAM_TTS_MODEL,
    )
    return response.audios[0]


def text_to_speech(text: str, language_code: str | None = None) -> bytes:
    """
    Same as text_to_speech_b64() but decodes and returns raw WAV bytes.
    Useful for streaming as audio/wav or playing via sounddevice.
    """
    import base64
    return base64.b64decode(text_to_speech_b64(text, language_code))
