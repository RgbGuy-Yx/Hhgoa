from sarvamai import SarvamAI

import config

_sarvam_client = None


def get_sarvam_client() -> SarvamAI:
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
    return _sarvam_client


def transcribe_audio(audio_path: str) -> str:
    client = get_sarvam_client()
    with open(audio_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model=config.SARVAM_STT_MODEL,
            language_code=config.SARVAM_LANGUAGE_CODE,
        )
    return response.transcript
