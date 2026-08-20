"""
voice_assistant.py
------------------
Always-on voice loop for Hacker House Goa 2026.

Wake word : "Hey Goa"  (case-insensitive, substring match)
Flow      : listen (long window) → detect wake word in transcript
            → strip wake word → use remainder as question (or prompt for one)
            → run_agent → Bulbul v3 TTS → play audio → back to listen

STT : Sarvam saaras:v3  (language_code="unknown" → auto-detects 23 languages)
TTS : Sarvam bulbul:v3  (natural Indian-accented voice, replaces pyttsx3)

Every question requires "Hey Goa". After answering the assistant returns to
passive wake-word listening immediately.

Usage:
    python voice_assistant.py

Dependencies:
    sounddevice, soundfile  (already installed)
"""

import io
import re
import os
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf

import config
from agent import run_agent
from stt import transcribe_audio, text_to_speech

# ---------------------------------------------------------------------------
# Audio capture settings
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000        # Hz - Sarvam STT works well at 16 kHz
CHANNELS = 1

# Single recording window - long enough for "Hey Goa <full question>"
LISTEN_SECS = 7.0

# Silence gate - RMS below this skips STT entirely (saves API quota).
# Raise if you get false triggers in a noisy room.
SILENCE_RMS_THRESHOLD = 0.01   # normalised float32 amplitude

# Wake word variants (all lowercase, substring match)
WAKE_WORDS = ["hey goa", "hey, goa", "hei goa"]

# Strip the wake word prefix from a transcript to isolate the question
_WAKE_STRIP_RE = re.compile(
    r"^(?:hey[,\s]+goa|hei[,\s]+goa)[,\s!?.]*",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# TTS - Sarvam bulbul:v3
# ---------------------------------------------------------------------------

def speak(text: str) -> None:
    """
    Convert text → Bulbul v3 WAV bytes → play through speakers via sounddevice.
    Blocking - returns only after playback finishes.
    """
    print(f"[speaking] {text!r}")
    try:
        wav_bytes = text_to_speech(text)
        # soundfile can read from a bytes buffer
        buf = io.BytesIO(wav_bytes)
        data, samplerate = sf.read(buf, dtype="float32")
        sd.play(data, samplerate=samplerate)
        sd.wait()  # block until playback done
    except Exception as exc:
        # TTS failure is non-fatal - just print and continue
        print(f"[TTS error] {exc}")


# ---------------------------------------------------------------------------
# Audio capture helpers
# ---------------------------------------------------------------------------

def record_audio(duration_secs: float) -> np.ndarray:
    """Record mic audio, return 1-D float32 numpy array (mono, −1…1)."""
    frames = int(SAMPLE_RATE * duration_secs)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS,
                   dtype="float32", blocking=True)
    return audio.flatten()


def rms(audio: np.ndarray) -> float:
    """Root-mean-square amplitude of a float32 audio array."""
    return float(np.sqrt(np.mean(audio ** 2)))


def audio_to_wav_tmp(audio: np.ndarray) -> str:
    """Write numpy array → temp WAV file. Caller must delete the file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    sf.write(tmp.name, audio, SAMPLE_RATE, subtype="PCM_16")
    return tmp.name


def transcribe_audio_array(audio: np.ndarray) -> str:
    """numpy array → temp WAV → Sarvam STT → transcript string."""
    tmp_path = audio_to_wav_tmp(audio)
    try:
        return transcribe_audio(tmp_path).strip()
    finally:
        os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Wake word detection + question extraction
# ---------------------------------------------------------------------------

def contains_wake_word(text: str) -> bool:
    lower = text.lower()
    return any(w in lower for w in WAKE_WORDS)


def extract_question(transcript: str) -> str:
    """
    "Hey Goa, when is the deadline?" → "when is the deadline?"
    Returns empty string if nothing follows the wake word.
    """
    question = _WAKE_STRIP_RE.sub("", transcript).strip()
    return question.lstrip(",;:.!? ")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_voice_loop() -> None:
    """
    Runs forever:
      1. Record LISTEN_SECS of audio.
      2. Skip if silent (RMS gate).
      3. STT → check for wake word; ignore if absent.
      4. Extract question from same transcript.
         If empty (user said only "Hey Goa") → prompt and loop.
      5. Run RAG agent → speak answer via Bulbul v3 → back to step 1.
    """
    print("Voice assistant started.")
    print("Say 'Hey Goa <your question>' to activate. Ctrl+C to quit.\n")
    speak("Hey Goa assistant is ready. Say Hey Goa followed by your question.")

    try:
        while True:
            print("[listening...]")
            audio = record_audio(LISTEN_SECS)

            # Gate: skip silent frames
            if rms(audio) < SILENCE_RMS_THRESHOLD:
                continue

            try:
                transcript = transcribe_audio_array(audio)
            except Exception as exc:
                print(f"[STT error] {exc}")
                continue

            if not transcript:
                continue

            print(f"[heard] {transcript!r}")

            if not contains_wake_word(transcript):
                continue

            question = extract_question(transcript)

            if not question:
                print("[wake word only - no question]")
                speak("Say Hey Goa followed by your question in one go.")
                continue

            print(f"[question] {question!r}")

            speak("Let me look that up.")
            try:
                result = run_agent(question)
            except Exception as exc:
                print(f"[agent error] {exc}")
                speak("Sorry, something went wrong. Please try again.")
                continue

            answer = result.get("answer", "I don't have an answer for that.")
            print(f"[answer] {answer!r}")

            speak(answer)
            # next question also needs "Hey Goa"

    except KeyboardInterrupt:
        print("\nShutting down.")
        speak("Goodbye!")


if __name__ == "__main__":
    run_voice_loop()
