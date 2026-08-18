import glob
import os
import time

from sarvamai import SarvamAI

import config

SAMPLE_DIR = os.getenv("SAMPLE_AUDIO_DIR", "./sample_audio")
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac", "*.webm")


def find_sample_clips(sample_dir: str) -> list:
    paths = []
    for pattern in AUDIO_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(sample_dir, pattern)))
    return sorted(paths)


def run_latency_test(sample_dir: str = SAMPLE_DIR):
    client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)

    audio_files = find_sample_clips(sample_dir)
    if not audio_files:
        print(
            f"No audio files found in {sample_dir}. "
            f"Add 3-4 short clips ({', '.join(AUDIO_EXTENSIONS)}) and rerun."
        )
        return

    results = []
    for path in audio_files:
        file_size_kb = os.path.getsize(path) / 1024
        start = time.perf_counter()
        with open(path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model=config.SARVAM_STT_MODEL,
                language_code=config.SARVAM_LANGUAGE_CODE,
            )
        elapsed = time.perf_counter() - start
        results.append((os.path.basename(path), file_size_kb, elapsed, response.transcript))
        print(
            f"{os.path.basename(path):30s} "
            f"size={file_size_kb:7.1f}KB  "
            f"latency={elapsed:6.2f}s  "
            f"transcript={response.transcript!r}"
        )

    avg = sum(r[2] for r in results) / len(results)
    fastest = min(results, key=lambda r: r[2])
    slowest = max(results, key=lambda r: r[2])
    print(f"\n{len(results)} clips | avg={avg:.2f}s | fastest={fastest[0]} ({fastest[2]:.2f}s) | slowest={slowest[0]} ({slowest[2]:.2f}s)")


if __name__ == "__main__":
    run_latency_test()
