import os, requests

PIPER_TTS_API = os.getenv("PIPER_TTS_API", "http://127.0.0.1:5000/tts")

def synthesize_to_wav(text: str, lang: str, outfile: str, timeout=600):
    payload = {"text": text, "lang": lang}
    r = requests.post(PIPER_TTS_API, json=payload, timeout=timeout)
    r.raise_for_status()
    if r.headers.get("Content-Type","").lower() != "audio/wav":
        raise RuntimeError(f"TTS error {r.status_code}: {r.text[:300]}")
    with open(outfile, "wb") as f:
        f.write(r.content)
    return outfile
