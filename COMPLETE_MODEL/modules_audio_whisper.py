import os, io, time, threading, queue
from pathlib import Path
import numpy as np, sounddevice as sd, soundfile as sf, webrtcvad, requests
from dotenv import load_dotenv
load_dotenv()

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


def run_whisper_capture(*, endpoint: str, sec: float, sr: int,
                        lang_hint: str, task: str, vad_mode: int,
                        energy_mult: float, energy_floor: float,
                        min_speech: float, device, debug: bool,
                        debug_keep: int,
                        transcripts_dir: Path,
                        silence_timeout: float = 8.0) -> Path:
    """
    Records in rolling windows of `sec` seconds, keeps only VAD+energy positive
    frames, and stops once we observe `silence_timeout` seconds of continuous
    silence AFTER some speech was captured. Writes a single transcript file and
    returns its path.
    """
    ensure_dir(transcripts_dir)
    out_path = transcripts_dir / f"transcript_{int(time.time())}.txt"
    out_path.write_text("", encoding="utf-8")

    if device is not None:
        sd.default.device = (device, None)

    # --- ambient calibration & thresholds ---
    ambient = calibrate_ambient(sr, 1.0, device)
    energy_thr = max(energy_floor, ambient * energy_mult)
    print(f"[cfg] lang_hint={lang_hint} vad_mode={vad_mode} energy_thr≈{energy_thr:.1f} "
          f"min_speech={min_speech}s  silence_timeout={silence_timeout}s")
    print(f"[run] {sec:.1f}s windows  →  {endpoint}")
    print(f"     transcript file: {out_path}\nCtrl+C to stop.\n")

    vad = webrtcvad.Vad(vad_mode)
    debug_dir = transcripts_dir.parent / "debug"
    if debug: ensure_dir(debug_dir)

    # Aggregate buffers (raw int16 speech) until we hit silence timeout.
    agg_speech_bytes = bytearray()
    agg_speech_seconds = 0.0
    seen_any_speech = False
    silence_run = 0.0

    def _append_speech(arr_i16: np.ndarray):
        nonlocal agg_speech_seconds
        if arr_i16.size == 0:
            return
        agg_speech_bytes.extend(arr_i16.astype("<i2", copy=False).tobytes())
        agg_speech_seconds += arr_i16.size / float(sr)

    try:
        win = 0
        while True:
            win += 1
            # Record one window
            audio = sd.rec(int(sec * sr), samplerate=sr, channels=1, dtype="int16")
            sd.wait()
            x = audio.reshape(-1)

            # Optional debug: raw window
            if debug:
                if debug_keep > 0:
                    idx = ((win - 1) % debug_keep) + 1
                    sf.write(debug_dir / f"window_{idx:02d}.wav", x, sr)
                else:
                    sf.write(debug_dir / "last_window.wav", x, sr)

            # Extract only speech frames from this window
            speech, kept_sec = extract_speech_frames(x, sr, vad, energy_thr)

            # Optional debug: speech-only cut of the window
            if debug:
                if debug_keep > 0:
                    idx = ((win - 1) % debug_keep) + 1
                    sf.write(debug_dir / f"speech_{idx:02d}.wav", speech, sr)
                else:
                    sf.write(debug_dir / "last_speech.wav", speech, sr)

            # Update silence/speech state
            if kept_sec > 0.0:
                # We have speech in this window
                _append_speech(speech)
                silence_run = 0.0
                seen_any_speech = True
                print(f"[win {win}] +{kept_sec:.2f}s speech (total {agg_speech_seconds:.2f}s)")
            else:
                # No speech in this window
                silence_run += sec
                if seen_any_speech:
                    print(f"[win {win}] silence +{sec:.1f}s  (run={silence_run:.1f}s)")

            # Stop condition: after we've captured some speech, accumulate
            # `silence_timeout` seconds of trailing silence.
            if seen_any_speech and silence_run >= silence_timeout:
                print(f"⏹️  {silence_timeout:.0f}s of trailing silence detected. Finalizing…")
                break

    except KeyboardInterrupt:
        print("\nStopping capture…")

    # If we didn't collect enough speech, just return the (empty) transcript path
    if agg_speech_seconds < max(0.0, float(min_speech)):
        print(f"[final] Not enough speech captured ({agg_speech_seconds:.2f}s < {min_speech:.2f}s).")
        print(f"Done. Transcript saved at: {out_path}")
        return out_path

    # Build a single WAV in-memory with the aggregated speech frames
    speech_i16 = np.frombuffer(bytes(agg_speech_bytes), dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, speech_i16, sr, format="WAV")

    # Transcribe the aggregated clip
    try:
        txt = post_wav_bytes(endpoint, task, lang_hint, buf.getvalue()).strip()
    except Exception as e:
        txt = ""
        print(f"[final] Transcription error: {e}")

    if txt:
        print(txt, flush=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(txt + "\n")
    else:
        print("[final] (empty transcript)")

    print(f"Done. Transcript saved at: {out_path}")
    return out_path


def calibrate_ambient(sr: int, seconds: float, device):
    if device is not None:
        sd.default.device = (device, None)
    print(f"[calib] Listening {seconds:.1f}s to estimate ambient…")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16"); sd.wait()
    x = audio.astype(np.float32).ravel()
    rms = float(np.sqrt(np.mean(x * x)) + 1e-6)
    print(f"[calib] Ambient RMS ≈ {rms:.1f}")
    return rms

def to_pcm_bytes(x_int16: np.ndarray) -> bytes:
    return x_int16.astype("<i2", copy=False).tobytes()

def extract_speech_frames(x_int16: np.ndarray, sr: int, vad: webrtcvad.Vad,
                        energy_thr: float, frame_ms: int = 20):
    frame_len = int(sr * frame_ms / 1000)
    total = len(x_int16)
    keep = []
    kept_frames = 0
    i = 0
    while i + frame_len <= total:
        frame = x_int16[i:i + frame_len]
        rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
        speech = vad.is_speech(to_pcm_bytes(frame), sr) and (rms >= energy_thr)
        if speech:
            keep.append(frame); kept_frames += 1
        i += frame_len
    if not keep:
        return np.zeros((0,), dtype=np.int16), 0.0
    y = np.concatenate(keep, axis=0)
    kept_sec = kept_frames * frame_ms / 1000.0
    return y, kept_sec

def post_wav_bytes(endpoint: str, task: str, lang: str, wav_bytes: bytes) -> str:
    r = requests.post(
        endpoint,
        files={"file": ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav")},
        data={"task": task, "language": lang, "return_timestamps": "false"},
        timeout=None,
    )
    r.raise_for_status()
    j = r.json()
    return (j.get("text") or "").strip() if isinstance(j, dict) else str(j)

def run_whisper_capture(*, endpoint: str, sec: float, sr: int,
                        lang_hint: str, task: str, vad_mode: int,
                        energy_mult: float, energy_floor: float,
                        min_speech: float, device, debug: bool,
                        debug_keep: int,
                        transcripts_dir: Path) -> Path:
    ensure_dir(transcripts_dir)
    out_path = transcripts_dir / f"transcript_{int(time.time())}.txt"
    out_path.write_text("", encoding="utf-8")

    if device is not None:
        sd.default.device = (device, None)

    ambient = calibrate_ambient(sr, 1.0, device)
    energy_thr = max(energy_floor, ambient * energy_mult)
    print(f"[cfg] lang_hint={lang_hint} vad_mode={vad_mode} energy_thr≈{energy_thr:.1f} min_speech={min_speech}s")
    print(f"[run] {sec:.1f}s windows  →  {endpoint}")
    print(f"     transcript file: {out_path}\nCtrl+C to stop.\n")

    vad = webrtcvad.Vad(vad_mode)
    debug_dir = transcripts_dir.parent / "debug"
    if debug: ensure_dir(debug_dir)

    full_txt = []

    try:
        win = 0
        while True:
            win += 1
            audio = sd.rec(int(sec * sr), samplerate=sr, channels=1, dtype="int16")
            sd.wait()
            x = audio.reshape(-1)

            if debug:
                if debug_keep > 0:
                    idx = ((win - 1) % debug_keep) + 1
                    sf.write(debug_dir / f"window_{idx:02d}.wav", x, sr)
                else:
                    sf.write(debug_dir / "last_window.wav", x, sr)

            speech, kept_sec = extract_speech_frames(x, sr, vad, energy_thr)

            if debug:
                if debug_keep > 0:
                    idx = ((win - 1) % debug_keep) + 1
                    sf.write(debug_dir / f"speech_{idx:02d}.wav", speech, sr)
                else:
                    sf.write(debug_dir / "last_speech.wav", speech, sr)

            if kept_sec < min_speech or len(speech) == 0:
                print(f"[win {win}] kept={kept_sec:.2f}s → SKIP")
                continue

            buf = io.BytesIO(); sf.write(buf, speech, sr, format="WAV")
            txt = post_wav_bytes(endpoint, task, lang_hint, buf.getvalue())
            if txt:
                print(txt, flush=True)
                full_txt.append(txt)
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(txt + "\n")
            else:
                print(f"[win {win}] (empty)")
    except KeyboardInterrupt:
        print("\nStopping capture…")
    finally:
        # return the transcript path
        final_text = "\n".join(full_txt).strip()
        if final_text:
            with out_path.open("a", encoding="utf-8") as f:
                f.write("\n")
        print(f"Done. Transcript saved at: {out_path}")
        return out_path
