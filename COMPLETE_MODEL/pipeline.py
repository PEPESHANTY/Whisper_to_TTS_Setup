#!/usr/bin/env python3
import os, argparse, time
from pathlib import Path
from dotenv import load_dotenv

from modules_audio_whisper import run_whisper_capture
from modules_gpt import decide_and_answer_full
from modules_tts import synthesize_to_wav

load_dotenv()

def parse_args():
    p = argparse.ArgumentParser(description="Mic → Whisper → GPT-4o-mini → Piper TTS (one WAV)")
    p.add_argument("--whisper-endpoint", default=os.getenv("WHISPER_ENDPOINT","http://127.0.0.1:9000/transcribe"))
    p.add_argument("--piper-endpoint",   default=os.getenv("PIPER_TTS_API","http://127.0.0.1:5000/tts"))
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--sec", type=float, default=8.0)
    p.add_argument("--lang-hint", choices=["en","vi"], default="en",
                   help="language hint for Whisper ASR (does not lock output)")
    p.add_argument("--task", choices=["transcribe","translate"], default="transcribe")
    p.add_argument("--vad-mode", type=int, choices=[0,1,2,3], default=1)
    p.add_argument("--energy-mult", type=float, default=1.2)
    p.add_argument("--energy-floor", type=float, default=30.0)
    p.add_argument("--min-speech", type=float, default=0.08)
    p.add_argument("--device", default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-keep", type=int, default=0)

    p.add_argument("--data-root", default=str(Path(__file__).parent / "data"),
                   help="base folder to store transcripts/answers/audio")
    p.add_argument("--prefix", default="session", help="output filename prefix")
    return p.parse_args()

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    transcripts_dir = data_root / "transcripts"
    answers_dir     = data_root / "answers"
    audio_dir       = data_root / "audio"
    for d in (transcripts_dir, answers_dir, audio_dir): d.mkdir(parents=True, exist_ok=True)

    # 1) Record mic and build transcript (Ctrl+C to stop)
    transcript_path = run_whisper_capture(
        endpoint=args.whisper_endpoint, sec=args.sec, sr=args.sr,
        lang_hint=args.lang_hint, task=args.task, vad_mode=args.vad_mode,
        energy_mult=args.energy_mult, energy_floor=args.energy_floor,
        min_speech=args.min_speech, device=args.device, debug=args.debug,
        debug_keep=args.debug_keep, transcripts_dir=transcripts_dir
    )

    # Read full transcript
    full_text = transcript_path.read_text(encoding="utf-8").strip()
    if not full_text:
        print("No transcript text found; exiting.")
        return

    # 2) GPT-4o-mini → decide language + answer (with [TTS] prefix)
    lang, answer = decide_and_answer_full(full_text)
    print(f"[GPT] language={lang}  chars={len(answer)}")

    # 3) Save answer text
    ts = int(time.time())
    answer_txt = answers_dir / f"{args.prefix}_{ts}.txt"
    answer_txt.write_text(answer, encoding="utf-8")
    print(f"[SAVE] Answer → {answer_txt}")

    # 4) TTS once for the whole answer
    wav_out = audio_dir / f"{args.prefix}_{ts}.wav"
    synthesize_to_wav(answer, lang, str(wav_out))
    print(f"[TTS] Wrote WAV → {wav_out}")

if __name__ == "__main__":
    main()
