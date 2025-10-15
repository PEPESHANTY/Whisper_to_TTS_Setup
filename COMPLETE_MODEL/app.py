import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
import io, os, requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Whisper → TTS", page_icon="🎙", layout="centered")
st.title("🎙 Talk to AI")

def record_audio(duration=10, fs=16000):
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    st.success("Recording complete.")
    buf = io.BytesIO()
    wavio.write(buf, audio, fs, sampwidth=2)
    buf.seek(0)
    return buf

def transcribe(buf):
    st.write("🧠 Transcribing...")
    audio_file = io.BytesIO(buf.read())
    audio_file.name = "speech.wav"
    resp = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
    )
    return resp.text.strip()

def get_answer(transcript):
    st.write("💬 Thinking...")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly conversational assistant. Reply naturally."},
            {"role": "user", "content": transcript}
        ]
    )
    return resp.choices[0].message.content.strip()

def synthesize_tts(answer_text):
    st.write("🔊 Generating audio (via Piper TTS)...")
    try:
        response = requests.post(
            "http://localhost:5000/tts",
            json={"text": answer_text},   # 👈 send JSON body, not raw bytes
            timeout=30
        )
        if response.status_code != 200:
            raise RuntimeError(f"TTS server error {response.status_code}: {response.text}")

        audio_data = io.BytesIO(response.content)
        st.success("Audio generated successfully ✅")
        return audio_data

    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None

# --- UI ---
if st.button("🎙 Start Talking"):
    buf = record_audio(duration=8)
    transcript = transcribe(buf)
    st.markdown(f"**You said:** {transcript}")

    answer = get_answer(transcript)
    st.markdown(f"**AI:** {answer}")

    audio_data = synthesize_tts(answer)
    st.audio(audio_data, format="audio/mp3")
