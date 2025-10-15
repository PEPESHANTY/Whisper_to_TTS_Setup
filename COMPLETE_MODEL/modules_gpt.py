import os, json
from typing import Literal, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Conversational, acknowledges pleasantries, EN/VI only
SYSTEM = """You are a friendly voice assistant.
GOALS
- Briefly acknowledge greetings/thanks (e.g., “Hi!” / “Chào bạn!” / “Thanks” / “Cảm ơn bạn”).
- Then answer the user’s actual question in a natural, conversational tone.
- Keep replies short: 1–2 sentences unless the user asks for more.
- Use contractions in English (it's, that's, you're).
LANGUAGE (strict):
- Reply ONLY in English ("en") or Vietnamese ("vi").
- If the user speaks Vietnamese, reply in Vietnamese.
- If the user speaks English, reply in English.
- If both appear, choose the language of the main question (usually the last question).
OUTPUT
Return ONLY compact JSON:
{"language":"en"|"vi","answer":"your spoken reply (1–2 short sentences)"}
No extra text outside the JSON.
"""

def decide_and_answer_full(transcript: str) -> Tuple[Literal["en","vi"], str]:
    """
    Returns (language, answer_with_prefix).
    language: 'en' or 'vi'
    answer_with_prefix: '[en] ...' or '[vi] ...'
    """
    prompt = (
        "Conversation transcript (raw user speech, including greetings/thanks):\n"
        f"{transcript}\n\n"
        "Write a short, conversational reply that first acknowledges pleasantries (briefly), "
        "then answers the main question. Respond in either English or Vietnamese only.\n"
        "Return JSON with keys language, answer."
    )

    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    data = json.loads(content)

    lang = (data.get("language") or "en").lower()
    if lang not in ("en", "vi"):
        lang = "en"

    answer = (data.get("answer") or "").strip()

    # Prefix with [en] or [vi]
    prefix = "[en]" if lang == "en" else "[vi]"
    if not answer.startswith(prefix):
        answer = f"{prefix} {answer}"

    return lang, answer



# import os, json
# from typing import Literal, Tuple
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()
# _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Force a compact, structured reply so we can parse reliably.
# SYSTEM = """You are a concise assistant. 
# 1) First, decide the language to answer in:
#    - If the question is in Vietnamese, answer in Vietnamese.
#    - If the user asks for another language explicitly, obey that.
#    - Otherwise, answer in the question's language.
# 2) Produce a clear, helpful answer.
# Return ONLY a compact JSON object: {"language":"en|vi","answer":"..."}"""

# def decide_and_answer_full(transcript: str) -> Tuple[Literal["en","vi"], str]:
#     prompt = (
#         "Conversation transcript (user speech):\n"
#         f"{transcript}\n\n"
#         "Return JSON only with keys language, answer."
#     )
#     resp = _client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role":"system","content":SYSTEM},
#                 {"role":"user","content":prompt}],
#         temperature=0.3,
#         response_format={"type":"json_object"}
#     )
#     content = resp.choices[0].message.content
#     data = json.loads(content)
#     lang = data.get("language","en").lower()
#     if lang not in ("en","vi"): lang = "en"
#     answer = str(data.get("answer","")).strip()
#     if not answer.startswith(" "):
#         answer = answer
#     return lang, answer
