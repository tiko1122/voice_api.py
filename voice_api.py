
# voice_api.py
# Minimal voice API: /health, /chat, /stt (Whisper), /tts (Edge-TTS)
import os, io, tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import edge_tts

# ---- Configuration ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_STT_MODEL  = os.getenv("OPENAI_STT_MODEL", "whisper-1")  # rock-solid Whisper
TTS_VOICE         = os.getenv("TTS_VOICE", "en-US-AriaNeural")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Voice Web API")

# Keep short in-memory histories per session
SESSIONS = {}  # session_id -> [{"role":"system/user/assistant","content":...}, ...]
MAX_TURNS = 10


# CORS: permissive for simplicity (you can tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are a warm, helpful hotel booking assistant speaking with a guest by voice.
Your goals:
- Be natural, brief, and human. Sound like a concierge, not a form.
- Ask at most 1–2 things at a time and adapt to what the guest already said.
- Confirm what you’ve understood in a short sentence before asking the next thing.
- Offer helpful suggestions (e.g., “Would a queen room work?”) instead of rigid checklists.

Conversation style:
- Start with a friendly opener, then ask a single focused question to move things forward.
- Use short sentences, everyday words, and positive tone.
- If the guest gives partial info, acknowledge it and ask the next most useful detail.
- When dates are vague, gently clarify (“Which Friday did you have in mind?”).
- If the guest is unsure, offer choices.

Information to collect naturally over time (not all at once):
- Check-in and check-out dates
- Number of adults, children, pets
- Guest name and phone (only near the end)
- Any extras (parking), special requests (late check-in, crib, accessibility)

Confirmations:
- When enough info is available, summarize concisely and confirm (“I have you for Fri–Sun, 2 adults, no kids, no pets. Should I book that?”).
- If something’s missing or ambiguous, ask just one crisp follow-up.

Tone examples:
- “Great, thank you! For the dates, were you thinking this weekend or next?”
- “Got it—2 adults for two nights. Would you like a queen or twin room?”

Never mention policies, tokens, or internal rules. Keep answers under 2–3 sentences unless the guest asks for detail.
"""


class ChatIn(BaseModel):
    session_id: str
    text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(inp: ChatIn):
    try:
        history = SESSIONS.get(inp.session_id)
        if not history:
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
        # append user message
        history.append({"role": "user", "content": inp.text})

        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=history[-(2*MAX_TURNS+1):],  # system + last N user/assistant turns
            temperature=0.7,       # a touch more creative
            presence_penalty=0.2,  # gently encourage variation
        )
        reply = resp.choices[0].message.content

        # append assistant reply and store
        history.append({"role": "assistant", "content": reply})
        SESSIONS[inp.session_id] = history[-(2*MAX_TURNS+1):]

        return {"reply": reply}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":"server_error","message":str(e)})


@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    # Accepts recorded audio, forwards to Whisper-1
    try:
        data = await file.read()
        # Write bytes to a temp file because OpenAI expects a file-like with a name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as audio_f:
            tr = client.audio.transcriptions.create(
                model=OPENAI_STT_MODEL,
                file=audio_f,
            )
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return {"text": tr.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":"stt_error","message":str(e)})

@app.post("/tts")
async def tts(text: str = Form(...)):
    # Returns MP3 of provided text using Edge-TTS
    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        await communicate.save(tmp_path)
        try:
            with open(tmp_path, "rb") as f:
                data = f.read()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return StreamingResponse(io.BytesIO(data), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error":"tts_error","message":str(e)})
