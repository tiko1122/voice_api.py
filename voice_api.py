
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

# CORS: permissive for simplicity (you can tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = (
    "You are a concise, friendly hotel booking assistant. "
    "Ask for missing details together (check-in/out, adults, children, pets, name, phone, parking). "
    "Be brief and helpful."
)

class ChatIn(BaseModel):
    session_id: str
    text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(inp: ChatIn):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inp.text},
        ]
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content
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
