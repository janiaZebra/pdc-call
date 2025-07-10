from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json, asyncio, websockets, logging
from jania import env
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ivr")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

API_KEY = env("OPENAI_API_KEY")

config_store = {
    "voice": env("OPENAI_VOICE", "coral"),
    "model": env("OPENAI_MODEL", "gpt-4o-realtime-preview"),
    "instructions": env("SYSTEM_PROMPT", "Eres un agente que responde en Español castellano plano. Serio y servicial"),
    "temperature": 0.8,
    "modalities": ["text", "audio"],
    "input_audio_format": "g711_ulaw",
    "output_audio_format": "g711_ulaw",
    "input_audio_transcription": {"model": "whisper-1"},
    "turn_detection": {"type": "server_vad", "threshold": 0.5,
                       "prefix_padding_ms": 300, "silence_duration_ms": 500,
                       "create_response": True},
    "max_response_output_tokens": 4096,
    "tool_choice": "auto",
    "tools": []
}

class SessionConfig(BaseModel):
    voice: Optional[str] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    temperature: Optional[float] = None
    modalities: Optional[List[str]] = None
    input_audio_format: Optional[str] = None
    output_audio_format: Optional[str] = None
    input_audio_transcription: Optional[Dict[str, Any]] = None
    turn_detection: Optional[Dict[str, Any]] = None
    max_response_output_tokens: Optional[int] = None
    tool_choice: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None

@app.get("/")          async def index():        return FileResponse("index.html")
@app.get("/config")    async def get_cfg():      return config_store
@app.post("/config")   async def upd_cfg(c:SessionConfig):
    for k,v in c.dict(exclude_unset=True).items(): config_store[k]=v
    return {"status":"ok", "config":config_store}

@app.post("/twilio/voice")
async def twilio_voice(r:Request):
    host=r.headers.get("host","localhost")
    twiml=f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect><Stream url="wss://{host}/ws/audio"/></Connect>
  <Pause length="60"/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/ws/audio")
async def ws_audio(ws:WebSocket):
    await ws.accept(); log.info("WS ok")
    stream_sid=None; mark_id=0
    user_speaking=False; assistant_speaking=False
    stop=asyncio.Event()

    # --- connect to OpenAI realtime
    headers={"Authorization":f"Bearer {API_KEY}","OpenAI-Beta":"realtime=v1"}
    oa=await websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={config_store['model']}",
        additional_headers=headers)
    await oa.send(json.dumps({"type":"session.update","session":config_store}))

    # ---------- Twilio → OpenAI ----------
    async def twilio_loop():
        nonlocal stream_sid
        try:
            while True:
                m=json.loads(await ws.receive_text())
                typ=m["event"]
                if typ=="start":
                    stream_sid=m["start"]["streamSid"]; log.info(f"stream {stream_sid}")
                elif typ=="media":
                    await oa.send(json.dumps({"type":"input_audio_buffer.append",
                                              "audio":m["media"]["payload"]}))
                elif typ=="stop":
                    stop.set(); break
        except WebSocketDisconnect: stop.set()

    # ---------- OpenAI → Twilio ----------
    async def oa_loop():
        nonlocal user_speaking, assistant_speaking, mark_id
        async for raw in oa:
            r=json.loads(raw)
            t=r["type"]

            if t=="input_audio_buffer.speech_started":
                user_speaking=True
                if assistant_speaking:
                    await oa.send(json.dumps({"type":"response.cancel"}))
                    if stream_sid:
                        await ws.send_text(json.dumps({"event":"clear","streamSid":stream_sid}))
                    assistant_speaking=False

            elif t=="input_audio_buffer.speech_stopped":
                user_speaking=False

            elif t=="response.started":
                assistant_speaking=True

            elif t=="response.audio.delta":
                if user_speaking or not stream_sid: continue
                payload=r.get("delta","")
                if payload:
                    mark_name=f"m{mark_id}"; mark_id+=1
                    await ws.send_text(json.dumps({"event":"media","streamSid":stream_sid,
                                                   "media":{"payload":payload}}))
                    await ws.send_text(json.dumps({"event":"mark","streamSid":stream_sid,
                                                   "mark":{"name":mark_name}}))

            elif t in ("response.audio.done","response.cancelled"):
                assistant_speaking=False

            elif t=="error": log.error(f"OA error {r}")

            if stop.is_set(): break

    try:
        await asyncio.gather(twilio_loop(), oa_loop())
    finally:
        await oa.close(); await ws.close(); log.info("closed")
