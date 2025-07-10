from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import websockets
import logging
from collections import deque
from jania import env
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "create_response": True
    },
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

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/config")
async def get_config():
    return config_store

@app.post("/config")
async def update_config(config: SessionConfig):
    for key, value in config.dict(exclude_unset=True).items():
        if value is not None:
            config_store[key] = value
    return {"status": "success", "config": config_store}

@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    host = request.headers.get("host", "localhost")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="wss://{host}/ws/audio" />
        </Connect>
        <Pause length="60"/>
    </Response>
    """
    return Response(content=twiml.strip(), media_type="application/xml")

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    openai_ws = None
    stop_event = asyncio.Event()
    assistant_responding = False
    user_is_speaking = False
    sending_audio = True
    audio_queue = deque()

    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        OPENAI_WS_URL = f"wss://api.openai.com/v1/realtime?model={config_store['model']}"
        openai_ws = await websockets.connect(OPENAI_WS_URL, additional_headers=headers)
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": config_store["modalities"],
                "instructions": config_store["instructions"],
                "voice": config_store["voice"],
                "input_audio_format": config_store["input_audio_format"],
                "output_audio_format": config_store["output_audio_format"],
                "input_audio_transcription": config_store["input_audio_transcription"],
                "turn_detection": config_store["turn_detection"],
                "temperature": config_store["temperature"],
                "max_response_output_tokens": config_store["max_response_output_tokens"],
                "tool_choice": config_store["tool_choice"],
                "tools": config_store["tools"]
            }
        }
        await openai_ws.send(json.dumps(session_config))

        async def handle_twilio_messages():
            nonlocal stream_sid
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    if message["event"] == "start":
                        stream_sid = message["start"]["streamSid"]
                    elif message["event"] == "media":
                        payload = message["media"]["payload"]
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": payload
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif message["event"] == "stop":
                        stop_event.set()
                        break
            except WebSocketDisconnect:
                stop_event.set()
            except Exception:
                stop_event.set()

        async def handle_openai_messages():
            nonlocal assistant_responding, user_is_speaking, sending_audio
            try:
                async for message in openai_ws:
                    response = json.loads(message)
                    t = response.get("type", "")
                    if t == "input_audio_buffer.speech_started":
                        user_is_speaking = True
                        sending_audio = False
                        audio_queue.clear()
                        if assistant_responding:
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))
                    elif t == "input_audio_buffer.speech_stopped":
                        user_is_speaking = False
                    elif t == "response.started":
                        assistant_responding = True
                        sending_audio = True
                    elif t == "response.done":
                        assistant_responding = False
                        sending_audio = False
                        audio_queue.clear()
                    elif t == "response.audio.delta":
                        audio_delta = response.get("delta", "")
                        if audio_delta and stream_sid and not user_is_speaking:
                            audio_queue.append(audio_delta)
                    elif t == "error":
                        logger.error(f"OpenAI error: {response}")
                    if stop_event.is_set():
                        break
            except Exception:
                stop_event.set()

        async def send_audio_to_twilio():
            while not stop_event.is_set():
                if sending_audio and stream_sid and audio_queue:
                    audio_delta = audio_queue.popleft()
                    media_message = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": audio_delta}
                    }
                    await websocket.send_text(json.dumps(media_message))
                await asyncio.sleep(0.03)  # 30ms to simulate real-time

        await asyncio.gather(
            handle_twilio_messages(),
            handle_openai_messages(),
            send_audio_to_twilio()
        )

    except Exception as e:
        logger.error(f"Unhandled error: {e}")
    finally:
        stop_event.set()
        if openai_ws:
            await openai_ws.close()
        await websocket.close()
