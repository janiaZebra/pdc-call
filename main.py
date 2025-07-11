from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import websockets
import logging
from jania import env
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import base64
from collections import deque

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
MENSAJE_INICIAL = env("MENSAJE_INICIAL", "Hola")

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

send_to_twilio = True

@app.websocket("/ws/control")
async def control_websocket(websocket: WebSocket):
    global send_to_twilio
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "pause":
                send_to_twilio = False
                logger.info("⛔ Envío a Twilio pausado por control remoto")
            elif msg == "resume":
                send_to_twilio = True
                logger.info("▶️ Envío a Twilio reanudado por control remoto")
    except WebSocketDisconnect:
        logger.info("🔌 Conexión de control cerrada")

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
    audio_queue = deque()
    global send_to_twilio

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
        await asyncio.sleep(1.7)
        initial_message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "text", "text": MENSAJE_INICIAL}
                ]
            }
        }
        await openai_ws.send(json.dumps(initial_message))

        initial_response = {
            "type": "response.create",
            "response": {
                "modalities": config_store["modalities"],
                "instructions": config_store["instructions"],
                "voice": config_store["voice"],
                "output_audio_format": config_store["output_audio_format"],
                "temperature": config_store["temperature"],
                "tool_choice": config_store["tool_choice"],
                "tools": config_store["tools"]
            }
        }
        await openai_ws.send(json.dumps(initial_response))
        logger.info(f"📨 Enviado mensaje inicial: {MENSAJE_INICIAL}")

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
            global send_to_twilio
            nonlocal assistant_responding, user_is_speaking
            try:
                async for message in openai_ws:
                    response = json.loads(message)
                    t = response.get("type", "")
                    logger.info(f"OpenAI: {t}")
                    if t == "input_audio_buffer.speech_started":
                        user_is_speaking = True
                        send_to_twilio = False
                        total_bytes = sum(len(base64.b64decode(p)) for p in audio_queue)
                        buffer_seconds = round(total_bytes / 8000, 2)
                        logger.info(f"🎙️ Buffer antes de cancelar: {buffer_seconds}s en {len(audio_queue)} paquetes")
                        audio_queue.clear()
                        if assistant_responding:
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))
                            assistant_responding = False
                        send_to_twilio = True
                    elif t == "input_audio_buffer.speech_stopped":
                        user_is_speaking = False
                        send_to_twilio = True
                    elif t == "response.created":
                        send_to_twilio = True
                    elif t == "response.started":
                        assistant_responding = True
                        send_to_twilio = True
                        logger.info("🔊 Nueva respuesta iniciada. Envío a Twilio reactivado.")
                        assistant_responding = True
                    elif t == "response.done":
                        assistant_responding = False
                        send_to_twilio = True
                        logger.info("Fin de respuesta. Envío a Twilio reactivado.")
                    elif t == "response.audio.delta":
                        audio_delta = response.get("delta", "")
                        if audio_delta and send_to_twilio:
                            audio_queue.append(audio_delta)
                    elif t == "error":
                        logger.error(f"OpenAI error: {response}")
                    if stop_event.is_set():
                        break
            except Exception:
                stop_event.set()

        async def send_audio_to_twilio():
            while not stop_event.is_set():
                if send_to_twilio and stream_sid and audio_queue and not user_is_speaking:
                    try:
                        audio_delta = audio_queue.popleft()
                        raw_audio = base64.b64decode(audio_delta)
                        duration = len(raw_audio) / 8000
                        media_message = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": audio_delta}
                        }
                        await websocket.send_text(json.dumps(media_message))
                        await asyncio.sleep(duration)
                    except Exception as e:
                        logger.error(f"Error al enviar audio a Twilio: {e}")
                else:
                    await asyncio.sleep(0.005)

        await asyncio.gather(
            handle_twilio_messages(),
            handle_openai_messages(),
            send_audio_to_twilio()
        )

    except Exception as e:
        logger.error(f"Error en conexión principal: {e}")
    finally:
        stop_event.set()
        if openai_ws:
            await openai_ws.close()
        await websocket.close()
