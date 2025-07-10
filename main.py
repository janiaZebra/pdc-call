from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import websockets
import logging
import base64
from jania import env
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
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
    logger.info("WebSocket accepted")
    stream_sid = None
    openai_ws = None
    stop_event = asyncio.Event()
    user_speaking = False
    assistant_speaking = False
    audio_queue = deque()
    MAX_QUEUE_SIZE = 3

    async def send_audio_to_twilio():
        while not stop_event.is_set():
            if audio_queue:
                item = audio_queue.popleft()
                await websocket.send_text(json.dumps(item['media_message']))
                await asyncio.sleep(item['duration'])
            else:
                await asyncio.sleep(0.005)

    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        OPENAI_WS_URL = f"wss://api.openai.com/v1/realtime?model={config_store['model']}"
        openai_ws = await websockets.connect(OPENAI_WS_URL, additional_headers=headers)
        logger.info("OpenAI WS connected")

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
                    if message["event"] == "connected":
                        logger.info("Twilio connected")
                    elif message["event"] == "start":
                        stream_sid = message["start"]["streamSid"]
                        logger.info(f"Stream: {stream_sid}")
                    elif message["event"] == "media":
                        if not assistant_speaking:
                            payload = message["media"]["payload"]
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": payload
                            }
                            await openai_ws.send(json.dumps(audio_append))
                    elif message["event"] == "stop":
                        logger.info("Stream stopped")
                        stop_event.set()
                        break
            except WebSocketDisconnect:
                logger.info("Twilio WS disconnect")
                stop_event.set()
            except Exception as e:
                logger.error(f"Twilio handler: {e}")
                stop_event.set()

        async def handle_openai_messages():
            nonlocal user_speaking, assistant_speaking
            try:
                async for message in openai_ws:
                    response = json.loads(message)

                    if response["type"] == "input_audio_buffer.speech_started":
                        user_speaking = True
                        logger.info("User speaking")
                        if assistant_speaking:
                            logger.info("User interrupted assistant. Cancelling response.")
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))
                            await openai_ws.send(json.dumps({"type": "output_audio_buffer.clear"}))
                            if stream_sid:
                                await websocket.send_text(json.dumps({
                                    "event": "clear",
                                    "streamSid": stream_sid
                                }))
                            audio_queue.clear()
                            assistant_speaking = False

                    elif response["type"] == "input_audio_buffer.speech_stopped":
                        user_speaking = False
                        logger.info("User stopped")

                    elif response["type"] == "response.started":
                        logger.info("Agent speaking")
                        assistant_speaking = True

                    elif response["type"] == "response.audio.delta":
                        if user_speaking:
                            continue
                        audio_delta = response.get("delta", "")
                        if audio_delta and stream_sid:
                            decoded = base64.b64decode(audio_delta)
                            duration = len(decoded) / 8000
                            media_message = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_delta
                                }
                            }
                            if len(audio_queue) < MAX_QUEUE_SIZE:
                                audio_queue.append({'media_message': media_message, 'duration': duration})

                    elif response["type"] == "response.audio.done":
                        assistant_speaking = False
                        logger.info("Agent done")

                    elif response["type"] == "response.audio_transcript.done":
                        logger.info(f"Agent: {response.get('transcript', '')}")
                    elif response["type"] == "conversation.item.input_audio_transcription.completed":
                        logger.info(f"User: {response.get('transcript', '')}")
                    elif response["type"] == "error":
                        logger.error(f"OpenAI error: {response}")
                    elif response["type"] == "response.cancelled":
                        assistant_speaking = False
                        logger.info("Response cancelled.")

                    if stop_event.is_set():
                        break

            except Exception as e:
                logger.error(f"OpenAI handler: {e}")

        audio_pacing_task = asyncio.create_task(send_audio_to_twilio())

        await asyncio.gather(
            handle_twilio_messages(),
            handle_openai_messages()
        )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if openai_ws:
            await openai_ws.close()
        await websocket.close()
        logger.info("WebSocket closed")
