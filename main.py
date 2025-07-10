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
import time

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

    # Sistema mejorado de gestión de audio
    audio_queue = deque()
    current_response_id = 0
    cancelled_response_ids = set()

    # Control de flujo mejorado
    BUFFER_SIZE_MS = 200  # Mantener solo 200ms de audio en buffer
    CHUNK_DURATION_MS = 20  # Enviar chunks más pequeños

    # Para rastrear el audio enviado
    sent_audio_timestamp = 0
    last_mark_id = 0

    async def send_audio_to_twilio():
        nonlocal sent_audio_timestamp, last_mark_id

        while not stop_event.is_set():
            try:
                if audio_queue:
                    item = audio_queue.popleft()

                    # Verificar si este audio pertenece a una respuesta cancelada
                    if item['response_id'] in cancelled_response_ids:
                        continue

                    # Calcular si debemos enviar este audio basado en el buffer
                    current_time = time.time() * 1000
                    if sent_audio_timestamp - current_time > BUFFER_SIZE_MS:
                        # Demasiado audio en buffer, esperar
                        audio_queue.appendleft(item)  # Devolver a la cola
                        await asyncio.sleep(0.01)
                        continue

                    # Enviar el audio
                    await websocket.send_text(json.dumps(item['media_message']))

                    # Enviar mark event cada cierto tiempo para rastrear progreso
                    if current_time - last_mark_id > 100:  # Cada 100ms
                        last_mark_id = current_time
                        mark_message = {
                            "event": "mark",
                            "streamSid": stream_sid,
                            "mark": {
                                "name": f"audio_{item['response_id']}_{int(current_time)}"
                            }
                        }
                        await websocket.send_text(json.dumps(mark_message))

                    sent_audio_timestamp = current_time + item['duration_ms']
                    await asyncio.sleep(item['duration_ms'] / 1000)
                else:
                    await asyncio.sleep(0.001)  # Más responsivo

            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    # Función para generar silencio en g711_ulaw
    def generate_silence_g711(duration_ms):
        # En G.711 u-law, el silencio es 0xFF (255 en decimal)
        samples = int(8000 * duration_ms / 1000)  # 8kHz sample rate
        silence = bytes([0xFF] * samples)
        return base64.b64encode(silence).decode('utf-8')

    async def send_silence_burst(duration_ms=200):
        """Envía una ráfaga de silencio para limpiar el buffer"""
        if stream_sid:
            silence_audio = generate_silence_g711(duration_ms)
            silence_message = {
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": silence_audio
                }
            }
            await websocket.send_text(json.dumps(silence_message))

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
                        # Solo procesar audio del usuario si el asistente no está siendo interrumpido
                        if not assistant_speaking or user_speaking:
                            payload = message["media"]["payload"]
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": payload
                            }
                            await openai_ws.send(json.dumps(audio_append))
                    elif message["event"] == "mark":
                        # Twilio confirmó que reprodujo un mark
                        mark_name = message.get("mark", {}).get("name", "")
                        logger.debug(f"Mark reached: {mark_name}")
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
            nonlocal user_speaking, assistant_speaking, current_response_id

            try:
                async for message in openai_ws:
                    response = json.loads(message)

                    if response["type"] == "input_audio_buffer.speech_started":
                        user_speaking = True
                        logger.info("User speaking - interrupting assistant")

                        if assistant_speaking:
                            # Marcar la respuesta actual como cancelada
                            cancelled_response_ids.add(current_response_id)

                            # Cancelar inmediatamente en OpenAI
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))

                            # Limpiar buffers
                            audio_queue.clear()

                            # Enviar silencio para sobrescribir audio en buffer de Twilio
                            await send_silence_burst(300)

                            # Enviar evento clear a Twilio
                            if stream_sid:
                                await websocket.send_text(json.dumps({
                                    "event": "clear",
                                    "streamSid": stream_sid
                                }))

                            assistant_speaking = False
                            logger.info("Assistant interrupted successfully")

                    elif response["type"] == "input_audio_buffer.speech_stopped":
                        user_speaking = False
                        logger.info("User stopped speaking")

                    elif response["type"] == "response.started":
                        logger.info("Assistant starting response")
                        assistant_speaking = True
                        current_response_id += 1
                        # Limpiar IDs cancelados antiguos para no acumular memoria
                        if len(cancelled_response_ids) > 10:
                            cancelled_response_ids.clear()

                    elif response["type"] == "response.audio.delta":
                        # No procesar audio si esta respuesta fue cancelada
                        if current_response_id in cancelled_response_ids:
                            continue

                        audio_delta = response.get("delta", "")
                        if audio_delta and stream_sid:
                            decoded = base64.b64decode(audio_delta)
                            # Dividir en chunks más pequeños si es necesario
                            chunk_size = int(8000 * CHUNK_DURATION_MS / 1000)  # samples per chunk

                            for i in range(0, len(decoded), chunk_size):
                                chunk = decoded[i:i + chunk_size]
                                chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                                duration_ms = len(chunk) * 1000 / 8000

                                media_message = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": chunk_b64
                                    }
                                }

                                audio_queue.append({
                                    'media_message': media_message,
                                    'duration_ms': duration_ms,
                                    'response_id': current_response_id
                                })

                    elif response["type"] == "response.audio.done":
                        if current_response_id not in cancelled_response_ids:
                            assistant_speaking = False
                            logger.info("Assistant finished speaking")

                    elif response["type"] == "response.audio_transcript.done":
                        if current_response_id not in cancelled_response_ids:
                            logger.info(f"Assistant: {response.get('transcript', '')}")

                    elif response["type"] == "conversation.item.input_audio_transcription.completed":
                        logger.info(f"User: {response.get('transcript', '')}")

                    elif response["type"] == "error":
                        logger.error(f"OpenAI error: {response}")

                    elif response["type"] == "response.cancelled":
                        assistant_speaking = False
                        logger.info(f"Response {current_response_id} cancelled by OpenAI")

                    if stop_event.is_set():
                        break

            except Exception as e:
                logger.error(f"OpenAI handler: {e}")

        # Iniciar el task de envío de audio
        audio_pacing_task = asyncio.create_task(send_audio_to_twilio())

        # Ejecutar ambos handlers
        await asyncio.gather(
            handle_twilio_messages(),
            handle_openai_messages()
        )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        stop_event.set()
        if openai_ws:
            await openai_ws.close()
        await websocket.close()
        logger.info("WebSocket closed")