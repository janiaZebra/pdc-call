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
import numpy as np
import struct

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


# Simple VAD (Voice Activity Detection) local
class SimpleVAD:
    def __init__(self, energy_threshold=1000, speech_frames=5, silence_frames=15):
        self.energy_threshold = energy_threshold
        self.speech_frames = speech_frames  # Frames necesarios para confirmar voz
        self.silence_frames = silence_frames  # Frames necesarios para confirmar silencio
        self.speech_counter = 0
        self.silence_counter = 0
        self.is_speaking = False

    def decode_ulaw(self, data):
        """Decodifica G.711 u-law a PCM"""
        # Tabla de decodificación u-law
        ULAW_TABLE = [
            -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
            -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
            -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
            -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
            -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
            -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
            -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
            -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
            -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
            -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
            -876, -844, -812, -780, -748, -716, -684, -652,
            -620, -588, -556, -524, -492, -460, -428, -396,
            -372, -356, -340, -324, -308, -292, -276, -260,
            -244, -228, -212, -196, -180, -164, -148, -132,
            -120, -112, -104, -96, -88, -80, -72, -64,
            -56, -48, -40, -32, -24, -16, -8, 0,
            32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
            23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
            15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
            11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
            7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
            5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
            3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
            2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
            1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
            1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
            876, 844, 812, 780, 748, 716, 684, 652,
            620, 588, 556, 524, 492, 460, 428, 396,
            372, 356, 340, 324, 308, 292, 276, 260,
            244, 228, 212, 196, 180, 164, 148, 132,
            120, 112, 104, 96, 88, 80, 72, 64,
            56, 48, 40, 32, 24, 16, 8, 0
        ]

        decoded = []
        for byte in data:
            decoded.append(ULAW_TABLE[byte])
        return np.array(decoded, dtype=np.int16)

    def process_audio(self, audio_base64):
        """Procesa audio y detecta si hay voz"""
        try:
            # Decodificar base64
            audio_bytes = base64.b64decode(audio_base64)

            # Convertir u-law a PCM
            pcm_data = self.decode_ulaw(audio_bytes)

            # Calcular energía RMS
            energy = np.sqrt(np.mean(pcm_data.astype(np.float32) ** 2))

            # Detectar voz basado en energía
            if energy > self.energy_threshold:
                self.speech_counter += 1
                self.silence_counter = 0

                if self.speech_counter >= self.speech_frames and not self.is_speaking:
                    self.is_speaking = True
                    return "speech_started"
            else:
                self.silence_counter += 1
                self.speech_counter = 0

                if self.silence_counter >= self.silence_frames and self.is_speaking:
                    self.is_speaking = False
                    return "speech_stopped"

            return "no_change"

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return "no_change"


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
    BUFFER_SIZE_MS = 100  # Buffer aún más pequeño
    CHUNK_DURATION_MS = 20  # Enviar chunks más pequeños

    # Para rastrear el audio enviado
    sent_audio_timestamp = 0
    mark_counter = 0
    pending_marks = {}  # Track marks that haven't been confirmed

    # VAD local para detección inmediata
    local_vad = SimpleVAD(energy_threshold=1000)
    local_user_speaking = False

    # Control de pausa de transmisión
    transmission_paused = False

    async def send_audio_to_twilio():
        nonlocal sent_audio_timestamp, mark_counter, transmission_paused

        while not stop_event.is_set():
            try:
                # Si la transmisión está pausada, no enviar nada
                if transmission_paused:
                    await asyncio.sleep(0.001)
                    continue

                if audio_queue:
                    item = audio_queue.popleft()

                    # Verificar si este audio pertenece a una respuesta cancelada
                    if item['response_id'] in cancelled_response_ids:
                        continue

                    # Verificar de nuevo si debemos pausar
                    if transmission_paused:
                        audio_queue.appendleft(item)
                        continue

                    # Calcular si debemos enviar este audio basado en el buffer
                    current_time = time.time() * 1000
                    if sent_audio_timestamp - current_time > BUFFER_SIZE_MS:
                        # Demasiado audio en buffer, esperar
                        audio_queue.appendleft(item)  # Devolver a la cola
                        await asyncio.sleep(0.005)
                        continue

                    # Enviar el audio
                    await websocket.send_text(json.dumps(item['media_message']))

                    # Enviar mark después de cada chunk de audio para rastrear con precisión
                    mark_counter += 1
                    mark_name = f"resp_{item['response_id']}_mark_{mark_counter}"
                    mark_message = {
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {
                            "name": mark_name
                        }
                    }
                    await websocket.send_text(json.dumps(mark_message))

                    # Registrar este mark como pendiente
                    pending_marks[mark_name] = {
                        'response_id': item['response_id'],
                        'timestamp': current_time
                    }

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
            nonlocal stream_sid, pending_marks, local_user_speaking, transmission_paused
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
                        payload = message["media"]["payload"]

                        # VAD LOCAL - Detectar voz inmediatamente
                        vad_result = local_vad.process_audio(payload)

                        if vad_result == "speech_started" and not local_user_speaking:
                            local_user_speaking = True
                            logger.info("LOCAL VAD: User started speaking - PAUSING transmission")

                            # PAUSAR INMEDIATAMENTE la transmisión
                            transmission_paused = True

                            # Si el asistente está hablando, interrumpir
                            if assistant_speaking:
                                # Limpiar la cola local
                                audio_queue.clear()

                                # Enviar silencio para sobrescribir buffer
                                await send_silence_burst(200)

                                # Clear en Twilio
                                if stream_sid:
                                    await websocket.send_text(json.dumps({
                                        "event": "clear",
                                        "streamSid": stream_sid
                                    }))

                                # Avisar a OpenAI que cancele
                                await openai_ws.send(json.dumps({"type": "response.cancel"}))
                                logger.info("Interruption signal sent to OpenAI")

                        elif vad_result == "speech_stopped" and local_user_speaking:
                            local_user_speaking = False
                            transmission_paused = False
                            logger.info("LOCAL VAD: User stopped speaking - RESUMING transmission")

                        # Siempre enviar audio a OpenAI para su procesamiento
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": payload
                        }
                        await openai_ws.send(json.dumps(audio_append))

                    elif message["event"] == "mark":
                        # Twilio confirmó que reprodujo un mark
                        mark_data = message.get("mark", {})
                        mark_name = mark_data.get("name", "")

                        if mark_name in pending_marks:
                            mark_info = pending_marks.pop(mark_name)
                            # Este audio fue reproducido completamente
                            if mark_info['response_id'] not in cancelled_response_ids:
                                logger.debug(f"Audio confirmed played: {mark_name}")
                            else:
                                # Este mark llegó después de un clear event
                                logger.debug(f"Mark received after clear (audio not played): {mark_name}")

                        # Limpiar marks antiguos (más de 10 segundos)
                        current_time = time.time() * 1000
                        old_marks = [k for k, v in pending_marks.items()
                                     if current_time - v['timestamp'] > 10000]
                        for mark in old_marks:
                            del pending_marks[mark]

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
            nonlocal user_speaking, assistant_speaking, current_response_id, transmission_paused

            try:
                async for message in openai_ws:
                    response = json.loads(message)

                    if response["type"] == "input_audio_buffer.speech_started":
                        user_speaking = True
                        logger.info("OpenAI VAD: User speaking confirmed")

                        # Asegurar que la transmisión esté pausada
                        transmission_paused = True

                        if assistant_speaking:
                            # Marcar la respuesta actual como cancelada
                            cancelled_response_ids.add(current_response_id)
                            assistant_speaking = False

                    elif response["type"] == "input_audio_buffer.speech_stopped":
                        user_speaking = False
                        # Solo reanudar si el VAD local también indica silencio
                        if not local_user_speaking:
                            transmission_paused = False
                        logger.info("OpenAI VAD: User stopped speaking")

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