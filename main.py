import os
import json
import base64
import asyncio
import websockets
import traceback
import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from jania import env
import aiohttp

# ============================================
# CONFIGURACIÓN - Edita estos valores
# ============================================
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_VOICE = env("OPENAI_VOICE", "alloy")  # alloy, echo, fable, onyx, nova, shimmer
SYSTEM_MESSAGE = env("SYSTEM_MESSAGE",
                     "Eres un asistente telefónico amigable. Habla en español de forma natural y conversacional.")
PORT = int(env("PORT", "8080"))
LOG_LEVEL = env("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR

# Configuración de OpenAI Realtime
OPENAI_MODEL = "gpt-4o-realtime-preview-2024-10-01"
OPENAI_TEMPERATURE = float(env("OPENAI_TEMPERATURE", "0.8"))
VAD_THRESHOLD = float(env("VAD_THRESHOLD", "0.5"))
SILENCE_DURATION_MS = int(env("SILENCE_DURATION_MS", "500"))
PREFIX_PADDING_MS = int(env("PREFIX_PADDING_MS", "300"))

# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    """Maneja las llamadas entrantes de Twilio"""
    logger.info("📞 Llamada entrante recibida")

    host = request.headers.get("host")
    logger.info(f"Host: {host}")

    # Cloud Run siempre usa HTTPS/WSS
    protocol = "wss" if ("run.app" in host or "https" in str(request.url)) else "ws"
    websocket_url = f"{protocol}://{host}/media-stream"

    logger.info(f"🔗 URL WebSocket: {websocket_url}")

    response_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{websocket_url}" />
    </Connect>
</Response>"""

    logger.info("✅ Respuesta TwiML enviada")
    return Response(content=response_xml, media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Maneja el stream de audio bidireccional"""
    logger.info("🎧 Iniciando conexión WebSocket")
    await websocket.accept()
    logger.info("✅ WebSocket aceptado")

    session = None
    stream_sid = None

    try:
        # Conectar a OpenAI usando aiohttp
        logger.info("🤖 Conectando a OpenAI Realtime API...")

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        session = aiohttp.ClientSession()
        openai_ws = await session.ws_connect(
            f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
            headers=headers
        )

        logger.info("✅ Conectado a OpenAI Realtime API")

        # Configurar sesión de OpenAI
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": SYSTEM_MESSAGE,
                "voice": OPENAI_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": VAD_THRESHOLD,
                    "prefix_padding_ms": PREFIX_PADDING_MS,
                    "silence_duration_ms": SILENCE_DURATION_MS
                },
                "temperature": OPENAI_TEMPERATURE
            }
        }

        await openai_ws.send_str(json.dumps(session_config))
        logger.info("🔧 Configuración de sesión enviada a OpenAI")

        # Tareas asíncronas
        async def handle_twilio():
            nonlocal stream_sid
            try:
                logger.info("🔄 Iniciando manejo de mensajes de Twilio")
                async for message in websocket.iter_text():
                    try:
                        data = json.loads(message)
                        event_type = data.get("event")

                        logger.debug(f"📨 Evento Twilio: {event_type}")

                        if event_type == "start":
                            stream_sid = data["start"]["streamSid"]
                            logger.info(f"🎬 Stream iniciado - SID: {stream_sid}")

                            # Iniciar conversación con saludo
                            greeting_message = {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": "Saluda amablemente en español"}]
                                }
                            }
                            await openai_ws.send_str(json.dumps(greeting_message))
                            await openai_ws.send_str(json.dumps({"type": "response.create"}))
                            logger.info("👋 Saludo inicial enviado a OpenAI")

                        elif event_type == "media":
                            media_data = data.get("media", {})
                            payload = media_data.get("payload")

                            if payload:
                                logger.debug(f"🎵 Audio recibido - Tamaño: {len(payload)} bytes")

                                audio_message = {
                                    "type": "input_audio_buffer.append",
                                    "audio": payload
                                }
                                await openai_ws.send_str(json.dumps(audio_message))

                        elif event_type == "stop":
                            logger.info("⏹️  Stream detenido por Twilio")
                            break

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Error decodificando JSON de Twilio: {e}")
                    except Exception as e:
                        logger.error(f"❌ Error procesando mensaje de Twilio: {e}")

            except Exception as e:
                logger.error(f"❌ Error en handle_twilio: {e}")
                traceback.print_exc()

        async def handle_openai():
            try:
                logger.info("🔄 Iniciando manejo de respuestas de OpenAI")
                async for msg in openai_ws:
                    try:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            response = json.loads(msg.data)
                            response_type = response.get("type")

                            logger.debug(f"📨 Respuesta OpenAI: {response_type}")

                            if response_type == "response.audio.delta" and stream_sid:
                                audio = response.get("delta", "")
                                if audio:
                                    logger.debug(f"🔊 Enviando audio a Twilio - Tamaño: {len(audio)} bytes")

                                    twilio_message = {
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {"payload": audio}
                                    }
                                    await websocket.send_text(json.dumps(twilio_message))

                            elif response_type == "input_audio_buffer.speech_stopped":
                                logger.info("🔇 Speech stopped detectado")
                                await openai_ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))

                            elif response_type == "input_audio_buffer.committed":
                                logger.info("✅ Audio buffer committed")
                                await openai_ws.send_str(json.dumps({"type": "response.create"}))

                            elif response_type == "conversation.item.input_audio_transcription.completed":
                                transcript = response.get("transcript", "")
                                logger.info(f"📝 Transcripción: {transcript}")

                            elif response_type == "response.done":
                                logger.info("✅ Respuesta completada")

                            elif response_type == "error":
                                error_details = response.get("error", {})
                                logger.error(f"❌ Error de OpenAI: {error_details}")

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"❌ Error WebSocket OpenAI: {msg.data}")
                            break

                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Error decodificando JSON de OpenAI: {e}")
                    except Exception as e:
                        logger.error(f"❌ Error procesando mensaje de OpenAI: {e}")

            except Exception as e:
                logger.error(f"❌ Error en handle_openai: {e}")
                traceback.print_exc()

        # Ejecutar ambas tareas
        logger.info("🚀 Iniciando tareas asíncronas")
        await asyncio.gather(handle_twilio(), handle_openai())

    except Exception as e:
        logger.error(f"❌ Error general en media stream: {e}")
        traceback.print_exc()
    finally:
        logger.info("🧹 Limpiando recursos")
        if session:
            await session.close()
        try:
            await websocket.close()
        except:
            pass
        logger.info("✅ Recursos limpiados")


@app.get("/")
async def root():
    """Endpoint de salud"""
    return {
        "status": "running",
        "port": PORT,
        "model": OPENAI_MODEL,
        "voice": OPENAI_VOICE
    }


@app.get("/health")
async def health_check():
    """Health check para Cloud Run"""
    return {"status": "healthy", "timestamp": "2025-07-04"}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Iniciando servidor en puerto {PORT}")
    logger.info(f"🤖 Modelo OpenAI: {OPENAI_MODEL}")
    logger.info(f"🎙️ Voz: {OPENAI_VOICE}")
    logger.info(f"🌡️ Temperatura: {OPENAI_TEMPERATURE}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)