import asyncio
import base64
import json
import os
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
import xml.etree.ElementTree as ET
from jania import env

# ============================================
# CONFIGURACIÓN - MODIFICA ESTOS VALORES
# ============================================

# OpenAI Configuration
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-10-01")
OPENAI_VOICE = env("OPENAI_VOICE", "alloy")  # Opciones: alloy, echo, shimmer

# Assistant Configuration
SYSTEM_MESSAGE = env("SYSTEM_MESSAGE", """
Eres un asistente telefónico profesional y amigable. 
Habla únicamente en español de forma natural y conversacional.
Sé conciso pero útil. No uses jerga técnica.
Si no entiendes algo, pide que lo repitan amablemente.
""").strip()

# Twilio Messages
TWILIO_INITIAL_MESSAGE = env("TWILIO_INITIAL_MESSAGE", "Hola, un momento por favor mientras te conecto.")
TWILIO_ERROR_MESSAGE = env("TWILIO_ERROR_MESSAGE", "Lo siento, hay un problema técnico. Por favor intenta más tarde.")

# Server Configuration
PORT = int(env("PORT", "8080"))  # Cloud Run usa PORT
SERVICE_URL = env("SERVICE_URL", "")  # Se auto-detecta en Cloud Run

# Audio Configuration
VOICE_TEMPERATURE = float(env("VOICE_TEMPERATURE", "0.8"))
VAD_THRESHOLD = float(env("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_MS = int(env("VAD_SILENCE_MS", "500"))

# ============================================
# FIN DE CONFIGURACIÓN
# ============================================

app = FastAPI(title="OpenAI-Twilio Bridge")


def get_service_url(request: Request) -> str:
    """Obtiene la URL del servicio automáticamente"""
    if SERVICE_URL:
        return SERVICE_URL

    # Detectar URL en Cloud Run
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_host:
        # Cloud Run siempre usa HTTPS
        return f"https://{forwarded_host}"

    # Fallback para desarrollo local
    return f"{request.url.scheme}://{request.headers.get('host', 'localhost:8080')}"


@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Webhook que Twilio llama cuando recibe una llamada"""
    service_url = get_service_url(request)
    websocket_url = f"{service_url}/media-stream".replace("http://", "ws://").replace("https://", "wss://")

    print(f"Llamada entrante - WebSocket URL: {websocket_url}")

    response = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="es-ES">{TWILIO_INITIAL_MESSAGE}</Say>
        <Connect>
            <Stream url="{websocket_url}" />
        </Connect>
        <Say language="es-ES">{TWILIO_ERROR_MESSAGE}</Say>
    </Response>"""

    return Response(content=response, media_type="text/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """WebSocket endpoint para manejar el stream de audio de Twilio"""
    await websocket.accept()
    print("Cliente Twilio conectado vía WebSocket")

    openai_ws = None
    stream_sid = None
    call_sid = None

    try:
        # Conectar a OpenAI Realtime API
        print("Conectando a OpenAI Realtime API...")
        try:
            openai_ws = await websockets.connect(
                f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
            )
            print("✓ Conectado a OpenAI")
        except Exception as e:
            print(f"✗ Error conectando a OpenAI: {e}")
            # Enviar mensaje de error a Twilio
            error_message = {
                "event": "clear"
            }
            await websocket.send_text(json.dumps(error_message))
            raise

        # Configurar la sesión de OpenAI
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": SYSTEM_MESSAGE,
                "voice": OPENAI_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": VAD_THRESHOLD,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": VAD_SILENCE_MS
                },
                "temperature": VOICE_TEMPERATURE,
                "max_response_output_tokens": 4096
            }
        }

        await openai_ws.send(json.dumps(session_config))
        print("✓ Sesión configurada")

        # Esperar confirmación de sesión creada
        session_created = False
        while not session_created:
            response = await openai_ws.recv()
            if isinstance(response, str):
                data = json.loads(response)
                if data.get('type') == 'session.created':
                    session_created = True
                    print("✓ Sesión OpenAI creada")
                    break

        # Crear un saludo inicial
        greeting = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Di 'Hola, ¿en qué puedo ayudarte?' de forma amigable"
                }]
            }
        }
        await openai_ws.send(json.dumps(greeting))
        await openai_ws.send(json.dumps({"type": "response.create"}))

        async def handle_twilio_messages():
            """Procesa mensajes de Twilio y envía audio a OpenAI"""
            nonlocal stream_sid, call_sid

            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    event_type = data.get('event')

                    if event_type == 'start':
                        stream_sid = data['start']['streamSid']
                        call_sid = data['start']['callSid']
                        print(f"✓ Stream iniciado - Call SID: {call_sid}")

                    elif event_type == 'media' and data['media'].get('payload'):
                        # Reenviar audio a OpenAI (ya viene en g711_ulaw base64)
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))

                    elif event_type == 'stop':
                        print(f"✓ Llamada terminada - Call SID: {call_sid}")
                        break

            except WebSocketDisconnect:
                print("✗ Twilio desconectado")
            except Exception as e:
                print(f"✗ Error procesando mensajes de Twilio: {e}")

        async def handle_openai_messages():
            """Procesa respuestas de OpenAI y envía audio a Twilio"""
            try:
                async for message in openai_ws:
                    if isinstance(message, str):
                        response = json.loads(message)
                        event_type = response.get('type')

                        # Log de eventos importantes
                        if event_type == 'error':
                            print(f"✗ Error de OpenAI: {response}")
                            continue

                        # Enviar audio a Twilio
                        if event_type == 'response.audio.delta' and stream_sid:
                            audio_delta = response.get('delta')
                            if audio_delta:
                                media_message = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": audio_delta
                                    }
                                }
                                await websocket.send_text(json.dumps(media_message))

                        # Manejar detección de voz
                        elif event_type == 'input_audio_buffer.speech_stopped':
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.commit"
                            }))

                        elif event_type == 'input_audio_buffer.committed':
                            await openai_ws.send(json.dumps({
                                "type": "response.create"
                            }))

                        # Log de transcripciones
                        elif event_type == 'conversation.item.input_audio_transcription.completed':
                            transcript = response.get('transcript', '')
                            if transcript:
                                print(f"👤 Usuario: {transcript}")

                        elif event_type == 'response.audio_transcript.done':
                            transcript = response.get('transcript', '')
                            if transcript:
                                print(f"🤖 Asistente: {transcript}")

            except Exception as e:
                print(f"✗ Error procesando mensajes de OpenAI: {e}")

        # Ejecutar ambos handlers concurrentemente
        await asyncio.gather(
            handle_twilio_messages(),
            handle_openai_messages()
        )

    except Exception as e:
        print(f"✗ Error general en media stream: {e}")
    finally:
        print("Cerrando conexiones...")
        if openai_ws:
            await openai_ws.close()
        await websocket.close()


@app.get("/")
async def root():
    """Endpoint de health check"""
    return {
        "status": "running",
        "service": "OpenAI-Twilio Bridge",
        "endpoints": {
            "/": "Health check",
            "/twilio/voice": "POST - Webhook para llamadas entrantes",
            "/media-stream": "WebSocket - Stream de audio"
        },
        "configuration": {
            "model": OPENAI_MODEL,
            "voice": OPENAI_VOICE,
            "temperature": VOICE_TEMPERATURE
        }
    }


@app.get("/health")
async def health():
    """Health check para Cloud Run"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("OpenAI-Twilio Bridge")
    print("=" * 50)
    print(f"Puerto: {PORT}")
    print(f"Modelo: {OPENAI_MODEL}")
    print(f"Voz: {OPENAI_VOICE}")
    print(f"API Key configurada: {'✓' if OPENAI_API_KEY else '✗'}")
    print("=" * 50)

    if not OPENAI_API_KEY:
        print("⚠️  ADVERTENCIA: OPENAI_API_KEY no está configurada")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )