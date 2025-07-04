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
OPENAI_VOICE = env("OPENAI_VOICE", "alloy")  # Opciones: alloy, echo, shimmer

# Assistant Configuration
SYSTEM_MESSAGE = env("SYSTEM_MESSAGE", """
Eres un asistente telefónico profesional y amigable. 
Habla únicamente en español de forma natural y conversacional.
Sé conciso pero útil. No uses jerga técnica.
Si no entiendes algo, pide que lo repitan amablemente.
""").strip()

# Twilio Messages
TWILIO_INITIAL_MESSAGE = env("TWILIO_INITIAL_MESSAGE", "Hola, conectándote con nuestro asistente.")

# Server Configuration
PORT = int(env("PORT", "8080"))  # Cloud Run usa PORT

# Audio Configuration
VOICE_TEMPERATURE = float(env("VOICE_TEMPERATURE", "0.8"))
VAD_THRESHOLD = float(env("VAD_THRESHOLD", "0.5"))
VAD_SILENCE_MS = int(env("VAD_SILENCE_MS", "500"))

# Logging Configuration
LOG_EVENT_TYPES = [
    'response.audio.delta',
    'conversation.item.input_audio_transcription.completed',
    'response.done'
]

# ============================================
# FIN DE CONFIGURACIÓN
# ============================================

app = FastAPI(title="OpenAI-Twilio Bridge")


@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    """Webhook que Twilio llama cuando recibe una llamada"""
    # Obtener la URL base del request
    host = request.headers.get("host")
    # Cloud Run siempre usa HTTPS
    protocol = "wss" if "run.app" in host or "https" in str(request.url) else "ws"
    stream_url = f"{protocol}://{host}/media-stream"

    print(f"Llamada entrante - Stream URL: {stream_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say language="es-ES">{TWILIO_INITIAL_MESSAGE}</Say>
        <Connect>
            <Stream url="{stream_url}" />
        </Connect>
    </Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """WebSocket endpoint para manejar el stream de audio de Twilio"""
    print("Cliente conectando...")
    await websocket.accept()
    print("Cliente Twilio conectado vía WebSocket")

    openai_ws = None
    stream_sid = None

    try:
        # Verificar API key
        if not OPENAI_API_KEY:
            print("ERROR: OPENAI_API_KEY no está configurada")
            await websocket.close()
            return

        print(f"API Key presente: {OPENAI_API_KEY[:10]}...")

        # Conectar a OpenAI Realtime API
        print("Conectando a OpenAI Realtime API...")
        try:
            openai_ws = await websockets.connect(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                },
                ping_interval=20,
                ping_timeout=10
            )
            print("✓ Conectado a OpenAI Realtime API")
        except Exception as e:
            print(f"ERROR conectando a OpenAI: {str(e)}")
            print(f"Tipo de error: {type(e).__name__}")
            await websocket.close()
            return

        # Enviar configuración de sesión
        session_update = {
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
                "temperature": VOICE_TEMPERATURE
            }
        }

        print("Enviando configuración de sesión...")
        await openai_ws.send(json.dumps(session_update))
        print("Configuración enviada")

        # NO enviar mensaje inicial por ahora, dejar que el usuario hable primero

        async def receive_from_twilio():
            """Recibe audio de Twilio y lo envía a OpenAI"""
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)

                    if data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream started {stream_sid}")

                    elif data['event'] == 'media':
                        # El audio ya viene en g711 ulaw base64 de Twilio
                        audio_payload = data['media']['payload']
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": audio_payload
                        }
                        await openai_ws.send(json.dumps(audio_append))

                    elif data['event'] == 'stop':
                        print(f"Stream {stream_sid} ended")
                        break

            except WebSocketDisconnect:
                print("Client disconnected.")
            except Exception as e:
                print(f"Error receiving from Twilio: {e}")

        async def send_to_twilio():
            """Recibe eventos de OpenAI y envía audio a Twilio"""
            nonlocal stream_sid
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get('type') == 'session.updated':
                        print("Session updated successfully")

                    if response['type'] == 'response.audio.delta' and stream_sid:
                        # Audio de OpenAI ya viene en g711_ulaw base64
                        audio_delta = response.get('delta', '')
                        if audio_delta:
                            audio_message = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_delta
                                }
                            }
                            await websocket.send_text(json.dumps(audio_message))

                    # Manejo de transcripciones
                    if response['type'] == 'conversation.item.input_audio_transcription.completed':
                        print(f"User said: {response.get('transcript', '')}")

                    if response['type'] == 'response.audio_transcript.done':
                        print(f"Assistant said: {response.get('transcript', '')}")

                    # Cuando se detecta fin de habla, hacer commit
                    if response['type'] == 'input_audio_buffer.speech_stopped':
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.commit"
                        }))

                    # Generar respuesta después del commit
                    if response['type'] == 'input_audio_buffer.committed':
                        await openai_ws.send(json.dumps({
                            "type": "response.create"
                        }))

                    if response['type'] == 'response.done':
                        print("Response completed")

                    # Manejo de errores
                    if response['type'] == 'error':
                        print(f"Error from OpenAI: {response}")

            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        # Ejecutar ambas tareas concurrentemente
        await asyncio.gather(
            receive_from_twilio(),
            send_to_twilio()
        )

    except Exception as e:
        print(f"ERROR GENERAL en WebSocket: {str(e)}")
        print(f"Tipo: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    finally:
        print("Limpiando conexiones...")
        if openai_ws:
            try:
                await openai_ws.close()
                print("OpenAI WebSocket cerrado")
            except:
                pass
        try:
            await websocket.close()
            print("Twilio WebSocket cerrado")
        except:
            pass


async def send_initial_conversation_item(openai_ws):
    """Envía el mensaje inicial para que OpenAI salude"""
    # Primero, enviar un mensaje para establecer contexto
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Hola"
                }
            ]
        }
    }

    await openai_ws.send(json.dumps(initial_conversation_item))

    # Luego generar respuesta
    await openai_ws.send(json.dumps({
        "type": "response.create"
    }))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "OpenAI-Twilio Bridge",
        "endpoints": {
            "/": "Health check",
            "/incoming-call": "POST - Webhook para llamadas entrantes de Twilio",
            "/media-stream": "WebSocket - Stream de audio bidireccional"
        }
    }


@app.get("/health")
async def health():
    """Health check para Cloud Run"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    print("=" * 50)
    print("OpenAI-Twilio Realtime Voice Bridge")
    print("=" * 50)
    print(f"Puerto: {PORT}")
    print(f"Voz: {OPENAI_VOICE}")
    print(f"API Key configurada: {'✓' if OPENAI_API_KEY else '✗'}")
    print("=" * 50)

    if not OPENAI_API_KEY:
        print("⚠️  ADVERTENCIA: OPENAI_API_KEY no está configurada")
        print("Configura la variable de entorno OPENAI_API_KEY")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )