import os
import json
import base64
import asyncio
import websockets
import traceback
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from jania import env

# ============================================
# CONFIGURACIÓN
# ============================================
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_VOICE = env("OPENAI_VOICE", "alloy")
SYSTEM_MESSAGE = env("SYSTEM_MESSAGE",
                     "Eres un asistente telefónico amigable. Habla en español de forma natural y conversacional.")
PORT = int(env("PORT", "8080"))

# ============================================

app = FastAPI()


@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    """Maneja las llamadas entrantes de Twilio"""
    host = request.headers.get("host")
    # Cloud Run siempre usa HTTPS/WSS
    protocol = "wss" if ("run.app" in host or "https" in str(request.url)) else "ws"

    response_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{protocol}://{host}/media-stream" />
    </Connect>
</Response>"""

    return Response(content=response_xml, media_type="application/xml")


@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Maneja el stream de audio bidireccional"""
    await websocket.accept()

    openai_ws = None
    stream_sid = None

    try:
        # Conectar a OpenAI
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        openai_ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            extra_headers=headers
        )

        # Configurar sesión
        await openai_ws.send(json.dumps({
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
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200
                },
                "temperature": 0.8
            }
        }))

        # Tareas asíncronas
        async def handle_twilio():
            nonlocal stream_sid
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)

                    if data.get("event") == "start":
                        stream_sid = data["start"]["streamSid"]
                        # Iniciar conversación
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": "Saluda amablemente en español"}]
                            }
                        }))
                        await openai_ws.send(json.dumps({"type": "response.create"}))

                    elif data.get("event") == "media" and data.get("media", {}).get("payload"):
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"]
                        }))

                    elif data.get("event") == "stop":
                        break

            except Exception as e:
                print(f"Error en handle_twilio: {e}")

        async def handle_openai():
            try:
                async for message in openai_ws:
                    response = json.loads(message)

                    if response.get("type") == "response.audio.delta" and stream_sid:
                        audio = response.get("delta", "")
                        if audio:
                            await websocket.send_text(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": audio}
                            }))

                    elif response.get("type") == "input_audio_buffer.speech_stopped":
                        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                    elif response.get("type") == "input_audio_buffer.committed":
                        await openai_ws.send(json.dumps({"type": "response.create"}))

            except Exception as e:
                print(f"Error en handle_openai: {e}")

        # Ejecutar ambas tareas
        await asyncio.gather(handle_twilio(), handle_openai())

    except Exception as e:
        print(f"Error general: {e}")
        traceback.print_exc()
    finally:
        if openai_ws:
            await openai_ws.close()
        await websocket.close()


@app.get("/")
async def root():
    return {"status": "running", "port": PORT}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)