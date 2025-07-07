import asyncio
import websockets
import json
import base64
import logging
import os
import ssl
import audioop
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdc-call")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

def mulaw_to_pcm16_24khz(mulaw):
    """Convert 8kHz mu-law to 24kHz PCM16 mono"""
    pcm8 = audioop.ulaw2lin(mulaw, 2)
    pcm24 = audioop.ratecv(pcm8, 2, 1, 8000, 24000, None)[0]
    return pcm24

def pcm16_24khz_to_mulaw(pcm):
    """Convert 24kHz PCM16 to 8kHz mu-law mono, with gain boost"""
    # Opcional: Amplifica el volumen aquí (+6dB)
    pcm = audioop.mul(pcm, 2, 2.0)
    pcm8 = audioop.ratecv(pcm, 2, 1, 24000, 8000, None)[0]
    mulaw = audioop.lin2ulaw(pcm8, 2)
    return mulaw

@app.post("/voice")
async def voice(request: Request):
    host = request.headers.get("host")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="wss://{host}/twilio-stream" />
    </Start>
    <Say language="es-ES" voice="alice">Conectando con tu asistente de inteligencia artificial. Un momento por favor.</Say>
    <Pause length="90" />
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    openai_ws = None
    try:
        # Abrir WebSocket a OpenAI
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
        ssl_ctx = ssl.create_default_context()
        openai_ws = await websockets.connect(
            OPENAI_WS_URL,
            extra_headers=headers,
            ssl=ssl_ctx
        )

        # Configurar sesión OpenAI (español, respuesta hablada)
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": "Eres un asistente telefónico en español, responde de forma natural y concisa.",
                "voice": "alloy",
                "input_audio_format": {"encoding": "pcm16", "sample_rate": 24000, "channels": 1},
                "output_audio_format": {"encoding": "pcm16", "sample_rate": 24000, "channels": 1},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 700
                },
                "input_audio_transcription": {"model": "whisper-1"},
                "temperature": 0.6
            }
        }
        await openai_ws.send(json.dumps(session_config))
        logger.info("OpenAI session configurada y websocket abierto.")

        # Empieza el pipe entre Twilio y OpenAI
        async def openai_to_twilio():
            async for message in openai_ws:
                evt = json.loads(message)
                # Envía solo audio chunks (delta)
                if evt.get("type") == "response.audio.delta" and evt.get("delta"):
                    pcm24 = base64.b64decode(evt["delta"])
                    mulaw = pcm16_24khz_to_mulaw(pcm24)
                    # Devuelve en formato Twilio MediaStream
                    chunk_size = 160  # 20ms a 8kHz
                    for i in range(0, len(mulaw), chunk_size):
                        payload = base64.b64encode(mulaw[i:i+chunk_size]).decode()
                        await ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": "OpenAI",  # Twilio espera este campo
                            "media": {"payload": payload}
                        }))

        openai_task = asyncio.create_task(openai_to_twilio())

        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            if data.get("event") == "media":
                mulaw = base64.b64decode(data["media"]["payload"])
                pcm24 = mulaw_to_pcm16_24khz(mulaw)
                # Envía audio a OpenAI
                await openai_ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm24).decode()
                }))
            elif data.get("event") == "start":
                logger.info("Stream Twilio iniciado")
            elif data.get("event") == "stop":
                logger.info("Stream Twilio finalizado")
                break
    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error en pipeline Twilio/OpenAI: {e}")
    finally:
        if openai_ws:
            await openai_ws.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
