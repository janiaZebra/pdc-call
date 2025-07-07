import asyncio
import websockets
import json
import base64
import logging
import os
import ssl
import audioop
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "TU_API_KEY_AQUI")
OPENAI_WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
logging.basicConfig(level=logging.DEBUG)  # Cambia a DEBUG para mayor detalle
logger = logging.getLogger("twilio-openai-bridge")

def mulaw_to_pcm16_24khz(mulaw):
    try:
        logger.debug(f"Convirtiendo {len(mulaw)} bytes mu-law a PCM 24kHz")
        pcm8 = audioop.ulaw2lin(mulaw, 2)
        pcm24 = audioop.ratecv(pcm8, 2, 1, 8000, 24000, None)[0]
        return pcm24
    except Exception as e:
        logger.error(f"Error en mulaw_to_pcm16_24khz: {e}")
        return b''

def pcm16_24khz_to_mulaw(pcm):
    try:
        logger.debug(f"Convirtiendo {len(pcm)} bytes PCM 24kHz a mu-law 8kHz")
        pcm = audioop.mul(pcm, 2, 2.0)
        pcm8 = audioop.ratecv(pcm, 2, 1, 24000, 8000, None)[0]
        mulaw = audioop.lin2ulaw(pcm8, 2)
        return mulaw
    except Exception as e:
        logger.error(f"Error en pcm16_24khz_to_mulaw: {e}")
        return b''

@app.post("/voice")
async def voice(request: Request):
    host = request.headers.get("host", "localhost:8000")
    logger.info(f"POST /voice recibido de {host}")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="wss://{host}/twilio-stream" />
    </Start>
    <Say language="es-ES" voice="alice">Conectando con tu asistente virtual. Un momento, por favor.</Say>
    <Pause length="60" />
</Response>"""
    logger.debug(f"Enviando TwiML: {twiml}")
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("Twilio MediaStream conectado (WebSocket aceptado)")

    # Conecta a OpenAI Realtime WebSocket
    logger.info("Conectando a OpenAI Realtime API websocket...")
    try:
        sslctx = ssl.create_default_context()
        openai_ws = await websockets.connect(
            OPENAI_WS_URL,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            },
            ssl=sslctx
        )
    except Exception as e:
        logger.error(f"ERROR: No se pudo conectar a OpenAI WebSocket: {e}")
        await ws.close()
        return

    logger.info("WebSocket a OpenAI Realtime abierto correctamente")

    # Configura la sesión en OpenAI
    session_config = {
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "instructions": "Eres un asistente de voz telefónico. Sé breve y claro.",
            "voice": "alloy",
            "input_audio_format": {"encoding": "pcm16", "sample_rate": 24000, "channels": 1},
            "output_audio_format": {"encoding": "pcm16", "sample_rate": 24000, "channels": 1},
            "temperature": 0.6,
            "max_response_output_tokens": 1024
        }
    }
    await openai_ws.send(json.dumps(session_config))
    logger.info("Configuración de sesión enviada a OpenAI")

    async def twilio_to_openai():
        """Recibe audio de Twilio y lo envía a OpenAI"""
        try:
            while True:
                msg = await ws.receive_text()
                logger.debug(f"Mensaje recibido de Twilio: {msg[:200]}...")
                data = json.loads(msg)
                if data.get("event") == "media":
                    mulaw = base64.b64decode(data["media"]["payload"])
                    logger.debug(f"Audio recibido de Twilio (mu-law): {len(mulaw)} bytes")
                    pcm24 = mulaw_to_pcm16_24khz(mulaw)
                    if not pcm24:
                        logger.error("PCM24 vacío, algo falló en la conversión")
                        continue
                    await openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm24).decode()
                    }))
                    logger.debug(f"Audio PCM enviado a OpenAI: {len(pcm24)} bytes")
                elif data.get("event") == "stop":
                    logger.info("Twilio MediaStream finalizado (evento stop recibido)")
                    break
                elif data.get("event") == "connected":
                    logger.info("Twilio MediaStream evento: connected")
                elif data.get("event") == "start":
                    logger.info("Twilio MediaStream evento: start")
                else:
                    logger.warning(f"Evento desconocido desde Twilio: {data.get('event')}")
        except Exception as e:
            logger.error(f"Error en Twilio->OpenAI: {e}")

    async def openai_to_twilio():
        """Recibe audio de OpenAI y lo envía de vuelta a Twilio"""
        try:
            async for msg in openai_ws:
                logger.debug(f"Mensaje recibido de OpenAI: {msg[:200]}...")
                event = json.loads(msg)
                logger.info(f"Evento OpenAI: {event.get('type')}")
                if event.get("type") == "response.audio.delta" and event.get("delta"):
                    pcm24 = base64.b64decode(event["delta"])
                    logger.debug(f"Audio recibido de OpenAI (PCM): {len(pcm24)} bytes")
                    mulaw = pcm16_24khz_to_mulaw(pcm24)
                    if not mulaw:
                        logger.error("mulaw vacío, error en conversión OpenAI->Twilio")
                        continue
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(mulaw).decode()
                        }
                    }))
                    logger.debug(f"Audio mu-law enviado a Twilio: {len(mulaw)} bytes")
                elif event.get("type") == "error":
                    logger.error(f"OpenAI ERROR: {event}")
                elif event.get("type") == "session.created":
                    logger.info("Sesión OpenAI creada correctamente")
                elif event.get("type") == "response.done":
                    logger.info("OpenAI: respuesta completada (response.done)")
                elif event.get("type") == "response.audio_transcript.delta":
                    logger.info(f"Transcript parcial: {event.get('delta')}")
                elif event.get("type") == "conversation.item.input_audio_transcription.completed":
                    logger.info(f"Transcripción completa: {event.get('transcript')}")
                else:
                    logger.debug(f"Evento OpenAI no manejado: {event.get('type')}")
        except Exception as e:
            logger.error(f"Error en OpenAI->Twilio: {e}")

    try:
        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio()
        )
    finally:
        await openai_ws.close()
        logger.info("WebSocket a OpenAI cerrado")
        await ws.close()
        logger.info("Twilio stream cerrado.")

@app.get("/health")
async def health():
    logger.info("Chequeo de salud recibido")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor Uvicorn en 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
