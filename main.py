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

# Configuración de logging a DEBUG para verlo
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("twilio-openai-bridge")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def mulaw_to_pcm16_24khz(mulaw: bytes) -> bytes:
    """Convierte mu-law 8kHz a PCM16 24kHz mono little-endian."""
    try:
        pcm8 = audioop.ulaw2lin(mulaw, 2)
        pcm24 = audioop.ratecv(pcm8, 2, 1, 8000, 24000, None)[0]
        return pcm24
    except Exception as e:
        logger.error(f"mulaw->pcm16_24khz error: {e}")
        return b''

def pcm16_24khz_to_mulaw(pcm: bytes) -> bytes:
    """Convierte PCM16 24kHz a mu-law 8kHz."""
    try:
        pcm8 = audioop.ratecv(pcm, 2, 1, 24000, 8000, None)[0]
        mulaw = audioop.lin2ulaw(pcm8, 2)
        return mulaw
    except Exception as e:
        logger.error(f"pcm16_24khz->mulaw error: {e}")
        return b''

@app.post("/voice")
async def voice(request: Request):
    host = request.headers.get("host", "localhost:8000")
    logger.info(f"[Twilio] POST /voice de {host}")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start><Stream url="wss://{host}/twilio-stream"/></Start>
  <Say voice="alice" language="es-ES">
    Conectando con tu asistente virtual. Un momento, por favor.
  </Say>
  <Pause length="60"/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/twilio-stream")
async def twilio_stream(ws: WebSocket):
    await ws.accept()
    logger.info("[Twilio] WebSocket conectado")

    # 1) Conectar a OpenAI Realtime
    logger.info("[OpenAI] Conectando websocket...")
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
        logger.error(f"[OpenAI] No se pudo conectar: {e}")
        await ws.close()
        return

    logger.info("[OpenAI] WebSocket abierto")

    # 2) Enviar configuración de sesión (sin VAD automático)
    session_update = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": "Eres un asistente de voz telefónico. Sé breve y claro.",
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            # Desactivamos VAD automático para controlar nosotros el commit
            "turn_detection": None,
            "temperature": 0.6,
            "max_response_output_tokens": 1024
        }
    }
    await openai_ws.send(json.dumps(session_update))
    logger.info(f"[OpenAI] session.update enviado: {session_update}")

    # Variables para control de inactividad
    last_audio_ts = asyncio.get_event_loop().time()
    stop_signal = asyncio.Event()

    async def twilio_to_openai():
        nonlocal last_audio_ts
        try:
            while not stop_signal.is_set():
                data = await ws.receive_text()
                msg = json.loads(data)
                ev = msg.get("event")
                if ev == "media":
                    payload = msg["media"]["payload"]
                    mulaw = base64.b64decode(payload)
                    pcm24 = mulaw_to_pcm16_24khz(mulaw)
                    if pcm24:
                        b64 = base64.b64encode(pcm24).decode("ascii")
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": b64
                        }))
                        logger.debug(f"[OpenAI] append audio ({len(pcm24)} bytes)")
                        last_audio_ts = asyncio.get_event_loop().time()
                elif ev == "stop":
                    logger.info("[Twilio] Evento stop recibido")
                    stop_signal.set()
                    break
                else:
                    logger.debug(f"[Twilio] Evento: {ev}")
        except Exception as e:
            logger.error(f"[Bridge] Error Twilio→OpenAI: {e}")
            stop_signal.set()

    async def openai_to_twilio():
        try:
            async for msg in openai_ws:
                ev = json.loads(msg)
                etype = ev.get("type")
                logger.info(f"[OpenAI] Evento: {etype}")
                # Log de eventos clave
                if etype in ("session.created", "input_audio_buffer.committed",
                             "response.created", "response.done"):
                    logger.info(f"[OpenAI] {etype}: {ev}")
                # Audio delta → reenvío a Twilio
                if etype == "response.audio.delta" and ev.get("delta"):
                    pcm24 = base64.b64decode(ev["delta"])
                    mulaw = pcm16_24khz_to_mulaw(pcm24)
                    b64 = base64.b64encode(mulaw).decode("ascii")
                    await ws.send_text(json.dumps({
                        "event": "media",
                        "media": {"payload": b64}
                    }))
                    logger.debug(f"[Twilio] enviado audio ({len(mulaw)} bytes)")
        except Exception as e:
            logger.error(f"[Bridge] Error OpenAI→Twilio: {e}")
            stop_signal.set()

    async def commit_and_respond_loop():
        """Cada 1s, si no hay audio nuevo desde hace >1s, commit+response.create"""
        while not stop_signal.is_set():
            await asyncio.sleep(1)
            now = asyncio.get_event_loop().time()
            if now - last_audio_ts > 1.0:
                logger.info("[Bridge] Commit por inactividad")
                try:
                    await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                    logger.info("[OpenAI] Enviado input_audio_buffer.commit")
                    await openai_ws.send(json.dumps({"type": "response.create"}))
                    logger.info("[OpenAI] Enviado response.create")
                except Exception as e:
                    logger.error(f"[Bridge] Error al commitear+responder: {e}")

    # Ejecutamos los 3 bucles concurrentemente
    try:
        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio(),
            commit_and_respond_loop()
        )
    finally:
        await openai_ws.close()
        logger.info("[OpenAI] WebSocket cerrado")
        await ws.close()
        logger.info("[Twilio] Conexión cerrada")

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
