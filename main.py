from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import base64
import asyncio
import websockets
import logging
from jania import env

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
OPENAI_VOICE = env("OPENAI_VOICE", "alloy")
MODEL = "gpt-4o-realtime-preview"
OPENAI_WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    host = request.headers.get("host", "localhost")
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <Stream url="wss://{host}/ws/audio" />
        </Connect>
        <Say language="es-ES">Conectando con el asistente virtual...</Say>
        <Pause length="60"/>
    </Response>
    """
    return Response(content=twiml.strip(), media_type="application/xml")


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    stream_sid = None
    openai_ws = None
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        openai_ws = await websockets.connect(OPENAI_WS_URL, additional_headers=headers )
        logger.info("Connected to OpenAI WebSocket")

        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "Eres un asistente telefónico amable y servicial. Responde en español de forma clara y concisa.",
                "voice": OPENAI_VOICE,
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "temperature": 0.8,
                "max_response_output_tokens": 4096
            }
        }

        await openai_ws.send(json.dumps(session_config))
        logger.info("OpenAI session configured")

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
                        logger.info(f"Stream started: {stream_sid}")
                    elif message["event"] == "media":
                        payload = message["media"]["payload"]
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": payload
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif message["event"] == "stop":
                        logger.info("Stream stopped")
                        break

            except WebSocketDisconnect:
                logger.info("Twilio WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error in Twilio handler: {e}")

        async def handle_openai_messages():
            try:
                async for message in openai_ws:
                    response = json.loads(message)
                    if response["type"] == "response.audio.delta":
                        audio_delta = response.get("delta", "")
                        if audio_delta:
                            media_message = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_delta
                                }
                            }
                            await websocket.send_text(json.dumps(media_message))
                    elif response["type"] == "response.audio_transcript.done":
                        transcript = response.get("transcript", "")
                        logger.info(f"Assistant: {transcript}")
                    elif response["type"] == "conversation.item.input_audio_transcription.completed":
                        transcript = response.get("transcript", "")
                        logger.info(f"User: {transcript}")
                    elif response["type"] == "error":
                        logger.error(f"OpenAI error: {response}")

            except Exception as e:
                logger.error(f"Error in OpenAI handler: {e}")

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
        logger.info("WebSocket connections closed")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "twilio-openai-bridge"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
