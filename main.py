from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import base64
import asyncio
import audioop
import numpy as np
from scipy.signal import resample
import websockets
import logging
from typing import Optional

# Configure logging
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

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-realtime-preview"  # Updated model name
OPENAI_WS_URL = f"wss://api.openai.com/v1/realtime?model={MODEL}"

# Validate API key
if not API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set")


@app.post("/twilio/voice")
async def twilio_voice(request: Request):
    """Handle incoming Twilio voice call and return TwiML response"""
    # Get the host from the request for dynamic URL generation
    host = request.headers.get("host", "localhost")
    protocol = "wss" if request.url.scheme == "https" else "ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Start>
            <Stream url="wss://{host}/ws/audio" />
        </Start>
        <Say language="es-ES">Conectando con el asistente virtual...</Say>
        <Pause length="60"/>
    </Response>
    """
    return Response(content=twiml.strip(), media_type="application/xml")


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """WebSocket endpoint that bridges Twilio and OpenAI audio streams"""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    stream_sid = None
    openai_ws = None

    try:
        # Connect to OpenAI WebSocket
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        openai_ws = await websockets.connect(OPENAI_WS_URL, extra_headers=headers)
        logger.info("Connected to OpenAI WebSocket")

        # Configure OpenAI session
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": "Eres un asistente telefónico amable y servicial. Responde en español de forma clara y concisa.",
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
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
            """Receive audio from Twilio and forward to OpenAI"""
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
                        # Extract audio payload
                        payload = message["media"]["payload"]

                        # Convert from μ-law to PCM16
                        mulaw_audio = base64.b64decode(payload)
                        pcm16_audio = audioop.ulaw2lin(mulaw_audio, 2)

                        # Resample from 8kHz to 16kHz (Twilio uses 8kHz, OpenAI expects 16kHz)
                        pcm_array = np.frombuffer(pcm16_audio, dtype=np.int16)
                        resampled = resample(pcm_array, len(pcm_array) * 2)
                        resampled_pcm16 = resampled.astype(np.int16).tobytes()

                        # Send to OpenAI
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(resampled_pcm16).decode()
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
            """Receive responses from OpenAI and forward to Twilio"""
            try:
                async for message in openai_ws:
                    response = json.loads(message)

                    if response["type"] == "response.audio.delta":
                        # Extract audio delta
                        audio_delta = response.get("delta", "")
                        if audio_delta:
                            # Decode base64 audio
                            pcm16_audio = base64.b64decode(audio_delta)

                            # Resample from 16kHz to 8kHz for Twilio
                            pcm_array = np.frombuffer(pcm16_audio, dtype=np.int16)
                            resampled = resample(pcm_array, len(pcm_array) // 2)
                            resampled_pcm16 = resampled.astype(np.int16).tobytes()

                            # Convert to μ-law
                            mulaw_audio = audioop.lin2ulaw(resampled_pcm16, 2)

                            # Send to Twilio
                            media_message = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": base64.b64encode(mulaw_audio).decode()
                                }
                            }
                            await websocket.send_text(json.dumps(media_message))

                    elif response["type"] == "response.audio_transcript.done":
                        # Log assistant's transcript
                        transcript = response.get("transcript", "")
                        logger.info(f"Assistant: {transcript}")

                    elif response["type"] == "conversation.item.input_audio_transcription.completed":
                        # Log user's transcript
                        transcript = response.get("transcript", "")
                        logger.info(f"User: {transcript}")

                    elif response["type"] == "error":
                        logger.error(f"OpenAI error: {response}")

            except Exception as e:
                logger.error(f"Error in OpenAI handler: {e}")

        # Run both handlers concurrently
        await asyncio.gather(
            handle_twilio_messages(),
            handle_openai_messages()
        )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up connections
        if openai_ws:
            await openai_ws.close()
        await websocket.close()
        logger.info("WebSocket connections closed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "twilio-openai-bridge"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)