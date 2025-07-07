import asyncio
import websockets
import json
import base64
import logging
import os
import ssl
import struct
import audioop
from typing import Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="OpenAI Realtime API Bridge")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionConfig(BaseModel):
    api_key: str
    instructions: str = "You are a helpful AI assistant"
    voice: str = "alloy"
    temperature: float = 0.6
    vad_enabled: bool = True
    vad_threshold: float = 0.5


class RealtimeClient:
    """Client to handle OpenAI Realtime API WebSocket connection"""

    def __init__(self, api_key: str, instructions: str = "You are a helpful AI assistant",
                 voice: str = "alloy", vad_enabled: bool = True):
        self.url = "wss://api.openai.com/v1/realtime"
        self.model = "gpt-4o-realtime-preview-2024-12-17"  # Updated model
        self.api_key = api_key
        self.ws = None
        self.client_ws = None  # WebSocket connection to the frontend

        # SSL Configuration
        self.ssl_context = ssl.create_default_context()

        # Session configuration
        self.instructions = instructions
        self.voice = voice
        self.vad_enabled = vad_enabled

        # VAD configuration
        self.vad_config = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500  # Increased for phone calls
        }

        self.session_config = {
            "modalities": ["audio", "text"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": self.vad_config if self.vad_enabled else None,
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "temperature": 0.6,
            "max_response_output_tokens": 4096
        }

        self.running = False

    async def connect(self):
        """Connect to OpenAI Realtime API"""
        try:
            logger.info(f"Connecting to OpenAI Realtime API with model {self.model}...")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }

            # Use the updated URL format
            url = f"{self.url}?model={self.model}"

            self.ws = await websockets.connect(
                url,
                extra_headers=headers,
                ssl=self.ssl_context,
                ping_interval=20,
                ping_timeout=10
            )

            logger.info("Successfully connected to OpenAI Realtime API")

            # Configure session
            await self.send_event({
                "type": "session.update",
                "session": self.session_config
            })

            logger.info("Session configured")
            self.running = True

            return True

        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect from OpenAI Realtime API"""
        self.running = False
        if self.ws:
            await self.ws.close()
            logger.info("Disconnected from OpenAI Realtime API")

    async def send_event(self, event: Dict[str, Any]):
        """Send event to OpenAI"""
        if self.ws:
            await self.ws.send(json.dumps(event))
            logger.debug(f"Sent event: {event['type']}")

    async def send_audio(self, audio_data: bytes):
        """Send audio data to OpenAI"""
        base64_audio = base64.b64encode(audio_data).decode('utf-8')
        await self.send_event({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        })

    async def send_text(self, text: str):
        """Send text message to OpenAI"""
        await self.send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": text
                }]
            }
        })

        # Trigger response generation
        await self.send_event({
            "type": "response.create"
        })

    async def handle_openai_events(self):
        """Handle events from OpenAI"""
        try:
            async for message in self.ws:
                if not self.running:
                    break

                event = json.loads(message)
                event_type = event.get("type")

                logger.debug(f"Received event: {event_type}")

                # Forward event to frontend
                if self.client_ws:
                    await self.client_ws.send_text(json.dumps({
                        "type": "openai_event",
                        "event": event
                    }))

                # Handle specific events
                if event_type == "session.created":
                    logger.info("Session created successfully")

                elif event_type == "response.audio.delta":
                    # Audio chunk received
                    pass

                elif event_type == "response.audio_transcript.delta":
                    # Transcript chunk received
                    pass

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # User speech transcription completed
                    pass

                elif event_type == "error":
                    logger.error(f"OpenAI Error: {event.get('error', {})}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("OpenAI connection closed")
        except Exception as e:
            logger.error(f"Error handling OpenAI events: {str(e)}")
        finally:
            await self.disconnect()


# Store active connections
active_connections: Dict[str, RealtimeClient] = {}


# Twilio Media Stream specific client
class TwilioRealtimeClient(RealtimeClient):
    """Extended client for Twilio Media Streams"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_sid = None
        self.call_sid = None
        self.from_number = None
        self.to_number = None
        self.audio_buffer = bytearray()  # Buffer for accumulating audio
        self.chunk_size = 640  # Size for 20ms of 8kHz mulaw audio

    def convert_mulaw_to_pcm(self, mulaw_data: bytes) -> bytes:
        """Convert 8kHz mulaw to 24kHz PCM16"""
        try:
            # First, convert mulaw to PCM16 at 8kHz
            pcm_8khz = audioop.ulaw2lin(mulaw_data, 2)

            # Resample from 8kHz to 24kHz (3x upsampling)
            pcm_24khz = audioop.ratecv(pcm_8khz, 2, 1, 8000, 24000, None)[0]

            return pcm_24khz
        except Exception as e:
            logger.error(f"Error converting mulaw to PCM: {e}")
            return b''

    def convert_pcm_to_mulaw(self, pcm_data: bytes) -> bytes:
        """Convert 24kHz PCM16 to 8kHz mulaw"""
        try:
            # Resample from 24kHz to 8kHz
            pcm_8khz = audioop.ratecv(pcm_data, 2, 1, 24000, 8000, None)[0]

            # Convert to mulaw
            mulaw_data = audioop.lin2ulaw(pcm_8khz, 2)

            return mulaw_data
        except Exception as e:
            logger.error(f"Error converting PCM to mulaw: {e}")
            return b''

    async def send_audio_to_twilio(self, audio_data: bytes):
        """Send audio back to Twilio"""
        if self.client_ws and self.stream_sid:
            # Convert PCM to mulaw
            mulaw_audio = self.convert_pcm_to_mulaw(audio_data)

            # Send in chunks to avoid overwhelming
            chunk_size = 160  # 20ms of 8kHz audio
            for i in range(0, len(mulaw_audio), chunk_size):
                chunk = mulaw_audio[i:i + chunk_size]
                # Encode to base64
                audio_base64 = base64.b64encode(chunk).decode('utf-8')

                # Send to Twilio
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }

                await self.client_ws.send_text(json.dumps(message))

    async def process_audio_buffer(self):
        """Process accumulated audio buffer"""
        # Process in chunks to avoid overwhelming OpenAI
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = bytes(self.audio_buffer[:self.chunk_size])
            self.audio_buffer = self.audio_buffer[self.chunk_size:]

            # Convert and send to OpenAI
            pcm_audio = self.convert_mulaw_to_pcm(chunk)
            if pcm_audio:
                await self.send_audio(pcm_audio)

    async def handle_openai_events_for_twilio(self):
        """Handle OpenAI events specifically for Twilio"""
        try:
            async for message in self.ws:
                if not self.running:
                    break

                event = json.loads(message)
                event_type = event.get("type")

                logger.debug(f"Twilio - Received OpenAI event: {event_type}")

                # Handle audio responses
                if event_type == "response.audio.delta":
                    if event.get("delta"):
                        # Decode base64 audio from OpenAI
                        audio_data = base64.b64decode(event["delta"])

                        # Send to Twilio
                        await self.send_audio_to_twilio(audio_data)

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # Log transcriptions for debugging
                    if event.get("transcript"):
                        logger.info(f"User said: {event['transcript']}")

                elif event_type == "response.audio_transcript.delta":
                    # Log assistant responses
                    if event.get("delta"):
                        logger.info(f"Assistant saying: {event['delta']}")

                elif event_type == "response.done":
                    logger.debug("Response completed")

                elif event_type == "error":
                    logger.error(f"OpenAI Error: {event.get('error', {})}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("OpenAI connection closed for Twilio call")
        except Exception as e:
            logger.error(f"Error handling OpenAI events for Twilio: {str(e)}")
        finally:
            await self.disconnect()


@app.get("/")
async def root():
    """Serve the main HTML file"""
    return FileResponse("templates/index.html")


@app.post("/voice")
async def handle_incoming_call(request: Request):
    """Webhook for incoming Twilio calls"""
    # Get the base URL from the request
    host = request.headers.get("host", "localhost:8000")
    protocol = "wss" if request.url.scheme == "https" else "ws"

    # TwiML response to connect the call to our WebSocket
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{protocol}://{host}/twilio-stream" />
    </Start>
    <Say voice="alice">Connecting to your AI assistant. One moment please.</Say>
    <Pause length="60" />
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/twilio-stream")
async def twilio_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Twilio Media Streams"""
    await websocket.accept()
    connection_id = str(id(websocket))
    client = None

    try:
        # Get API key from environment or use default
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("No OpenAI API key found in environment")
            await websocket.send_json({
                "event": "error",
                "message": "No API key configured"
            })
            return

        # Create Twilio-specific client
        client = TwilioRealtimeClient(
            api_key=api_key,
            instructions="You are a helpful voice assistant on a phone call. Be concise, friendly, and natural in your responses. Speak conversationally.",
            voice="alloy",
            vad_enabled=True
        )

        client.client_ws = websocket
        active_connections[connection_id] = client

        # Start OpenAI event handler
        openai_task = None
        connected_to_openai = False

        # Handle Twilio Media Stream events
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                event_type = message.get("event")

                if event_type == "connected":
                    logger.info("Twilio Media Stream connected")
                    # Send clear response
                    await websocket.send_text(json.dumps({
                        "event": "connected",
                        "protocol": "Call",
                        "version": "1.0.0"
                    }))

                elif event_type == "start":
                    # Extract call information
                    start_data = message.get("start", {})
                    client.stream_sid = start_data.get("streamSid")
                    client.call_sid = start_data.get("callSid")

                    custom_params = start_data.get("customParameters", {})
                    client.from_number = custom_params.get("from")
                    client.to_number = custom_params.get("to")

                    logger.info(f"Call started - From: {client.from_number}, Stream: {client.stream_sid}")

                    # Connect to OpenAI only after receiving start event
                    if not connected_to_openai:
                        connected = await client.connect()
                        if connected:
                            connected_to_openai = True
                            # Start handling OpenAI events
                            openai_task = asyncio.create_task(client.handle_openai_events_for_twilio())

                            # Send initial greeting after a short delay
                            await asyncio.sleep(0.5)
                            await client.send_text("Hello! I'm your AI assistant. How can I help you today?")
                        else:
                            logger.error("Failed to connect to OpenAI")
                            break

                elif event_type == "media":
                    # Audio data from Twilio
                    if connected_to_openai:
                        media = message.get("media", {})
                        if media.get("payload"):
                            # Decode mulaw audio
                            mulaw_audio = base64.b64decode(media["payload"])

                            # Add to buffer
                            client.audio_buffer.extend(mulaw_audio)

                            # Process buffer
                            await client.process_audio_buffer()

                elif event_type == "stop":
                    logger.info("Twilio Media Stream stopped")
                    break

            except WebSocketDisconnect:
                logger.info("Twilio WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in Twilio WebSocket: {str(e)}")
                break

    except Exception as e:
        logger.error(f"Twilio WebSocket error: {str(e)}")
    finally:
        # Cleanup
        if connection_id in active_connections:
            if client:
                await client.disconnect()
            del active_connections[connection_id]

        if openai_task:
            openai_task.cancel()
            try:
                await openai_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Twilio call {connection_id} ended")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frontend communication"""
    await websocket.accept()
    connection_id = str(id(websocket))
    client = None

    try:
        # Wait for initial configuration
        config_data = await websocket.receive_json()

        if config_data.get("type") != "config":
            await websocket.send_json({
                "type": "error",
                "message": "First message must be configuration"
            })
            return

        # Create and connect Realtime client
        config = SessionConfig(**config_data.get("config", {}))
        client = RealtimeClient(
            api_key=config.api_key,
            instructions=config.instructions,
            voice=config.voice,
            vad_enabled=config.vad_enabled
        )

        client.client_ws = websocket
        active_connections[connection_id] = client

        # Connect to OpenAI
        connected = await client.connect()
        if not connected:
            await websocket.send_json({
                "type": "error",
                "message": "Failed to connect to OpenAI Realtime API"
            })
            return

        await websocket.send_json({
            "type": "connected",
            "message": "Successfully connected to OpenAI Realtime API"
        })

        # Start handling OpenAI events
        openai_task = asyncio.create_task(client.handle_openai_events())

        # Handle messages from frontend
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "audio":
                    # Decode base64 audio and send to OpenAI
                    audio_data = base64.b64decode(data.get("audio", ""))
                    await client.send_audio(audio_data)

                elif msg_type == "text":
                    # Send text message to OpenAI
                    text = data.get("text", "")
                    await client.send_text(text)

                elif msg_type == "audio_end":
                    # Signal end of audio input (for manual VAD)
                    await client.send_event({
                        "type": "input_audio_buffer.commit"
                    })

                elif msg_type == "interrupt":
                    # Interrupt current response
                    await client.send_event({
                        "type": "response.cancel"
                    })

                elif msg_type == "update_session":
                    # Update session configuration
                    new_config = data.get("session", {})
                    await client.send_event({
                        "type": "session.update",
                        "session": new_config
                    })

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling frontend message: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket.client_state.value == 1:  # If still connected
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
    finally:
        # Cleanup
        if connection_id in active_connections:
            if client:
                await client.disconnect()
            del active_connections[connection_id]

        logger.info(f"Client {connection_id} disconnected")


@app.post("/test_connection")
async def test_connection(config: SessionConfig):
    """Test OpenAI API connection"""
    try:
        client = RealtimeClient(api_key=config.api_key)
        connected = await client.connect()

        if connected:
            await client.disconnect()
            return {"status": "success", "message": "Connection successful"}
        else:
            raise HTTPException(status_code=400, detail="Failed to connect to OpenAI")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": len(active_connections)
    }


if __name__ == "__main__":
    # Create index.html if it doesn't exist
    if not os.path.exists("templates/index.html"):
        logger.warning("index.html not found. Please create it.")

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )