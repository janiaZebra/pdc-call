


import asyncio
import base64
import json
import websockets
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from jania import env

app = FastAPI()

OPENAI_API_KEY = env("OPENAI_API_KEY")
TWILIO_STREAM_URL = env("TWILIO_STREAM_URL", "wss://your-twilio-stream-url")
TWILIO_SAY_MESSAGE = env("TWILIO_SAY_MESSAGE")

def ulaw8k_to_pcm16k(ulaw_data):
    audio = AudioSegment(
        data=ulaw_data,
        sample_width=1,
        frame_rate=8000,
        channels=1,
        codec="ulaw"
    )
    audio = audio.set_frame_rate(16000).set_sample_width(2)
    return audio.raw_data

def pcm16k_to_ulaw8k(pcm_data):
    audio = AudioSegment(
        data=pcm_data,
        sample_width=2,
        frame_rate=16000,
        channels=1
    )
    audio = audio.set_frame_rate(8000).set_sample_width(1)
    return audio.raw_data

@app.post("/twilio/voice")
async def twilio_voice():
    resp = ET.Element("Response")
    start = ET.SubElement(resp, "Start")
    ET.SubElement(start, "Stream", url=TWILIO_STREAM_URL)
    say = ET.SubElement(resp, "Say")
    say.text = TWILIO_SAY_MESSAGE
    xml_str = ET.tostring(resp, encoding="unicode")
    return Response(content=xml_str, media_type="text/xml")

@app.websocket("/stream")
async def stream_bridge(websocket: WebSocket):
    await websocket.accept()
    print("Conexión Media Streams entrante")

    async with websockets.connect(
        "wss://api.openai.com/v1/audio/ws",
        extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    ) as openai_ws:

        # ENVÍA HANDSHAKE (mensaje de inicio, obligatorio)
        start_msg = {
            "type": "start",
            "model": "gpt-4o",
            "voice": "nova",
            "sample_rate": 16000,
            "response_format": "audio",
            "input_format": "pcm"
        }
        await openai_ws.send(json.dumps(start_msg))

        async def forward_twilio_to_openai():
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("event") == "media":
                    audio_bytes = base64.b64decode(data["media"]["payload"])
                    pcm16k = ulaw8k_to_pcm16k(audio_bytes)
                    await openai_ws.send(pcm16k)
                elif data.get("event") == "stop":
                    await openai_ws.send(json.dumps({"type": "stop"}))
                    break

        async def forward_openai_to_twilio():
            while True:
                reply = await openai_ws.recv()
                if isinstance(reply, bytes):
                    ulaw8k = pcm16k_to_ulaw8k(reply)
                    payload_b64 = base64.b64encode(ulaw8k).decode()
                    event = {
                        "event": "media",
                        "media": {"payload": payload_b64}
                    }
                    await websocket.send_text(json.dumps(event))
                else:
                    # Es un mensaje JSON (evento, fin de conversación, etc)
                    try:
                        event = json.loads(reply)
                        print(f"Evento OpenAI: {event}")
                        if event.get("type") == "stop":
                            break
                    except Exception:
                        pass

        await asyncio.gather(forward_twilio_to_openai(), forward_openai_to_twilio())

