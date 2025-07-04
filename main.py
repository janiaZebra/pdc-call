import asyncio
import base64
import json
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from jania import env

app = FastAPI()

# ------------ PARÁMETROS MODIFICABLES -------------
OPENAI_API_KEY = env("OPENAI_API_KEY")
TWILIO_STREAM_URL = env("TWILIO_STREAM_URL", "wss://your-twilio-stream-url")
TWILIO_SAY_MESSAGE = env("TWILIO_SAY_MESSAGE", "Estás hablando con la inteligencia artificial.")
OPENAI_MODEL = "gpt-4o"
OPENAI_VOICE = "nova"
# ------------ FIN PARÁMETROS MODIFICABLES ----------

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

        # HANDSHAKE OpenAI
        start_msg = {
            "type": "start",
            "model": OPENAI_MODEL,
            "voice": OPENAI_VOICE,
            "sample_rate": 16000,
            "response_format": "audio",
            "input_format": "pcm"
        }
        await openai_ws.send(json.dumps(start_msg))

        async def forward_twilio_to_openai():
            try:
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
            except WebSocketDisconnect:
                pass

        async def forward_openai_to_twilio():
            try:
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
                        # Mensaje JSON (evento OpenAI)
                        try:
                            event = json.loads(reply)
                            print(f"Evento OpenAI: {event}")
                            if event.get("type") == "stop":
                                break
                        except Exception:
                            pass
            except WebSocketDisconnect:
                pass

        await asyncio.gather(forward_twilio_to_openai(), forward_openai_to_twilio())

# ------------- ENDPOINT PARA TEST WEB --------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Test de Voz con OpenAI</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        button { font-size: 1.2em; margin: 0.5em; }
      </style>
    </head>
    <body>
      <h1>Test Audio Realtime OpenAI</h1>
      <button id="start">🎙️ Hablar</button>
      <button id="stop" disabled>⏹️ Parar</button>
      <p id="status"></p>
      <audio id="player" controls></audio>
      <script>
        let ws, mediaRecorder, audioChunks = [], audioContext, source, player = document.getElementById('player');
        const status = document.getElementById('status');
        document.getElementById('start').onclick = async () => {
          ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/client-ws');
          ws.binaryType = "arraybuffer";
          ws.onopen = async () => {
            status.innerText = "Grabando...";
            document.getElementById('start').disabled = true;
            document.getElementById('stop').disabled = false;
            const stream = await navigator.mediaDevices.getUserMedia({audio:true});
            mediaRecorder = new MediaRecorder(stream, {mimeType: 'audio/webm'});
            mediaRecorder.ondataavailable = e => {
              if (ws.readyState === 1) ws.send(e.data);
            };
            mediaRecorder.start(200);
          };
          ws.onmessage = (msg) => {
            if (typeof msg.data === 'object') {
              msg.data.arrayBuffer().then(buf => {
                const blob = new Blob([buf], {type: "audio/wav"});
                player.src = URL.createObjectURL(blob);
                player.play();
              });
            }
          };
        };
        document.getElementById('stop').onclick = () => {
          if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
          if (ws && ws.readyState === 1) ws.close();
          status.innerText = "Listo";
          document.getElementById('start').disabled = false;
          document.getElementById('stop').disabled = true;
        };
      </script>
    </body>
    </html>
    """

@app.websocket("/client-ws")
async def ws_client_bridge(websocket: WebSocket):
    await websocket.accept()
    # Conexión OpenAI Voice
    async with websockets.connect(
        "wss://api.openai.com/v1/audio/ws",
        extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
    ) as openai_ws:
        # HANDSHAKE
        start_msg = {
            "type": "start",
            "model": OPENAI_MODEL,
            "voice": OPENAI_VOICE,
            "sample_rate": 16000,
            "response_format": "audio",
            "input_format": "webm"
        }
        await openai_ws.send(json.dumps(start_msg))

        async def forward_client_to_openai():
            try:
                while True:
                    chunk = await websocket.receive_bytes()
                    await openai_ws.send(chunk)
            except WebSocketDisconnect:
                await openai_ws.send(json.dumps({"type": "stop"}))

        async def forward_openai_to_client():
            try:
                while True:
                    reply = await openai_ws.recv()
                    if isinstance(reply, bytes):
                        await websocket.send_bytes(reply)
                    else:
                        try:
                            event = json.loads(reply)
                            print(f"Evento OpenAI Cliente: {event}")
                            if event.get("type") == "stop":
                                break
                        except Exception:
                            pass
            except WebSocketDisconnect:
                pass

        await asyncio.gather(forward_client_to_openai(), forward_openai_to_client())
