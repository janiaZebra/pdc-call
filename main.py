import os, json, base64, asyncio, audioop
import numpy as np
from scipy.signal import resample
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import websockets

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    uri = f"wss://api.openai.com/v1/realtime?model={MODEL}"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with websockets.connect(uri, extra_headers=headers) as openai_ws:
        await openai_ws.send(json.dumps({"type": "start", "config": {"vad_config": {"silence_duration_ms": 800}}}))

        async def recv_twilio_send_openai():
            try:
                while True:
                    msg = await websocket.receive_text()
                    payload = json.loads(msg).get("media", {}).get("payload", "")
                    if not payload: continue
                    mulaw = base64.b64decode(payload)
                    pcm16 = audioop.ulaw2lin(mulaw, 2)
                    pcm_np = np.frombuffer(pcm16, dtype=np.int16)
                    resampled = resample(pcm_np, int(len(pcm_np) * 16000 / 8000)).astype(np.int16).tobytes()
                    b64 = base64.b64encode(resampled).decode()
                    await openai_ws.send(json.dumps({"type": "input_audio_buffer.append", "data": b64}))
            except: pass
            await openai_ws.send(json.dumps({"type": "input_audio_buffer.flush"}))

        async def recv_openai_send_twilio():
            try:
                async for msg in openai_ws:
                    data = json.loads(msg)
                    if data.get("type") != "audio": continue
                    pcm16 = base64.b64decode(data["audio"])
                    mulaw = audioop.lin2ulaw(pcm16, 2)
                    b64 = base64.b64encode(mulaw).decode()
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "streamSid": "stream",
                        "media": {"track": "outbound", "payload": b64}
                    }))
            except: pass

        await asyncio.gather(recv_twilio_send_openai(), recv_openai_send_twilio())
