import asyncio
import os
import json
import base64
import numpy as np
import websockets
import logging
from aiohttp import web
from jania import env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = env("OPENAI_API_KEY")
MODEL = env("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17")

ULAW_TO_PCM16 = np.array([
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0
], dtype=np.int16)


PCM_TO_ULAW_TABLE = np.zeros(65536, dtype=np.uint8)
for i in range(65536):
    pcm = i - 32768
    sign = 0
    if pcm < 0:
        sign = 0x80
        pcm = -pcm

    pcm = pcm + 0x84
    if pcm > 0x7FFF:
        pcm = 0x7FFF

    segment = 0
    for j in range(8):
        if pcm & (0x4000 >> j):
            segment = j
            break

    if segment >= 8:
        ulaw = 0x7F ^ sign
    else:
        shift = segment + 3
        ulaw = ((0x0F & (pcm >> shift)) | (segment << 4)) ^ sign ^ 0x7F

    PCM_TO_ULAW_TABLE[i] = ulaw


def ulaw_to_pcm16_fast(ulaw_bytes):
    """upsampling"""
    pcm_8khz = ULAW_TO_PCM16[np.frombuffer(ulaw_bytes, dtype=np.uint8)]
    pcm_16khz = np.repeat(pcm_8khz, 2)
    return pcm_16khz.tobytes()


def pcm16_to_ulaw_fast(pcm16_bytes):
    """Ultra-fast PCM16 to μ-law conversion with downsampling"""
    pcm_16khz = np.frombuffer(pcm16_bytes, dtype=np.int16)
    pcm_8khz = pcm_16khz[::2]
    pcm_unsigned = (pcm_8khz.astype(np.int32) + 32768).clip(0, 65535)
    ulaw = PCM_TO_ULAW_TABLE[pcm_unsigned]
    return ulaw.tobytes()


async def handle_websocket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logger.info("Twilio WebSocket connected")

    to_openai_queue = asyncio.Queue(maxsize=100)
    to_twilio_queue = asyncio.Queue(maxsize=100)

    openai_uri = f"wss://api.openai.com/v1/realtime?model={MODEL}"
    openai_headers = {
        "Authorization": f"Bearer {API_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async def openai_relay():
        try:
            async with websockets.connect(openai_uri, extra_headers=openai_headers) as openai_ws:
                logger.info("Connected to OpenAI")

                await openai_ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "instructions": "You are a helpful assistant.",
                        "voice": "alloy",
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 600
                        }
                    }
                }))

                async def forward_to_openai():
                    buffer = bytearray()
                    while True:
                        chunk = await to_openai_queue.get()
                        if chunk is None:
                            break

                        buffer.extend(chunk)
                        while len(buffer) >= 320:
                            audio_chunk = buffer[:320]
                            buffer = buffer[320:]

                            pcm16 = ulaw_to_pcm16_fast(audio_chunk)
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm16).decode()
                            }))

                async def forward_from_openai():
                    async for message in openai_ws:
                        try:
                            data = json.loads(message)

                            if data["type"] == "response.audio.delta":
                                pcm16_data = base64.b64decode(data["delta"])
                                ulaw_data = pcm16_to_ulaw_fast(pcm16_data)
                                for i in range(0, len(ulaw_data), 160):
                                    chunk = ulaw_data[i:i + 160]
                                    if len(chunk) < 160:
                                        chunk = chunk + b'\xFF' * (160 - len(chunk))
                                    await to_twilio_queue.put(chunk)

                            elif data["type"] == "error":
                                logger.error(f"OpenAI error: {data}")
                        except Exception as e:
                            logger.error(f"Error processing OpenAI message: {e}")

                await asyncio.gather(
                    forward_to_openai(),
                    forward_from_openai()
                )

        except Exception as e:
            logger.error(f"OpenAI connection error: {e}")
        finally:
            await to_openai_queue.put(None)
            await to_twilio_queue.put(None)

    openai_task = asyncio.create_task(openai_relay())

    async def twilio_sender():
        try:
            while True:
                chunk = await to_twilio_queue.get()
                if chunk is None:
                    break

                await ws.send_str(json.dumps({
                    "event": "media",
                    "media": {
                        "track": "outbound",
                        "payload": base64.b64encode(chunk).decode()
                    }
                }))
        except Exception as e:
            logger.error(f"Twilio sender error: {e}")

    sender_task = asyncio.create_task(twilio_sender())

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)

                if data.get("event") == "media" and data["media"]["track"] == "inbound":
                    audio_bytes = base64.b64decode(data["media"]["payload"])
                    try:
                        to_openai_queue.put_nowait(audio_bytes)
                    except asyncio.QueueFull:
                        logger.warning("OpenAI queue full, dropping audio")

                elif data.get("event") == "stop":
                    break

            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await to_openai_queue.put(None)
        await to_twilio_queue.put(None)
        openai_task.cancel()
        sender_task.cancel()
        await ws.close()
        logger.info("Twilio WebSocket closed")

    return ws

app = web.Application()
app.router.add_post('/audio-stream', handle_websocket)
app.router.add_get('/audio-stream', handle_websocket)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    web.run_app(app, host='0.0.0.0', port=port)