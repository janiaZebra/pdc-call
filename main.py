import asyncio
import base64
import json
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, HTMLResponse
from pydub import AudioSegment
import xml.etree.ElementTree as ET
from jania import env
import io

app = FastAPI()

# ------------ PARÁMETROS MODIFICABLES -------------
OPENAI_API_KEY = env("OPENAI_API_KEY")
TWILIO_STREAM_URL = env("TWILIO_STREAM_URL", "wss://your-twilio-stream-url")
TWILIO_SAY_MESSAGE = env("TWILIO_SAY_MESSAGE", "Estás hablando con la inteligencia artificial.")
OPENAI_MODEL = "gpt-4o-realtime-preview-2024-10-01"  # Modelo correcto para Realtime API
OPENAI_VOICE = "alloy"  # Voces disponibles: alloy, echo, shimmer


# ------------ FIN PARÁMETROS MODIFICABLES ----------

def ulaw8k_to_pcm16k(ulaw_data):
    """Convierte audio µ-law 8kHz a PCM 16kHz"""
    audio = AudioSegment(
        data=ulaw_data,
        sample_width=1,
        frame_rate=8000,
        channels=1
    )
    # Importante: el formato es mulaw, no ulaw
    audio = audio.set_frame_rate(16000).set_sample_width(2)
    return audio.raw_data


def pcm16k_to_ulaw8k(pcm_data):
    """Convierte audio PCM 16kHz a µ-law 8kHz"""
    audio = AudioSegment(
        data=pcm_data,
        sample_width=2,
        frame_rate=16000,
        channels=1
    )
    audio = audio.set_frame_rate(8000).set_sample_width(1)
    # Exportar como mulaw
    buffer = io.BytesIO()
    audio.export(buffer, format="mulaw", codec="pcm_mulaw")
    return buffer.getvalue()


@app.post("/twilio/voice")
async def twilio_voice():
    """Endpoint que Twilio llama cuando recibe una llamada"""
    resp = ET.Element("Response")
    start = ET.SubElement(resp, "Start")
    ET.SubElement(start, "Stream", url=TWILIO_STREAM_URL)
    say = ET.SubElement(resp, "Say")
    say.text = TWILIO_SAY_MESSAGE
    # Pause para mantener la llamada abierta
    ET.SubElement(resp, "Pause", length="3600")

    xml_str = ET.tostring(resp, encoding="unicode")
    return Response(content=xml_str, media_type="text/xml")


@app.websocket("/stream")
async def stream_bridge(websocket: WebSocket):
    """WebSocket que conecta Twilio Media Streams con OpenAI Realtime API"""
    await websocket.accept()
    print("Conexión Media Streams entrante")

    stream_sid = None

    try:
        async with websockets.connect(
                "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
        ) as openai_ws:
            print("Conectado a OpenAI Realtime API")

            # Configurar la sesión
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": "Eres un asistente de voz amigable. Responde en español de forma natural y conversacional.",
                    "voice": OPENAI_VOICE,
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
                    }
                }
            }
            await openai_ws.send(json.dumps(session_config))

            async def forward_twilio_to_openai():
                """Reenvía audio de Twilio a OpenAI"""
                try:
                    while True:
                        msg = await websocket.receive_text()
                        data = json.loads(msg)

                        if data.get("event") == "start":
                            stream_sid = data["start"]["streamSid"]
                            print(f"Stream iniciado: {stream_sid}")

                        elif data.get("event") == "media":
                            # Decodificar audio µ-law de Twilio
                            audio_bytes = base64.b64decode(data["media"]["payload"])
                            # Convertir a PCM 16kHz
                            pcm_audio = ulaw8k_to_pcm16k(audio_bytes)
                            # Codificar en base64
                            pcm_base64 = base64.b64encode(pcm_audio).decode('utf-8')

                            # Enviar a OpenAI
                            audio_event = {
                                "type": "input_audio_buffer.append",
                                "audio": pcm_base64
                            }
                            await openai_ws.send(json.dumps(audio_event))

                        elif data.get("event") == "stop":
                            print("Stream detenido por Twilio")
                            break

                except WebSocketDisconnect:
                    print("Desconexión de Twilio")
                except Exception as e:
                    print(f"Error en forward_twilio_to_openai: {e}")

            async def forward_openai_to_twilio():
                """Reenvía audio de OpenAI a Twilio"""
                try:
                    while True:
                        response = await openai_ws.recv()

                        if isinstance(response, str):
                            event = json.loads(response)
                            event_type = event.get("type")

                            if event_type == "session.created":
                                print("Sesión OpenAI creada")

                            elif event_type == "response.audio.delta":
                                # Audio de respuesta de OpenAI
                                if "delta" in event:
                                    audio_base64 = event["delta"]
                                    # Decodificar de base64
                                    pcm_audio = base64.b64decode(audio_base64)
                                    # Convertir a µ-law 8kHz para Twilio
                                    ulaw_audio = pcm16k_to_ulaw8k(pcm_audio)
                                    # Codificar en base64
                                    ulaw_base64 = base64.b64encode(ulaw_audio).decode('utf-8')

                                    # Enviar a Twilio
                                    media_event = {
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {
                                            "payload": ulaw_base64
                                        }
                                    }
                                    await websocket.send_text(json.dumps(media_event))

                            elif event_type == "response.audio_transcript.done":
                                print(f"Transcripción respuesta: {event.get('transcript', '')}")

                            elif event_type == "input_audio_buffer.speech_started":
                                print("Habla detectada")

                            elif event_type == "input_audio_buffer.speech_stopped":
                                print("Habla detenida")
                                # Commit del buffer de audio cuando se detecta fin del habla
                                commit_event = {
                                    "type": "input_audio_buffer.commit"
                                }
                                await openai_ws.send(json.dumps(commit_event))

                            elif event_type == "conversation.item.input_audio_transcription.completed":
                                print(f"Transcripción entrada: {event.get('transcript', '')}")

                            elif event_type == "error":
                                print(f"Error de OpenAI: {event}")

                except WebSocketDisconnect:
                    print("Desconexión de OpenAI")
                except Exception as e:
                    print(f"Error en forward_openai_to_twilio: {e}")

            # Ejecutar ambas tareas concurrentemente
            await asyncio.gather(
                forward_twilio_to_openai(),
                forward_openai_to_twilio()
            )

    except Exception as e:
        print(f"Error en stream_bridge: {e}")
    finally:
        await websocket.close()


# ------------- ENDPOINT PARA TEST WEB --------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Test de Voz con OpenAI Realtime</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        button { font-size: 1.2em; margin: 0.5em; padding: 0.5em 1em; }
        #status { margin: 1em 0; font-weight: bold; }
        .controls { margin: 2em 0; }
      </style>
    </head>
    <body>
      <h1>Test Audio Realtime OpenAI</h1>
      <div class="controls">
        <button id="start">🎙️ Iniciar Conversación</button>
        <button id="stop" disabled>⏹️ Detener</button>
      </div>
      <p id="status">Listo para comenzar</p>
      <div id="transcripts"></div>

      <script>
        let ws, mediaRecorder, audioContext, processor;
        const status = document.getElementById('status');
        const transcripts = document.getElementById('transcripts');

        document.getElementById('start').onclick = async () => {
          try {
            // Conectar WebSocket
            const protocol = location.protocol === 'https:' ? 'wss://' : 'ws://';
            ws = new WebSocket(protocol + location.host + '/client-ws');

            ws.onopen = async () => {
              status.innerText = "Conectado - Habla ahora...";
              document.getElementById('start').disabled = true;
              document.getElementById('stop').disabled = false;

              // Obtener acceso al micrófono
              const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

              // Crear contexto de audio para procesar
              audioContext = new AudioContext({ sampleRate: 16000 });
              const source = audioContext.createMediaStreamSource(stream);
              processor = audioContext.createScriptProcessor(4096, 1, 1);

              processor.onaudioprocess = (e) => {
                if (ws.readyState === WebSocket.OPEN) {
                  const inputData = e.inputBuffer.getChannelData(0);
                  // Convertir float32 a int16
                  const output = new Int16Array(inputData.length);
                  for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                  }
                  ws.send(output.buffer);
                }
              };

              source.connect(processor);
              processor.connect(audioContext.destination);
            };

            ws.onmessage = async (event) => {
              if (event.data instanceof Blob) {
                // Reproducir audio recibido
                const arrayBuffer = await event.data.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start();
              } else {
                // Manejar mensajes de texto (transcripciones, etc.)
                try {
                  const data = JSON.parse(event.data);
                  if (data.transcript) {
                    const p = document.createElement('p');
                    p.textContent = `${data.type}: ${data.transcript}`;
                    transcripts.appendChild(p);
                  }
                } catch (e) {
                  console.log('Mensaje recibido:', event.data);
                }
              }
            };

            ws.onerror = (error) => {
              console.error('WebSocket error:', error);
              status.innerText = "Error de conexión";
            };

            ws.onclose = () => {
              status.innerText = "Desconectado";
              cleanup();
            };

          } catch (error) {
            console.error('Error:', error);
            status.innerText = "Error: " + error.message;
          }
        };

        document.getElementById('stop').onclick = () => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.close();
          }
          cleanup();
        };

        function cleanup() {
          if (processor) {
            processor.disconnect();
            processor = null;
          }
          if (audioContext) {
            audioContext.close();
            audioContext = null;
          }
          document.getElementById('start').disabled = false;
          document.getElementById('stop').disabled = true;
        }
      </script>
    </body>
    </html>
    """


@app.websocket("/client-ws")
async def ws_client_bridge(websocket: WebSocket):
    """WebSocket para pruebas desde el navegador"""
    await websocket.accept()

    try:
        async with websockets.connect(
                "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
                extra_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta": "realtime=v1"
                }
        ) as openai_ws:

            # Configurar sesión
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": "Eres un asistente de voz amigable. Responde en español.",
                    "voice": OPENAI_VOICE,
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
                    }
                }
            }
            await openai_ws.send(json.dumps(session_config))

            async def forward_client_to_openai():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        # Convertir bytes a base64
                        audio_base64 = base64.b64encode(data).decode('utf-8')
                        # Enviar a OpenAI
                        audio_event = {
                            "type": "input_audio_buffer.append",
                            "audio": audio_base64
                        }
                        await openai_ws.send(json.dumps(audio_event))
                except WebSocketDisconnect:
                    print("Cliente desconectado")
                except Exception as e:
                    print(f"Error forwarding to OpenAI: {e}")

            async def forward_openai_to_client():
                try:
                    while True:
                        response = await openai_ws.recv()

                        if isinstance(response, str):
                            event = json.loads(response)
                            event_type = event.get("type")

                            if event_type == "response.audio.delta":
                                if "delta" in event:
                                    # Decodificar audio y enviar al cliente
                                    audio_data = base64.b64decode(event["delta"])
                                    await websocket.send_bytes(audio_data)

                            elif event_type in ["input_audio_buffer.speech_stopped"]:
                                # Commit audio buffer cuando se detecta fin del habla
                                commit_event = {"type": "input_audio_buffer.commit"}
                                await openai_ws.send(json.dumps(commit_event))

                            # Enviar transcripciones al cliente
                            elif event_type in ["conversation.item.input_audio_transcription.completed",
                                                "response.audio_transcript.done"]:
                                transcript_data = {
                                    "type": "user" if "input" in event_type else "assistant",
                                    "transcript": event.get("transcript", "")
                                }
                                await websocket.send_text(json.dumps(transcript_data))

                except WebSocketDisconnect:
                    print("OpenAI desconectado")
                except Exception as e:
                    print(f"Error forwarding from OpenAI: {e}")

            await asyncio.gather(
                forward_client_to_openai(),
                forward_openai_to_client()
            )

    except Exception as e:
        print(f"Error en client bridge: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)