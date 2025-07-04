import asyncio
import websockets
import json
import base64
import os
import struct
import audioop
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Variables de entorno - Configuración
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-realtime-preview-2024-12-17")
OPENAI_VOICE = os.environ.get("OPENAI_VOICE", "alloy")
OPENAI_INSTRUCTIONS = os.environ.get("OPENAI_INSTRUCTIONS", "You are a helpful assistant.")
WEBSOCKET_HOST = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.environ.get("WEBSOCKET_PORT", "8080"))
TWILIO_SAMPLE_RATE = int(os.environ.get("TWILIO_SAMPLE_RATE", "8000"))
OPENAI_SAMPLE_RATE = int(os.environ.get("OPENAI_SAMPLE_RATE", "24000"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Configuración de logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwilioOpenAIBridge:
    def __init__(self):
        self.openai_ws: Optional[websockets.WebSocketClientProtocol] = None
        self.twilio_ws: Optional[websockets.WebSocketServerProtocol] = None
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

    async def connect_to_openai(self):
        """Conecta con OpenAI Realtime API"""
        try:
            url = f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }

            self.openai_ws = await websockets.connect(url, extra_headers=headers)
            logger.info("Conectado a OpenAI Realtime API")

            # Configurar la sesión
            await self.configure_openai_session()

        except Exception as e:
            logger.error(f"Error conectando a OpenAI: {e}")
            raise

    async def configure_openai_session(self):
        """Configura la sesión de OpenAI"""
        session_config = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": OPENAI_INSTRUCTIONS,
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

        await self.openai_ws.send(json.dumps(session_config))
        logger.info("Sesión de OpenAI configurada")

    def mulaw_to_pcm16(self, mulaw_data: bytes) -> bytes:
        """Convierte audio μ-law 8kHz a PCM16"""
        try:
            # Decodificar μ-law a PCM16
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)

            # Resamplear de 8kHz a 24kHz para OpenAI
            pcm_data = audioop.ratecv(
                pcm_data,
                2,  # bytes por muestra
                1,  # canales
                TWILIO_SAMPLE_RATE,
                OPENAI_SAMPLE_RATE,
                None
            )[0]

            return pcm_data
        except Exception as e:
            logger.error(f"Error convirtiendo μ-law a PCM16: {e}")
            return b''

    def pcm16_to_mulaw(self, pcm_data: bytes) -> bytes:
        """Convierte audio PCM16 24kHz a μ-law 8kHz"""
        try:
            # Resamplear de 24kHz a 8kHz
            pcm_data = audioop.ratecv(
                pcm_data,
                2,  # bytes por muestra
                1,  # canales
                OPENAI_SAMPLE_RATE,
                TWILIO_SAMPLE_RATE,
                None
            )[0]

            # Convertir PCM16 a μ-law
            mulaw_data = audioop.lin2ulaw(pcm_data, 2)

            return mulaw_data
        except Exception as e:
            logger.error(f"Error convirtiendo PCM16 a μ-law: {e}")
            return b''

    async def handle_twilio_message(self, message: str):
        """Procesa mensajes de Twilio"""
        try:
            data = json.loads(message)

            if data['event'] == 'start':
                self.stream_sid = data['start']['streamSid']
                self.call_sid = data['start']['callSid']
                logger.info(f"Stream iniciado - SID: {self.stream_sid}")

            elif data['event'] == 'media':
                # Audio recibido de Twilio
                audio_payload = base64.b64decode(data['media']['payload'])

                # Convertir μ-law a PCM16
                pcm_audio = self.mulaw_to_pcm16(audio_payload)

                if pcm_audio and self.openai_ws:
                    # Enviar audio a OpenAI
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm_audio).decode('utf-8')
                    }
                    await self.openai_ws.send(json.dumps(audio_append))

            elif data['event'] == 'stop':
                logger.info("Stream detenido por Twilio")

        except Exception as e:
            logger.error(f"Error procesando mensaje de Twilio: {e}")

    async def handle_openai_message(self, message: str):
        """Procesa mensajes de OpenAI"""
        try:
            data = json.loads(message)

            if data['type'] == 'response.audio.delta':
                # Audio recibido de OpenAI
                audio_data = base64.b64decode(data['delta'])

                # Convertir PCM16 a μ-law
                mulaw_audio = self.pcm16_to_mulaw(audio_data)

                if mulaw_audio and self.twilio_ws:
                    # Enviar audio a Twilio
                    media_message = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {
                            "payload": base64.b64encode(mulaw_audio).decode('utf-8')
                        }
                    }
                    await self.twilio_ws.send(json.dumps(media_message))

            elif data['type'] == 'response.audio_transcript.done':
                logger.info(f"Transcripción de respuesta: {data.get('transcript', '')}")

            elif data['type'] == 'conversation.item.input_audio_transcription.completed':
                logger.info(f"Transcripción de entrada: {data.get('transcript', '')}")

            elif data['type'] == 'error':
                logger.error(f"Error de OpenAI: {data}")

        except Exception as e:
            logger.error(f"Error procesando mensaje de OpenAI: {e}")

    async def bridge_connection(self, twilio_websocket, path):
        """Maneja la conexión entre Twilio y OpenAI"""
        self.twilio_ws = twilio_websocket
        logger.info(f"Cliente Twilio conectado desde {twilio_websocket.remote_address}")

        try:
            # Conectar a OpenAI
            await self.connect_to_openai()

            # Crear tareas para manejar mensajes bidireccionales
            async def twilio_to_openai():
                async for message in self.twilio_ws:
                    if isinstance(message, str):
                        await self.handle_twilio_message(message)

            async def openai_to_twilio():
                async for message in self.openai_ws:
                    if isinstance(message, str):
                        await self.handle_openai_message(message)

            # Ejecutar ambas tareas concurrentemente
            await asyncio.gather(
                twilio_to_openai(),
                openai_to_twilio()
            )

        except websockets.exceptions.ConnectionClosed:
            logger.info("Conexión cerrada")
        except Exception as e:
            logger.error(f"Error en el bridge: {e}")
        finally:
            # Limpiar conexiones
            if self.openai_ws:
                await self.openai_ws.close()
            logger.info("Conexión finalizada")


async def main():
    """Función principal"""
    # Verificar configuración
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY no está configurada")
        return

    logger.info(f"Iniciando servidor WebSocket en {WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

    # Crear instancia del bridge
    bridge = TwilioOpenAIBridge()

    # Iniciar servidor WebSocket
    async with websockets.serve(
            bridge.bridge_connection,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT
    ):
        logger.info("Servidor WebSocket iniciado. Esperando conexiones...")
        await asyncio.Future()  # Ejecutar indefinidamente


if __name__ == "__main__":
    asyncio.run(main())