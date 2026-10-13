from __future__ import annotations

import asyncio
import ctypes
import io
import logging
import sys
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import ctranslate2
import edge_tts
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("voice_assistant")

SYSTEM_PROMPT = (
    "You are a concise voice assistant. Always answer in English, regardless of "
    "the language used by the user. Use one to three short, natural sentences. "
    "Do not use Markdown, emoji, lists, headings, or decorative symbols. Use "
    "plain text and ordinary punctuation."
)
SENTENCE_ENDINGS = frozenset(".!?")
CLOSING_PUNCTUATION = frozenset("\"'»”’)]}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_model: str = "qwen3.5-9b"
    llm_timeout_seconds: float = 90.0
    whisper_model: str = "base"
    whisper_device: str = "auto"
    whisper_compute_type: str | None = None
    whisper_language: str | None = None
    tts_voice: str = "en-US-AriaNeural"
    max_audio_bytes: int = 15 * 1024 * 1024


settings = Settings()


def cuda_runtime_available() -> bool:
    if ctranslate2.get_cuda_device_count() == 0:
        return False

    if sys.platform == "win32":
        try:
            ctypes.WinDLL("cublas64_12.dll")
        except OSError as error:
            logger.warning(
                "CTranslate2 detects a CUDA device, but cublas64_12.dll cannot be loaded. "
                "Using CPU; install the CUDA 12 runtime to enable GPU inference. Details: %s",
                error,
            )
            return False

    return True


def load_whisper_model(
    device_override: str | None = None,
    compute_type_override: str | None = None,
) -> WhisperModel:
    device = (device_override or settings.whisper_device).lower()
    if device == "auto":
        device = "cuda" if cuda_runtime_available() else "cpu"

    compute_type = compute_type_override or settings.whisper_compute_type
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    logger.info(
        "Loading Whisper model=%s device=%s compute_type=%s",
        settings.whisper_model,
        device,
        compute_type,
    )
    return WhisperModel(
        settings.whisper_model,
        device=device,
        compute_type=compute_type,
    )


def transcribe_wav(model: WhisperModel, audio_bytes: bytes) -> str:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16000 Hz audio, received {sample_rate} Hz")
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    else:
        audio = audio[:, 0]

    segments, _ = model.transcribe(
        audio,
        language=settings.whisper_language,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe_with_cuda_fallback(
    model: WhisperModel,
    audio_bytes: bytes,
) -> tuple[str, WhisperModel | None]:
    try:
        return transcribe_wav(model, audio_bytes), None
    except RuntimeError as error:
        message = str(error).lower()
        if "cuda" not in message and "cublas" not in message and "cudnn" not in message:
            raise

        logger.warning(
            "Whisper CUDA inference failed (%s). Retrying on CPU with int8.",
            error,
        )
        cpu_model = load_whisper_model(
            device_override="cpu",
            compute_type_override="int8",
        )
        return transcribe_wav(cpu_model, audio_bytes), cpu_model


class SentenceBuffer:
    """Extract completed sentences from arbitrarily split LLM token fragments."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, fragment: str) -> list[str]:
        self._buffer += fragment
        sentences: list[str] = []

        while True:
            boundary = self._find_boundary()
            if boundary is None:
                break
            sentence = self._buffer[:boundary].strip()
            self._buffer = self._buffer[boundary:].lstrip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        sentence = self._buffer.strip()
        self._buffer = ""
        return sentence or None

    def _find_boundary(self) -> int | None:
        for index, character in enumerate(self._buffer):
            if character not in SENTENCE_ENDINGS:
                continue
            if character == "." and index + 1 == len(self._buffer):
                continue
            if (
                character == "."
                and index > 0
                and index + 1 < len(self._buffer)
                and self._buffer[index - 1].isdigit()
                and self._buffer[index + 1].isdigit()
            ):
                continue

            end = index + 1
            while end < len(self._buffer) and self._buffer[end] in SENTENCE_ENDINGS:
                end += 1
            while end < len(self._buffer) and self._buffer[end] in CLOSING_PUNCTUATION:
                end += 1
            return end
        return None


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        application.state.executor = executor
        loop = asyncio.get_running_loop()
        try:
            application.state.whisper = await loop.run_in_executor(
                executor,
                load_whisper_model,
            )
            application.state.llm = AsyncOpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout_seconds,
            )
            logger.info("Voice assistant is ready")
            yield
        finally:
            llm = getattr(application.state, "llm", None)
            if llm is not None:
                await llm.close()
            executor.shutdown(wait=True, cancel_futures=True)

    application = FastAPI(
        title="Real-time Local Voice AI Assistant",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )

    @application.get("/", include_in_schema=False)
    async def home() -> FileResponse:
        return FileResponse(
            Path(__file__).parent / "static" / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @application.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.websocket("/ws")
    async def voice_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await websocket.send_json({"type": "ready"})
            audio_bytes = await receive_audio(websocket)
            await websocket.send_json({"type": "status", "stage": "transcribing"})

            loop = asyncio.get_running_loop()
            transcript, fallback_model = await loop.run_in_executor(
                websocket.app.state.executor,
                transcribe_with_cuda_fallback,
                websocket.app.state.whisper,
                audio_bytes,
            )
            if fallback_model is not None:
                websocket.app.state.whisper = fallback_model
            if not transcript:
                await websocket.send_json(
                    {"type": "error", "message": "No speech was recognized. Please try again."}
                )
                return

            await websocket.send_json({"type": "transcript", "text": transcript})
            await websocket.send_json({"type": "status", "stage": "generating"})
            await stream_answer(websocket, transcript)
            await websocket.send_json({"type": "response_done"})
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except (APIConnectionError, APIStatusError) as error:
            logger.exception("LM Studio request failed")
            await send_error(websocket, f"LM Studio is unavailable: {error}")
        except (ValueError, sf.SoundFileRuntimeError) as error:
            logger.warning("Invalid audio received: %s", error)
            await send_error(websocket, f"Invalid audio file: {error}")
        except Exception as error:
            logger.exception("Voice request failed")
            await send_error(websocket, f"Request processing failed: {error}")
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return application


async def receive_audio(websocket: WebSocket) -> bytes:
    chunks: list[bytes] = []
    total_bytes = 0

    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))

        chunk = message.get("bytes")
        if chunk is not None:
            total_bytes += len(chunk)
            if total_bytes > settings.max_audio_bytes:
                raise ValueError(
                    f"Audio exceeds the {settings.max_audio_bytes // (1024 * 1024)} MB limit"
                )
            chunks.append(chunk)
            continue

        text = message.get("text")
        if text is None:
            continue
        if text == '{"type":"end_audio"}':
            break
        raise ValueError("Expected binary WAV data followed by {\"type\":\"end_audio\"}")

    audio = b"".join(chunks)
    if not audio:
        raise ValueError("No audio data received")
    return audio


async def stream_answer(websocket: WebSocket, prompt: str) -> None:
    llm = websocket.app.state.llm
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)

    async def synthesize_queued_sentences() -> None:
        sentence_index = 0
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            sentence_index += 1
            await websocket.send_json(
                {
                    "type": "audio_start",
                    "index": sentence_index,
                    "text": sentence,
                    "format": "mp3",
                }
            )
            try:
                communicator = edge_tts.Communicate(sentence, settings.tts_voice)
                async for audio_chunk in communicator.stream():
                    if audio_chunk["type"] == "audio":
                        await websocket.send_bytes(audio_chunk["data"])
                await websocket.send_json(
                    {"type": "audio_end", "index": sentence_index}
                )
            except Exception as error:
                logger.exception("TTS failed for sentence %d", sentence_index)
                await websocket.send_json(
                    {
                        "type": "audio_error",
                        "index": sentence_index,
                        "message": f"Text reply is available, but speech synthesis failed: {error}",
                    }
                )

    worker = asyncio.create_task(synthesize_queued_sentences())

    async def enqueue(sentence: str | None) -> None:
        put_task = asyncio.create_task(sentence_queue.put(sentence))
        done, _ = await asyncio.wait(
            {put_task, worker},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if worker in done:
            put_task.cancel()
            await asyncio.gather(put_task, return_exceptions=True)
            await worker
        await put_task

    sentence_buffer = SentenceBuffer()
    response_text_chars = 0
    try:
        response = await llm.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            temperature=0.4,
            max_tokens=768,
        )
        async for event in response:
            if not event.choices:
                continue
            fragment = event.choices[0].delta.content
            if isinstance(fragment, list):
                fragment = "".join(
                    item.get("text", "")
                    for item in fragment
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
            if not isinstance(fragment, str) or not fragment:
                continue
            response_text_chars += len(fragment)
            for sentence in sentence_buffer.feed(fragment):
                await websocket.send_json(
                    {"type": "assistant_text", "text": sentence}
                )
                await enqueue(sentence)

        final_sentence = sentence_buffer.flush()
        if final_sentence:
            await websocket.send_json(
                {"type": "assistant_text", "text": final_sentence}
            )
            await enqueue(final_sentence)
        if response_text_chars == 0:
            raise RuntimeError(
                "LM Studio finished without returning assistant text. "
                "The model may have used its token budget for internal reasoning."
            )
        await enqueue(None)
        await worker
    except Exception:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        raise


async def send_error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "message": message})
    except (RuntimeError, WebSocketDisconnect):
        logger.info("Could not send error because the client disconnected")


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
    )
