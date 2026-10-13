from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging

import av
import numpy as np
import sounddevice as sd
import soundfile as sf
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

SAMPLE_RATE = 16_000
CHANNELS = 1
DEFAULT_SERVER = "ws://127.0.0.1:8000/ws"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("voice_client")


def record_wav(duration: float, device: int | None) -> bytes:
    logger.info("Recording for %.1f seconds. Speak now...", duration)
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device,
    )
    sd.wait()
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


async def play_decoded_frames(
    frames: list[av.AudioFrame],
    resampler: av.AudioResampler,
    output_stream: sd.OutputStream,
) -> None:
    for frame in frames:
        for pcm_frame in resampler.resample(frame):
            samples = np.ascontiguousarray(
                pcm_frame.to_ndarray().reshape(-1, 1)
            )
            await asyncio.to_thread(output_stream.write, samples)


async def run_turn(
    server: str,
    duration: float,
    device: int | None,
) -> None:
    audio_bytes = await asyncio.to_thread(record_wav, duration, device)
    decoder: av.CodecContext | None = None
    resampler: av.AudioResampler | None = None
    output_stream: sd.OutputStream | None = None

    try:
        async with connect(server, max_size=20 * 1024 * 1024, open_timeout=10) as ws:
            await ws.send(audio_bytes)
            await ws.send('{"type":"end_audio"}')

            async for message in ws:
                if isinstance(message, bytes):
                    if decoder is None or resampler is None or output_stream is None:
                        raise RuntimeError("Received audio data before audio_start")
                    for packet in decoder.parse(message):
                        frames = decoder.decode(packet)
                        await play_decoded_frames(frames, resampler, output_stream)
                    continue

                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "ready":
                    continue
                if event_type == "status":
                    logger.info("Status: %s", event.get("stage", "working"))
                elif event_type == "transcript":
                    logger.info("You: %s", event["text"])
                elif event_type == "audio_start":
                    decoder = av.CodecContext.create("mp3", "r")
                    resampler = av.AudioResampler(
                        format="s16",
                        layout="mono",
                        rate=24_000,
                    )
                    output_stream = sd.OutputStream(
                        samplerate=24_000,
                        channels=1,
                        dtype="int16",
                    )
                    await asyncio.to_thread(output_stream.start)
                    logger.info("Assistant: %s", event["text"])
                elif event_type == "audio_end":
                    if (
                        decoder is not None
                        and resampler is not None
                        and output_stream is not None
                    ):
                        for packet in decoder.parse(None):
                            frames = decoder.decode(packet)
                            await play_decoded_frames(frames, resampler, output_stream)
                        await play_decoded_frames(
                            decoder.decode(None),
                            resampler,
                            output_stream,
                        )
                        for pcm_frame in resampler.resample(None):
                            samples = np.ascontiguousarray(
                                pcm_frame.to_ndarray().reshape(-1, 1)
                            )
                            await asyncio.to_thread(output_stream.write, samples)
                        await asyncio.to_thread(output_stream.stop)
                        await asyncio.to_thread(output_stream.close)
                        decoder = None
                        resampler = None
                        output_stream = None
                elif event_type == "error":
                    logger.error("Server error: %s", event.get("message", "unknown error"))
                elif event_type == "response_done":
                    break

    except ConnectionClosed as error:
        raise RuntimeError(f"WebSocket connection closed unexpectedly: {error}") from error
    finally:
        if output_stream is not None:
            await asyncio.to_thread(output_stream.close)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a voice question and play the assistant's streamed response."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, help="WebSocket URL")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device index",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")

    logger.info("Voice assistant client. Press Ctrl+C to quit.")
    while True:
        try:
            input("\nPress Enter to record, or Ctrl+C to quit...")
            await run_turn(args.server, args.duration, args.device)
        except EOFError:
            return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nGoodbye.")
