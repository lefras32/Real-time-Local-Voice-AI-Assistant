from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import app as backend


class FakeStream:
    def __init__(self, fragments: list[str | None]) -> None:
        self.fragments = fragments

    async def __aiter__(self) -> AsyncIterator[SimpleNamespace]:
        for fragment in self.fragments:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=fragment),
                    )
                ]
            )


class FakeWebSocket:
    def __init__(self, llm_stream: FakeStream) -> None:
        create = AsyncMock(return_value=llm_stream)
        completions = SimpleNamespace(create=create)
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                llm=SimpleNamespace(
                    chat=SimpleNamespace(completions=completions),
                )
            )
        )
        self.events: list[dict[str, object]] = []
        self.audio_frames: list[bytes] = []

    async def send_json(self, event: dict[str, object]) -> None:
        self.events.append(event)

    async def send_bytes(self, audio: bytes) -> None:
        self.audio_frames.append(audio)


class FakeCommunicate:
    def __init__(self, text: str, voice: str) -> None:
        self.text = text
        self.voice = voice

    async def stream(self) -> AsyncIterator[dict[str, object]]:
        yield {"type": "audio", "data": b"fake-mp3-frame"}
        yield {"type": "SentenceBoundary"}


@pytest.fixture
def browser_client() -> Iterator[TestClient]:
    with (
        patch("app.load_whisper_model", return_value=object()),
        TestClient(backend.app) as client,
    ):
        yield client


def test_browser_routes_are_ready_for_github_package(
    browser_client: TestClient,
) -> None:
    page = browser_client.get("/")
    assert page.status_code == 200
    assert '<html lang="en">' in page.text
    assert page.headers["cache-control"].startswith("no-cache")

    worklet = browser_client.get("/static/capture-worklet.js")
    assert worklet.status_code == 200
    assert "registerProcessor" in worklet.text

    assert browser_client.get("/health").json() == {"status": "ok"}


def test_sentence_buffer_handles_split_tokens_and_decimal_points() -> None:
    buffer = backend.SentenceBuffer()

    assert buffer.feed("The value is 3.") == []
    assert buffer.feed("14. The next sentence!") == ["The value is 3.14.", "The next sentence!"]
    assert buffer.feed(" Final thought") == []
    assert buffer.flush() == "Final thought"
    assert buffer.flush() is None


@pytest.mark.asyncio
async def test_stream_answer_sends_text_and_audio_for_each_sentence() -> None:
    websocket = FakeWebSocket(
        FakeStream(["First sentence.", " Second", " sentence!"])
    )

    with patch("app.edge_tts.Communicate", FakeCommunicate):
        await backend.stream_answer(websocket, "A test prompt.")

    text_events = [
        event["text"]
        for event in websocket.events
        if event["type"] == "assistant_text"
    ]
    audio_start_events = [
        event for event in websocket.events if event["type"] == "audio_start"
    ]
    audio_end_events = [
        event for event in websocket.events if event["type"] == "audio_end"
    ]

    assert text_events == ["First sentence.", "Second sentence!"]
    assert len(audio_start_events) == 2
    assert len(audio_end_events) == 2
    assert websocket.audio_frames == [b"fake-mp3-frame", b"fake-mp3-frame"]


@pytest.mark.asyncio
async def test_stream_answer_rejects_empty_llm_response() -> None:
    websocket = FakeWebSocket(FakeStream([None]))

    with pytest.raises(RuntimeError, match="without returning assistant text"):
        await backend.stream_answer(websocket, "A test prompt.")
