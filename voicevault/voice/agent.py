"""Real-time voice session: mic audio in, spoken cited answers out.

    mic PCM ──▶ VAD/endpointing ──▶ STT ──▶ RAG (retrieve + LLM stream)
                    │                                  │ tokens
                    │ speech starts while              ▼
                    │ the agent is talking      sentence splitter ──▶ TTS ──▶ audio events
                    └────────────▶ barge-in: cancel generation + stop playback

Everything blocking (Whisper, retrieval, LLM HTTP stream, TTS) runs in worker
threads so one session never blocks the server's event loop. TTS of sentence N
overlaps with the LLM writing sentence N+1.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from ..config import settings
from ..llm import Message
from ..rag import RAGPipeline, strip_citations
from .sentences import SentenceStreamer
from .stt import STT
from .tts import TTS
from .vad import SAMPLE_RATE, UtteranceSegmenter

log = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]


class VoiceSession:
    def __init__(
        self,
        rag: RAGPipeline,
        stt: STT,
        tts: TTS,
        send: Send,
        language: Optional[str] = None,
        barge_in: Optional[bool] = None,
        segmenter: Optional[UtteranceSegmenter] = None,
        metrics_path: Optional[Path] = settings.metrics_path,
        max_history: int = 8,
    ):
        self.rag, self.stt, self.tts, self.send = rag, stt, tts, send
        self.language = language
        self.barge_in = settings.barge_in if barge_in is None else barge_in
        self.segmenter = segmenter or UtteranceSegmenter(silence_ms=settings.vad_silence_ms)
        self.metrics_path = metrics_path
        self.max_history = max_history
        self.history: list[Message] = []
        self.task: Optional[asyncio.Task] = None
        self._cancel: Optional[threading.Event] = None
        self.turn = 0

    # ------------------------------------------------------------ inputs
    @property
    def busy(self) -> bool:
        return self.task is not None and not self.task.done()

    async def feed_audio(self, pcm: bytes) -> None:
        for ev in self.segmenter.feed(pcm):
            if ev.type == "start":
                await self.send({"type": "vad", "state": "speech"})
                if self.busy and self.barge_in:
                    await self.interrupt("barge_in")
            elif ev.type == "end":
                await self.send({"type": "vad", "state": "silence"})
                if ev.audio is not None:
                    await self._start(audio=ev.audio)

    async def ask_text(self, text: str) -> None:
        await self._start(text=text)

    async def interrupt(self, reason: str = "user") -> None:
        if self._cancel:
            self._cancel.set()
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except BaseException:  # noqa: BLE001 - cancelled / already failed
                pass
        await self.send({"type": "interrupt", "reason": reason})

    async def wait(self) -> None:
        if self.task:
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        if self.busy:
            await self.interrupt("closed")

    # ------------------------------------------------------------ turn
    async def _start(self, audio: Optional[np.ndarray] = None, text: Optional[str] = None):
        if self.busy:
            await self.interrupt("new_turn")
        self.turn += 1
        self._cancel = threading.Event()
        self.task = asyncio.create_task(self._respond(self.turn, self._cancel, audio, text))

    async def _respond(self, turn: int, cancel: threading.Event,
                       audio: Optional[np.ndarray], text: Optional[str]) -> None:
        t0 = time.perf_counter()  # = end of user speech (or text submit)
        m: dict = {"turn": turn, "ts": time.time(), "mode": "voice" if audio is not None else "text",
                   "stt_model": getattr(self.stt, "name", ""), "llm": getattr(self.rag.llm, "name", ""),
                   "tts": getattr(self.tts, "name", "")}
        ms = lambda: round((time.perf_counter() - t0) * 1000)  # noqa: E731
        try:
            if audio is not None:
                m["audio_s"] = round(len(audio) / SAMPLE_RATE, 2)
                res = await asyncio.to_thread(self.stt.transcribe, audio, self.language)
                m["stt_ms"] = ms()
                text = res.text.strip()
                await self.send({"type": "transcript", "turn": turn, "text": text,
                                 "language": res.language, "confidence": res.confidence})
                if not text:
                    return
            else:
                await self.send({"type": "transcript", "turn": turn, "text": text, "language": None})

            loop = asyncio.get_running_loop()
            events: asyncio.Queue = asyncio.Queue()
            history = list(self.history)

            def produce() -> None:
                try:
                    for ev in self.rag.stream(text, history, voice=True):
                        if cancel.is_set():
                            break
                        loop.call_soon_threadsafe(events.put_nowait, ev)
                except Exception as e:  # surfaced to the client, logged
                    log.exception("RAG failed")
                    loop.call_soon_threadsafe(events.put_nowait, {"type": "error", "error": str(e)})
                finally:
                    loop.call_soon_threadsafe(events.put_nowait, None)

            producer = loop.run_in_executor(None, produce)
            sentences: asyncio.Queue = asyncio.Queue()
            speaker = asyncio.create_task(self._speak(sentences, turn, cancel, m, ms))
            splitter = SentenceStreamer()
            answer = None

            while (ev := await events.get()) is not None:
                kind = ev["type"]
                if kind == "sources":
                    m["retrieval_done_ms"] = ms()
                    await self.send({"type": "sources", "turn": turn, "query": ev["query"],
                                     "sources": ev["sources"]})
                elif kind == "token":
                    if "first_token_ms" not in m:
                        m["first_token_ms"] = ms()
                    await self.send({"type": "token", "turn": turn, "text": ev["text"]})
                    for s in splitter.feed(ev["text"]):
                        sentences.put_nowait(s)
                elif kind == "done":
                    answer = ev["answer"]
                elif kind == "error":
                    await self.send({"type": "error", "turn": turn, "error": ev["error"]})
            for s in splitter.flush():
                sentences.put_nowait(s)
            sentences.put_nowait(None)
            await speaker
            await producer

            if answer is not None:
                m["total_ms"] = ms()
                m.update({f"rag_{k}": v for k, v in answer.timings.items()})
                m["cited"] = len(answer.cited)
                self.history += [{"role": "user", "content": text},
                                 {"role": "assistant", "content": answer.text}]
                self.history = self.history[-self.max_history:]
                await self.send({"type": "done", "turn": turn, "question": text,
                                 "answer": answer.text, "query": answer.query,
                                 "cited": answer.cited, "sources": answer.sources(),
                                 "metrics": {k: v for k, v in m.items() if k.endswith("_ms")}})
                self._log(m)
        except asyncio.CancelledError:
            cancel.set()
            m["interrupted_ms"] = ms()
            self._log(m)
            raise

    async def _speak(self, sentences: asyncio.Queue, turn: int, cancel: threading.Event,
                     m: dict, ms) -> None:
        seq = 0
        while (s := await sentences.get()) is not None:
            if cancel.is_set():
                continue
            spoken = strip_citations(s).strip()
            if not spoken:
                continue
            try:
                speech = await asyncio.to_thread(self.tts.synthesize, spoken)
            except Exception as e:
                log.warning("TTS failed: %s", e)
                speech = None
            if cancel.is_set() or speech is None:
                continue
            if "first_audio_ms" not in m:
                m["first_audio_ms"] = ms()
            await self.send({"type": "audio", "turn": turn, "seq": seq, "text": spoken,
                             "mime": speech.mime, "data": base64.b64encode(speech.audio).decode()})
            seq += 1

    def _log(self, m: dict) -> None:
        if not self.metrics_path:
            return
        try:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(m) + "\n")
        except OSError as e:
            log.warning("could not write metrics: %s", e)
