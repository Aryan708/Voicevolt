"""VoiceVault web server: REST API, WebSocket voice channel and the web UI.

    uvicorn voicevault.server.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..ingest.documents import DOC_EXTS, ingest_document
from ..ingest.media import AUDIO_EXTS, VIDEO_EXTS
from ..llm import LLM, get_llm
from ..rag import RAGPipeline
from ..retrieval import HybridRetriever, Indexer, open_store
from ..store import VectorStore
from ..voice.agent import VoiceSession
from ..voice.stt import STT, get_stt
from ..voice.tts import TTS, get_tts

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
ALLOWED_EXTS = DOC_EXTS | AUDIO_EXTS | VIDEO_EXTS


# ---------------------------------------------------------------- components

@dataclass
class Components:
    store: VectorStore
    indexer: Indexer
    retriever: HybridRetriever
    rag: RAGPipeline
    tts: TTS
    _stt: Optional[STT] = None
    stt_factory: Any = get_stt
    stt_error: Optional[str] = None
    jobs: dict = field(default_factory=dict)

    @property
    def stt(self) -> Optional[STT]:
        """Whisper is loaded on first voice use so text-only users start instantly."""
        if self._stt is None and self.stt_error is None:
            try:
                self._stt = self.stt_factory()
            except Exception as e:
                self.stt_error = f"Speech recognition unavailable: {e}"
                log.warning(self.stt_error)
        return self._stt


def build_components(llm: Optional[LLM] = None, tts: Optional[TTS] = None,
                     stt: Optional[STT] = None) -> Components:
    store, temb, iemb = open_store()
    retriever = HybridRetriever(store, temb, iemb)
    return Components(store=store, indexer=Indexer(store, temb, iemb), retriever=retriever,
                      rag=RAGPipeline(retriever, llm or get_llm()), tts=tts or get_tts(),
                      _stt=stt)


# ------------------------------------------------------------- file urls

def _url_for(path: Optional[str]) -> Optional[str]:
    """Map an absolute data path to a /files URL the browser can load."""
    if not path:
        return None
    p = Path(path).resolve()
    roots = [("processed", settings.processed_dir.resolve()), ("raw", settings.raw_dir.resolve())]
    # most specific root first, in case one data folder is configured inside the other
    for name, root in sorted(roots, key=lambda r: len(r[1].parts), reverse=True):
        try:
            return f"/files/{name}/{p.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return None


def publicize(sources: list[dict]) -> list[dict]:
    out = []
    for s in sources:
        s = dict(s)
        s["image_url"] = _url_for(s.pop("image", None))
        raw = settings.raw_dir / s["source"]
        s["media_url"] = _url_for(str(raw)) if raw.exists() else None
        out.append(s)
    return out


def _safe_name(name: str) -> str:
    name = Path(name).name
    stem = re.sub(r"[^\w\-. ]+", "_", Path(name).stem, flags=re.UNICODE).strip() or "file"
    return f"{stem}{Path(name).suffix.lower()}"


# ------------------------------------------------------------------ models

class SearchReq(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=50)
    source: Optional[str] = None
    modality: Optional[list[str]] = None


class AskReq(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list)
    source: Optional[str] = None
    modality: Optional[list[str]] = None


class FeedbackReq(BaseModel):
    question: str
    answer: str
    rating: int = Field(..., ge=-1, le=1)   # 1 = thumbs up, -1 = thumbs down
    comment: str = ""
    sources: list[dict] = Field(default_factory=list)
    query: Optional[str] = None


# --------------------------------------------------------------------- app

def create_app(components: Optional[Components] = None) -> FastAPI:
    state: dict[str, Components] = {}
    build_lock = threading.Lock()

    def comp() -> Components:
        with build_lock:  # first requests may arrive together
            if "c" not in state:
                state["c"] = components or build_components()
        return state["c"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if "c" in state:
            state["c"].store.close()

    app = FastAPI(title="VoiceVault", version="1.0.0", lifespan=lifespan)

    # ---------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/files/{root}/{path:path}", include_in_schema=False)
    def files(root: str, path: str):
        base = {"processed": settings.processed_dir, "raw": settings.raw_dir}.get(root)
        if base is None:
            raise HTTPException(404)
        target = (base / path).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            raise HTTPException(404)
        return FileResponse(target)  # supports Range requests, so video seeking works

    # ------------------------------------------------------------------ api
    @app.get("/api/health")
    def health():
        c = comp()
        return {"status": "ok", "llm": getattr(c.rag.llm, "name", "?"),
                "tts": getattr(c.tts, "name", "?"),
                "stt": getattr(c._stt, "name", None) or settings.stt_provider,
                "stt_error": c.stt_error,
                "embeddings": [c.retriever.text_emb.name, c.retriever.img_emb.name]}

    @app.get("/api/stats")
    def stats():
        return comp().store.stats()

    @app.post("/api/search")
    def search(req: SearchReq):
        from ..rag import source_dict

        hits = comp().retriever.search(req.query, k=req.k, source=req.source,
                                       modality=req.modality)
        return {"query": req.query,
                "results": publicize([source_dict(i, h) for i, h in enumerate(hits, 1)])}

    @app.post("/api/ask")
    def ask(req: AskReq):
        a = comp().rag.ask(req.question, req.history, source=req.source, modality=req.modality)
        return {"question": a.question, "query": a.query, "answer": a.text, "cited": a.cited,
                "sources": publicize(a.sources(only_cited=False)), "timings": a.timings}

    @app.post("/api/feedback")
    def feedback(req: FeedbackReq):
        settings.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "llm": getattr(comp().rag.llm, "name", "?"), **req.model_dump()}
        with open(settings.feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return {"ok": True}

    @app.post("/api/ingest")
    async def upload(files: list[UploadFile] = File(...), vision: bool = True):
        c = comp()
        settings.raw_dir.mkdir(parents=True, exist_ok=True)
        jobs = []
        for up in files:
            name = _safe_name(up.filename or "upload")
            if Path(name).suffix.lower() not in ALLOWED_EXTS:
                raise HTTPException(400, f"Unsupported file type: {name}")
            dest = settings.raw_dir / name
            with open(dest, "wb") as out:
                while chunk := await up.read(1 << 20):
                    out.write(chunk)
            job_id = uuid.uuid4().hex[:12]
            c.jobs[job_id] = {"id": job_id, "file": name, "status": "queued"}
            threading.Thread(target=_run_job, args=(c, job_id, dest, vision), daemon=True).start()
            jobs.append(c.jobs[job_id])
        return {"jobs": jobs}

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": list(comp().jobs.values())[-50:]}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = comp().jobs.get(job_id)
        if not j:
            raise HTTPException(404)
        return j

    # ------------------------------------------------------------ voice ws
    @app.websocket("/ws/voice")
    async def voice(ws: WebSocket):
        await ws.accept()
        c = comp()
        send_lock = asyncio.Lock()

        async def send(ev: dict) -> None:
            if ev.get("type") in ("sources", "done") and "sources" in ev:
                ev = {**ev, "sources": publicize(ev["sources"])}
            async with send_lock:
                try:
                    await ws.send_json(ev)
                except Exception:
                    pass  # client went away mid-answer

        session = VoiceSession(c.rag, None, c.tts, send)  # type: ignore[arg-type]
        await send({"type": "ready", "llm": getattr(c.rag.llm, "name", "?"),
                    "tts": getattr(c.tts, "name", "?")})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if session.stt is None:
                        stt = await asyncio.to_thread(lambda: c.stt)
                        if stt is None:
                            await send({"type": "error", "error": c.stt_error})
                            continue
                        session.stt = stt
                    await session.feed_audio(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "text" and data.get("text", "").strip():
                        await session.ask_text(data["text"].strip())
                    elif kind == "interrupt":
                        await session.interrupt("user")
                    elif kind == "config":
                        lang = data.get("language")
                        session.language = lang if lang in ("en", "hi") else None
                        if "barge_in" in data:
                            session.barge_in = bool(data["barge_in"])
                    elif kind == "reset":
                        await session.interrupt("reset")
                        session.history.clear()
                    elif kind == "ping":
                        await send({"type": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()

    return app


def _run_job(c: Components, job_id: str, path: Path, vision: bool) -> None:
    job = c.jobs[job_id]
    job.update(status="processing", started=time.time())
    try:
        ext = path.suffix.lower()
        if ext in DOC_EXTS:
            result = ingest_document(path, vision=vision)
        else:
            from ..ingest.pipeline import ingest
            result = ingest(path, captions=vision)
        stats = c.indexer.index(result)
        job.update(status="done", **stats, seconds=round(time.time() - job["started"], 1))
    except Exception as e:
        log.exception("ingest job failed")
        job.update(status="error", error=str(e))


app = create_app()
