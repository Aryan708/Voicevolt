"""Week 3 server tests: REST API, file serving, upload->ingest->search, WebSocket voice."""
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import corpus
from voicevault.config import settings
from voicevault.llm import FakeLLM
from voicevault.rag import RAGPipeline
from voicevault.server.app import Components, create_app
from voicevault.voice.stt import FakeSTT
from voicevault.voice.tts import FakeTTS


@pytest.fixture
def client(kb, monkeypatch):
    store, indexer, retriever, tmp = kb
    monkeypatch.setattr(settings, "processed_dir", tmp)
    monkeypatch.setattr(settings, "raw_dir", tmp / "raw")
    (tmp / "raw").mkdir()
    (tmp / "raw" / "sync.mp4").write_bytes(b"\x00" * 2048)
    monkeypatch.setattr(type(settings), "feedback_path", property(lambda s: tmp / "fb.jsonl"))
    monkeypatch.setattr(type(settings), "metrics_path", property(lambda s: tmp / "m.jsonl"))
    for r in corpus(tmp):
        indexer.index(r)
    comps = Components(store=store, indexer=indexer, retriever=retriever,
                       rag=RAGPipeline(retriever, FakeLLM(), k=3), tts=FakeTTS(),
                       stt_factory=lambda: FakeSTT("what is the hiring plan"))
    with TestClient(create_app(comps)) as c:
        yield c, tmp


def test_health_and_stats(client):
    c, _ = client
    assert c.get("/api/health").json()["llm"] == "fake"
    s = c.get("/api/stats").json()
    assert s["chunks"] == 4 and s["sources"]["sync.mp4"] == 2


def test_search_returns_browser_urls(client):
    c, _ = client
    r = c.post("/api/search", json={"query": "scanned invoice 4471", "k": 3}).json()
    top = r["results"][0]
    assert top["chunk_id"] == "s1"
    assert top["image_url"] == "/files/processed/s1.png"
    assert c.get(top["image_url"]).status_code == 200


def test_ask_has_answer_citations_and_media_url(client):
    c, _ = client
    r = c.post("/api/ask", json={"question": "Why did the Chennai launch slip?"}).json()
    assert r["cited"] == [1, 2]
    first = r["sources"][0]
    assert first["citation"] == "sync.mp4 @ 12:30-13:15"
    assert first["media_url"] == "/files/raw/sync.mp4"
    assert first["image_url"] == "/files/processed/f1.jpg"


def test_file_serving_blocks_path_traversal(client):
    c, tmp = client
    (tmp.parent / "secret.txt").write_text("nope")
    for bad in ["/files/processed/../secret.txt", "/files/processed/%2e%2e/secret.txt",
                "/files/etc/passwd", "/files/raw/../../secret.txt"]:
        assert c.get(bad).status_code == 404, bad


def test_video_range_request_for_seeking(client):
    c, _ = client
    r = c.get("/files/raw/sync.mp4", headers={"Range": "bytes=100-199"})
    assert r.status_code == 206 and len(r.content) == 100


def test_feedback_is_logged(client):
    c, tmp = client
    c.post("/api/feedback", json={"question": "q", "answer": "a", "rating": -1, "comment": "wrong"})
    assert '"rating": -1' in (tmp / "fb.jsonl").read_text()
    assert c.post("/api/feedback", json={"question": "q", "answer": "a", "rating": 5}).status_code == 422


def test_upload_ingests_and_becomes_searchable(client, monkeypatch):
    c, tmp = client
    buf = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(buf, format="PNG")
    r = c.post("/api/ingest?vision=false",
               files=[("files", ("Whiteboard Q4 plan!.png", buf.getvalue(), "image/png"))])
    job = r.json()["jobs"][0]
    assert job["file"] == "Whiteboard Q4 plan_.png"
    for _ in range(100):
        j = c.get(f"/api/jobs/{job['id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    assert "Whiteboard Q4 plan_.png" in c.get("/api/stats").json()["sources"]
    assert c.post("/api/ingest", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]
                  ).status_code == 400


def _drain(ws, until="done", limit=200):
    events = []
    for _ in range(limit):
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] == until:
            break
    return events


def test_websocket_text_turn(client):
    c, _ = client
    with c.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "text", "text": "Why did the Chennai launch slip?"})
        events = _drain(ws)
    types = [e["type"] for e in events]
    assert types[0] == "transcript" and "sources" in types and "audio" in types
    done = events[-1]
    assert done["cited"] == [1, 2]
    assert done["sources"][0]["image_url"] == "/files/processed/f1.jpg"
    assert "first_audio_ms" in done["metrics"]


def test_websocket_voice_turn_from_pcm(client):
    c, _ = client
    sr = 16000
    t = np.arange(sr) / sr
    speech = (0.3 * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    quiet = np.zeros(int(sr * 1.0), np.int16)
    pcm = np.concatenate([quiet[:8000], speech, quiet]).tobytes()
    with c.websocket_connect("/ws/voice") as ws:
        ws.receive_json()
        for i in range(0, len(pcm), 640):       # 20 ms frames, like the browser sends
            ws.send_bytes(pcm[i:i + 640])
        events = _drain(ws)
    transcript = next(e for e in events if e["type"] == "transcript")
    assert transcript["text"] == "what is the hiring plan"
    assert events[-1]["type"] == "done" and "stt_ms" in events[-1]["metrics"]


def test_index_page_served(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200 and "VoiceVault" in r.text and "/ws/voice" in r.text
