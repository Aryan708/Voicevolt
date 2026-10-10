# VoiceVault architecture

## 1. Data model

Everything becomes a `Chunk` (`voicevault/schemas.py`), the unit that is embedded, retrieved and cited:

| Field | Recordings | Documents / images |
|---|---|---|
| `text` | speech lines + slide captions, interleaved by time: `[12:30 SPEAKER_01] …` / `[12:34 SLIDE] …` | `[deck.pptx p.4] …` page text or vision OCR |
| `start`, `end` | seconds | — |
| `speakers` | diarized labels | — |
| `frames` | keyframes (path, timestamp, caption) | — |
| `page`, `image_path` | — | page number + rendered page image |
| `citation` | `meeting.mp4 @ 12:30-13:15` | `deck.pptx p.4` |

Each ingest also writes `data/processed/<file>/ingest.json` (transcript, keyframes, chunks, stage timings),
so re-indexing with a new embedding model never re-runs Whisper.

## 2. Ingestion

**Audio:** ffmpeg → 16 kHz mono WAV → faster-whisper (VAD on, word timestamps) → pyannote turns →
*word-level* speaker assignment, splitting a Whisper segment when the speaker changes mid-sentence.

**Video keyframes** (`ingest/video.py`): sample at 2 fps at 160×90, streamed from ffmpeg. Keep a frame when
the share of changed pixels vs the *last kept* frame exceeds 1.2% **and** the picture then holds still
(<0.4% change), or on a hard cut (>30%), or every 30 s regardless. Full-resolution JPEGs are then grabbed
only at those timestamps. On the demo meeting this finds all 4 slides within 0.5 s, where ffmpeg's scene
score (0.05-0.26 on these changes) found none.

**Vision captions:** each keyframe is described by a vision LLM, which copies slide text and chart numbers
exactly. "Only a person talking" frames return a sentinel and are not embedded.

**Documents:** pypdfium2 renders each page and reads its text layer. Pages with <40 characters (scans,
image-only slides) go to vision OCR. Office files are converted to PDF by LibreOffice. Long pages are split
at paragraph boundaries; all pieces share the page image.

**Chunking:** ~45 s windows that never cut a segment, with ~8 s overlap so answers at a boundary aren't
lost. Keyframes attach to the window they fall in; silent stretches with slides become visual-only chunks.

## 3. Index

Qdrant (embedded on disk, or a server via `QDRANT_URL`) with two collections:

- `chunks`: multilingual-e5 vector of the chunk text, payload = the whole chunk
- `images`: CLIP vector per page render / informative keyframe, payload → `chunk_id`

BM25 is rebuilt in memory from the `chunks` payloads whenever the store's write version changes.
`meta.json` records which embedding models built the index; a mismatch refuses to open instead of returning
garbage. Re-indexing a file deletes its old points first. Store access is serialized with a lock because
embedded Qdrant isn't safe for concurrent writers.

## 4. Retrieval

```
query ─┬─ e5("query: …")       → chunks   top 20 ─┐
       ├─ BM25 (stop-words, Devanagari-aware)      ├─ weighted RRF: Σ w / (60 + rank) → top k
       └─ CLIP text encoder    → images   top 20 ─┘   (images mapped to their chunk, best first)
```

Weights default to dense 1.0, BM25 0.8, visual 0.6. The **proof image** for a hit is the keyframe whose
caption best matches the question, else CLIP's best image, else the page. Its timestamp drives the card
badge and the video seek point.

## 5. Answering

`RAGPipeline.stream()`:
1. If there is history, rewrite the follow-up into a standalone query ("and the south?" → "South region Q3 revenue").
2. Hybrid retrieval → numbered sources `[1] meeting.mp4 @ 12:30-13:15 (video)\n<text>`.
3. Stream the LLM. The system prompt enforces: answer only from sources, cite every factual sentence,
   reply in the user's language and script, and in voice mode use 2-4 short spoken sentences.
4. Citations are parsed, and only valid `[n]` are kept.

## 6. Real-time voice (`voice/agent.py`)

```
browser AudioWorklet ─PCM16 16 kHz, 20 ms─▶ WebSocket ─▶ UtteranceSegmenter (webrtcvad or energy VAD)
   "start" while the agent is busy ──▶ barge-in: cancel task + thread flag, client stops audio
   "end" (700 ms silence) ──▶ turn task:
        STT (thread) → transcript event
        RAG stream (thread → asyncio.Queue) → token events → SentenceStreamer
                                                   └─▶ TTS worker (thread) → audio events (base64)
        done event: answer, cited sources, per-stage latency → data/logs/metrics.jsonl
```

All blocking work (Whisper, Qdrant, LLM HTTP stream, TTS) runs in worker threads, so one session never
blocks the event loop. TTS of sentence *n* overlaps generation of sentence *n+1*. `[n]` markers are
stripped before speaking. Interrupted answers are not added to history.

**Latency budget** (measured from end of user speech), targets for a cloud LLM with Edge TTS:

| Stage | Typical |
|---|---|
| endpointing (silence wait) | 700 ms (configurable) |
| Whisper small, int8 CPU, 4 s utterance | 300-700 ms |
| retrieval | 20-80 ms |
| LLM first token | 300-800 ms |
| TTS first sentence | 200-500 ms |
| **voice-to-first-audio** | **~1-2 s** |

Measure your real numbers with `python scripts/evaluate.py latency`.

## 7. Evaluation

| Layer | Metric | Where |
|---|---|---|
| ASR | corpus WER/CER, by script (Hindi/Hinglish/English), top substitutions, RTF | `finetune/evaluate_asr.py` |
| Retrieval | Hit@k, Recall@k, MRR, nDCG@k for 5 retriever configs; by language / modality | `scripts/evaluate.py retrieval` |
| Answers | faithfulness, relevance, correctness, citation precision, refusal on unanswerable | `scripts/evaluate.py rag` |
| System | per-stage p50/p90/p95, interruptions | `scripts/evaluate.py latency` |
| Users | 👍 rate, top 👎 reasons, SFT + DPO export | `scripts/evaluate.py feedback` |

## 8. Failure handling

| Failure | Behaviour |
|---|---|
| No HF token / pyannote missing | single speaker, ingestion continues |
| Vision API down | that frame/page is uncaptioned; logged; the rest proceeds |
| Whisper can't load in the server | text chat keeps working; voice gets an error event |
| TTS provider missing | falls back to text-only answers |
| Embedding model changed | index refuses to open → `index.py --rebuild` |
| Client disconnects mid-answer | turn cancelled, sends swallowed |
| Path traversal / unsupported upload | 404 / 400 (tested) |
