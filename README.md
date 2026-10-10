# 🎙️ VoiceVault

**A voice-first multimodal knowledge copilot.** Upload meeting recordings, lecture videos, slide decks and scanned documents, ask questions by voice in Hindi or English, and get a spoken answer with the exact video timestamp and slide as proof.

> Status: **Week 2 complete** — audio/video/document ingestion + hybrid multimodal retrieval.

## Architecture

```
            ┌──────────────┐
 media ───▶ │  ffmpeg      │──▶ 16 kHz mono WAV ──┬──▶ faster-whisper ──▶ words + timestamps ─┐
 file       │  (media.py)  │                      └──▶ pyannote ─────────▶ speaker turns ─────┤
            └──────┬───────┘                                                                  ▼
                   │ video?                                               word-level speaker alignment
                   ▼                                                                          │
         scene-change keyframes ──▶ vision LLM captions (Claude / GPT) ──────────────────────┤
           (video.py)                 (vision.py)                                             ▼
                                                                         time-window chunker (overlap)
                                                                                              │
                                                                                              ▼
                                                                  data/processed/<file>/ingest.json
                                                                  (→ vector store in Week 2)
```

### Example chunk (what gets embedded)
```
[12:30 SPEAKER_01] So if we look at the Q3 numbers...
[12:34 SLIDE] Slide titled "Q3 Revenue by Region". Bar chart: North 4.2 Cr, South 3.1 Cr...
[12:41 SPEAKER_00] South is lower because the Chennai launch slipped.
```
Speech and slides are interleaved in time order, so the copilot can answer *"what did Rahul say about the South region?"* and cite **12:34** with the slide image.

## Week 2: multimodal retrieval

```
                         ┌── dense  : multilingual-e5 query  ─▶ chunk text vectors ──┐
 "why did the Chennai    │                                                           │   weighted
  launch slip?"   ───────┼── bm25   : keywords, numbers, IDs ─▶ chunk text ──────────┼─▶ Reciprocal ─▶ top-k chunks
  (English or Hindi)     │                                                           │   Rank Fusion   + citation
                         └── visual : CLIP text encoder ─────▶ page + keyframe images┘                 + proof image
```

| Source | How text is obtained | What gets a CLIP vector |
|---|---|---|
| Meeting / lecture video | Whisper transcript + slide captions (Week 1) | Each informative keyframe |
| Audio | Whisper transcript | — |
| PDF with text | PDF text layer (exact, free) | Each page render |
| Scan / photo / image-only slide | Vision LLM OCR + chart description | Each page render |
| PPTX / DOCX | LibreOffice → PDF → as above | Each slide/page render |

```
$ python scripts/search.py "why did the Chennai launch slip"
  1. report.pdf p.2   [document]  score=0.0390  (bm25#1 dense#1 visual#3)
     🖼  data/processed/report/pages/page_0002.png
     [report.pdf p.2] Risks: Chennai launch slipped to Q4 because of vendor delays.
```

## Design choices
| Decision | Why |
|---|---|
| faster-whisper over openai-whisper | ~4x faster on CPU with int8, same accuracy |
| VAD filter on | Removes hallucinated text in silent stretches |
| Word-level speaker alignment | Whisper segments often span two speakers; splitting at word level fixes mislabels |
| Scene detection + forced max gap | Catches every slide change, and still samples talking-head videos |
| `TALKING_HEAD` caption sentinel | Frames with no visual info are not embedded, saving tokens and noise |
| Chunks never cut a sentence + overlap | Answers near a chunk boundary are not lost |
| Pluggable Whisper model path | Week 4's fine-tuned Hinglish model drops in with `--model` |
| Three retrievers, RRF fusion | Dense handles paraphrase/Hindi, BM25 handles names/IDs/numbers, CLIP handles visuals; RRF fuses ranks so scores never need calibrating |
| Multilingual CLIP text tower | A Hindi query can retrieve an English slide image |
| Separate `chunks` / `images` collections | One video chunk with 3 slides gets 3 visual entry points, and the matching image is shown as proof |
| PDF text layer first, vision OCR only when empty | Exact text for free on normal PDFs; LLM cost only on scans |
| Index metadata check | Changing embedding model without `--rebuild` fails loudly instead of returning garbage |
| `EMBED_BACKEND=hash` | Whole system and test-suite run offline in CI |

## Setup
```bash
# 1. ffmpeg
#    Windows: winget install ffmpeg | macOS: brew install ffmpeg | Ubuntu: sudo apt install ffmpeg

# 2. Python env (3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Config
cp .env.example .env               # then fill in keys
```
- **Diarization:** create a Hugging Face token, accept the terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`, put it in `HF_TOKEN`. Without it everything still works, with a single speaker.
- **Frame captions:** set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, or `VISION_PROVIDER=none`.

## Usage
```bash
# Week 2: build the knowledge base and search it
python scripts/index.py data/raw/ -v                   # any mix: mp4, m4a, pdf, pptx, jpg...
python scripts/index.py --processed                    # index files already ingested
python scripts/index.py --stats
python scripts/search.py "what did we decide about hiring?"
python scripts/search.py "bar chart of regional revenue" --modality video document
python scripts/search.py "क्षेत्रीय राजस्व" --only visual   # ablation: one retriever
python scripts/search.py                                # interactive

# Week 1: ingestion only
python scripts/ingest.py data/raw/meeting.mp4 -v
python scripts/ingest.py data/raw/ --speakers 3          # a whole folder
python scripts/ingest.py lecture.mp4 --language hi       # force Hindi
python scripts/ingest.py call.m4a --no-diarize --model medium
```
Output: `data/processed/<file>/ingest.json` (transcript, keyframes, chunks, stage timings) plus `frames/*.jpg`.

## Tests
```bash
pytest -q
```
20 tests. Covers hybrid retrieval (keyword question → right timestamp, scan found only through its image, chart query returns the matching keyframe, filters, re-index without duplicates, BM25 refresh, embedder-change detection), document ingestion (text layer, scans, long-page splitting), chunking, overlap, frame/speech interleaving, visual-only videos, mid-sentence speaker changes, scene-change keyframes and the full pipeline wiring (Whisper and vision mocked, so tests need no GPU, keys or downloads).

## Roadmap
- [x] **Week 1** — audio/video ingestion, diarization, keyframes, captions, chunking
- [x] **Week 2** — document/image ingestion, CLIP + multilingual embeddings, Qdrant store, hybrid RRF retrieval
- [ ] **Week 3** — real-time voice loop: streaming STT → RAG → LLM → TTS, barge-in
- [ ] **Week 4** — Whisper LoRA fine-tune for Hinglish, WER before/after
- [ ] **Week 5** — evaluation (RAGAS, WER, latency) + Docker
- [ ] **Week 6** — demo video, HF Hub model card

## Project layout
```
voicevault/
├── voicevault/
│   ├── config.py          settings from .env
│   ├── schemas.py         pydantic models (Transcript, Keyframe, Chunk, ...)
│   ├── embed.py           multilingual text + CLIP embedders (+ offline hash backend)
│   ├── store.py           Qdrant: chunks + images collections
│   ├── retrieval.py       Indexer, BM25, HybridRetriever, RRF
│   └── ingest/
│       ├── media.py       ffmpeg helpers, modality detection
│       ├── transcribe.py  faster-whisper
│       ├── diarize.py     pyannote + word-level speaker alignment
│       ├── video.py       scene-change keyframes
│       ├── vision.py      frame captions + page OCR (Claude / OpenAI)
│       ├── documents.py   PDF / PPTX / DOCX / scans / images → page chunks
│       ├── chunker.py     time-window chunks
│       └── pipeline.py    orchestration
├── scripts/
│   ├── ingest.py          Week 1 ingestion CLI
│   ├── index.py           ingest + index anything
│   └── search.py          hybrid search CLI
├── tests/
└── data/{raw,processed}/
```
