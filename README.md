# 🎙️ VoiceVault

**A voice-first multimodal knowledge copilot.** Upload meeting recordings, lecture videos, slide decks and scanned documents, ask questions by voice in Hindi or English, and get a spoken answer with the exact video timestamp and slide as proof.

> Status: **Week 1 complete** — audio/video ingestion pipeline.

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
Covers chunking, overlap, frame/speech interleaving, visual-only videos, mid-sentence speaker changes, scene-change keyframes and the full pipeline wiring (Whisper and vision mocked, so tests need no GPU, keys or downloads).

## Roadmap
- [x] **Week 1** — audio/video ingestion, diarization, keyframes, captions, chunking
- [ ] **Week 2** — image ingestion (CLIP/ColPali) + multimodal vector store + hybrid retrieval
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
│   └── ingest/
│       ├── media.py       ffmpeg helpers, modality detection
│       ├── transcribe.py  faster-whisper
│       ├── diarize.py     pyannote + word-level speaker alignment
│       ├── video.py       scene-change keyframes
│       ├── vision.py      frame captioning (Claude / OpenAI)
│       ├── chunker.py     time-window chunks
│       └── pipeline.py    orchestration
├── scripts/ingest.py      CLI
├── tests/
└── data/{raw,processed}/
```
