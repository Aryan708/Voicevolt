# Resume bullets and interview prep

## Resume entry

**VoiceVault: Voice-first Multimodal RAG Copilot** · Python, PyTorch, Hugging Face (Whisper, PEFT), CLIP, Qdrant, FastAPI, Claude/GPT/Llama
· [GitHub](https://github.com/YOUR_USERNAME/voicevault) · [Demo](https://youtu.be/...)

- Built a voice assistant that answers spoken English/Hindi questions over meeting videos, slide decks and scanned
  documents, citing exact timestamps and pages; streamed LLM output to sentence-level TTS with barge-in, reaching
  **~X.X s** voice-to-first-audio (p50).
- Designed hybrid multimodal retrieval (multilingual-e5 + BM25 + CLIP, Reciprocal Rank Fusion) over Qdrant; ablation
  on a **N**-question bilingual eval set raised MRR from **0.XX** (best single retriever) to **0.XX**.
- Fine-tuned Whisper-small with **LoRA** on Hindi + self-recorded Hinglish speech; cut WER on held-out speakers from
  **XX%** to **XX%**, exported to CTranslate2 for real-time CPU inference.
- Wrote a slide-change keyframe detector for text-only slide transitions, which ffmpeg scene detection misses
  (4/4 slides found vs 1/4 on the test meeting; re-measure on your own recordings), plus vision-LLM captioning so
  charts and scans are searchable.
- Added LLM-as-judge evaluation (faithfulness **0.XX**, citation precision **0.XX**) and a feedback loop exporting
  DPO preference pairs; shipped with Docker Compose and CI (68 offline tests).

**Only put numbers you measured.** Run `finetune/evaluate_asr.py` and `scripts/evaluate.py all`, then fill in the Xs.
If a number is unimpressive, keep it and explain it. Interviewers trust honest numbers.

## JD → project mapping (PwC Gen AI Associate)

| Requirement | Where you did it |
|---|---|
| Gen AI apps with LLMs (GPT, Claude, Llama) and multimodal models | `llm.py` (3 providers), vision captions/OCR, CLIP |
| Prompt engineering | grounded-answer + voice prompts, follow-up rewriting, OCR prompt, judge prompts (`rag.py`, `vision.py`, `rag_eval.py`) |
| Fine-tune models | Whisper LoRA (`finetune/`) |
| Pipelines for text, image, audio, video | `ingest/` covers all four, plus TTS generation |
| Integrate via OpenAI / Anthropic / Hugging Face APIs | all three, behind one interface |
| Evaluation, RLHF, embeddings, RAG | `eval/` + feedback→DPO, e5/CLIP embeddings, hybrid RAG |
| Python, PyTorch, Transformers | throughout; training uses HF Trainer + PEFT |
| Vector DBs, RAG frameworks | Qdrant (FAISS-equivalent), RAG pipeline built from scratch (explain why below) |
| Chatbots, copilots, assistants | the whole product |

## Questions to rehearse

**"Why not just use LangChain?"** I wanted to control and measure each stage: three retrievers fused with RRF,
proof-image selection, streaming into sentence-level TTS with cancellation. Those don't map cleanly onto a chain
abstraction. I know the LangChain/LlamaIndex equivalents (EnsembleRetriever, MultiVector retrievers) and could port it.

**"Why RRF instead of a weighted score sum?"** Cosine scores from e5, BM25 scores and CLIP scores live on different
scales and distributions. RRF only uses ranks, so no calibration is needed and one retriever's score scale can't
dominate. The ablation shows it beats each single retriever.

**"How do you know answers are grounded?"** The prompt forbids unsupported claims and requires citations, and I measure
it: an LLM judge splits answers into claims and checks each against the retrieved sources (faithfulness). Citation
precision checks that cited chunks are actually relevant. There are also unanswerable questions to test refusal.

**"How does barge-in work?"** The server runs VAD on the incoming mic stream. Speech-start while a turn is running cancels
the asyncio task and sets a flag that the LLM-streaming thread checks; the client drops queued audio. Echo cancellation
in `getUserMedia` stops the assistant's own voice from triggering it.

**"What was hardest?"** Slide detection. ffmpeg's scene score barely moves when only the text on a slide changes.
I measured 0.05-0.26 against a 0.3 threshold, so every slide change was missed. Lowering the threshold floods webcam
footage. The fix uses the fact that slides are static: keep a frame when it differs from the last *kept* frame and has
then stopped changing.

**"Why LoRA? What would you change with more data?"** LoRA trains about 1-2% of weights, so it fits a free T4, the adapter
is small, and it forgets less English. With more data: whisper-medium, more speakers for a cleaner speaker-disjoint
test, a consistent Hinglish spelling convention, and data augmentation (noise, speed).

**"What is the RLHF part, honestly?"** I collect thumbs up/down in the app and export SFT examples and DPO preference
pairs. I haven't trained a preference-tuned model; that would be the next step, using TRL's DPOTrainer on an open model.
The immediate value is reading the 👎 reasons to fix prompts and retrieval.

**"Latency breakdown?"** Endpointing silence (700 ms) is the biggest fixed cost, then STT, then LLM first token.
Sentence streaming hides most of the generation time. Next steps: streaming STT and a shorter silence window with a
smarter end-of-turn model.

**"How would you productionise this for a client?"** Auth and per-client tenancy (Qdrant collections or payload filters
per tenant), a job queue (Celery/RQ) instead of threads for ingestion, Qdrant server, PII redaction before storage,
data residency for recordings, audit logs of what was retrieved, and monitoring on faithfulness and latency.
