# Demo video script (2.5 minutes)

Record with OBS or the Windows Game Bar at 1080p and show the browser full-screen. Use the real setup
(`.env` with an API key, `python scripts/make_demo_data.py --index`), not the offline demo.

| Time | Show | Say |
|---|---|---|
| 0:00-0:15 | README header | "Consultants drown in call recordings, decks and scanned documents. VoiceVault lets you just *ask* them, out loud, in English or Hindi." |
| 0:15-0:30 | Drag `Q3_review_meeting.mp4`, the PDF and the scan into the library; jobs tick to done | "It transcribes, labels speakers, detects every slide change, reads the slides and scans with a vision model, and indexes it all." |
| 0:30-0:55 | Click 🎙, ask: **"Why did the Chennai launch slip?"** | Let it answer out loud. Point at the `[1]` citation. |
| 0:55-1:10 | Click the source card → video opens at the moment | "Every answer is cited. This jumps to the exact moment in the meeting, with the slide that was on screen." |
| 1:10-1:25 | Ask: **"How much do we owe the vendor?"** | Answer comes from the *scanned invoice*. "There's no text in that file. It was found by image search and read by vision OCR." |
| 1:25-1:40 | Ask a long question, then interrupt mid-answer: **"Actually, just tell me the hiring budget"** | "You can interrupt it, like a person." |
| 1:40-1:55 | Ask in Hindi: **"चेन्नई लॉन्च में देरी क्यों हुई?"** | It answers in Hindi. "Hindi and Hinglish work end to end. I fine-tuned Whisper with LoRA for Hinglish speech." |
| 1:55-2:15 | `reports/asr/results.md` and `reports/eval/report.md` | "And it's measured: WER before and after fine-tuning, a retrieval ablation showing hybrid beats each retriever, faithfulness and latency." |
| 2:15-2:30 | Latency strip under an answer; GitHub Actions green tick | "First audio in about a second and a half. Dockerised, CI-tested. Links below." |

Tips: do two takes and keep the better one. If the answer is long, set `LLM_MAX_TOKENS=250` for the
recording. Mute notifications.
