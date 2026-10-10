---
language:
- hi
- en
license: apache-2.0
base_model: openai/whisper-small
tags:
- whisper
- automatic-speech-recognition
- lora
- peft
- hinglish
- code-switching
datasets:
- google/fleurs
metrics:
- wer
- cer
---

# whisper-small-hinglish (LoRA)

`openai/whisper-small` fine-tuned with LoRA for **Hindi and Hindi-English code-switched (Hinglish)** speech,
built as the speech-recognition component of [VoiceVault](https://github.com/Aryan708/Voicevolt), a
voice-first multimodal RAG copilot.

## Results

Test split speakers never appear in training (speaker-disjoint split).
WER uses a Hindi-aware normaliser (danda/full stop, Devanagari/ASCII digits, nukta and zero-width folding).

<!-- paste reports/asr/results.md table here -->
| Model | WER | CER | Δ WER vs base | WER hindi | WER hinglish |
|---|---|---|---|---|---|
| whisper-small | _fill in_ | | — | | |
| **whisper-small-hinglish-lora** | _fill in_ | | | | |

## Training

| | |
|---|---|
| Method | LoRA (PEFT) on `q_proj, k_proj, v_proj, out_proj`, r=32, alpha=64, dropout 0.05 |
| Trainable params | ~1-2% of the model |
| Data | FLEURS `hi_in` + self-recorded Hinglish clips (_N_ hours, _M_ speakers) |
| Filtering | 0.5-30 s clips, transcript/duration sanity check, de-duplication |
| Optimiser | AdamW, lr 5e-4, 50 warm-up steps, 3 epochs, batch 16, fp16 |
| Hardware | 1× T4 16 GB, _X_ minutes |
| Model selection | best checkpoint by validation WER |

## Usage

```python
# faster-whisper (CTranslate2 export)
from faster_whisper import WhisperModel
model = WhisperModel("path/to/ct2", compute_type="int8")
segments, info = model.transcribe("meeting.wav", language="hi")

# transformers
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="YOUR_USERNAME/whisper-small-hinglish")
asr("clip.wav", generate_kwargs={"language": "hindi"})
```

## Limitations

- Hinglish is written inconsistently (Devanagari vs romanised); the model follows the training transcripts'
  convention, which can be counted as "errors" against references using the other script.
- Small self-recorded data covers few speakers and accents; expect higher WER on unseen accents and noisy audio.
- Inherits Whisper's known failure modes, e.g. hallucinated text on long silences (use VAD).
