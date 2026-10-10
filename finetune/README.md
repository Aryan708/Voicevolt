# Week 4 · Whisper LoRA fine-tuning for Hindi / Hinglish

Off-the-shelf Whisper struggles with Indian code-switched speech ("kal ki meeting 3 baje shift ho gayi").
This folder fine-tunes it with LoRA and proves the gain with a held-out, speaker-disjoint test set.

```
prepare_data.py ─▶ manifest.jsonl ─▶ train_lora.py ─▶ adapter ─▶ export_model.py ─▶ ct2 model
 (FLEURS + your clips,  (train/val/test,     (PEFT LoRA,          (merge + CTranslate2,   │
  filter + dedupe)       speaker-disjoint)    best by val WER)     int8 for CPU)          ▼
                                                              evaluate_asr.py: before/after table
                                                              STT_MODEL=... : used live in VoiceVault
```

**Easiest path:** open `whisper_lora_colab.ipynb` in Google Colab with a free T4 GPU and run all cells.

## Getting Hinglish data
Public Hinglish speech corpora are scarce, so the most effective thing you can do is **record 30-60 minutes
yourself** (and ideally a few friends, which enables a speaker-disjoint test set):

```
my_clips/
  metadata.csv        # file,text,speaker
  0001.wav            # "Kal ki meeting teen baje shift ho gayi hai"
  0002.wav            # "Q3 ka revenue twelve percent badha"
```
Write sentences like ones you'd actually ask VoiceVault: meeting talk, numbers, names, English tech words.
Pick **one spelling convention** for Hinglish (romanised or Devanagari) and stick to it. Inconsistent
references inflate WER more than anything the model does.

Combine with public Hindi for robustness: `--local my_clips --hf google/fleurs --hf-config hi_in`.

## Commands
```bash
pip install -r finetune/requirements-finetune.txt
python finetune/prepare_data.py --local my_clips --hf google/fleurs --hf-config hi_in \
    --text-col transcription --speaker-col id --trust-remote-code --out data/asr
python finetune/train_lora.py --manifest data/asr/manifest.jsonl --base openai/whisper-small \
    --language hindi --epochs 3 --out runs/whisper-small-hinglish
python finetune/export_model.py --base openai/whisper-small \
    --adapter runs/whisper-small-hinglish/adapter --out models/whisper-small-hinglish
python finetune/evaluate_asr.py --manifest data/asr/manifest.jsonl --language hi \
    --model small --label whisper-small \
    --model models/whisper-small-hinglish/ct2 --label whisper-small-hinglish-lora
```
Then set `STT_MODEL=models/whisper-small-hinglish/ct2` in `.env`.

## Design choices
| Choice | Why |
|---|---|
| LoRA on attention projections (r=32) | ~1-2% of weights trained: fits a T4, small adapter, keeps Whisper's English |
| Speaker-disjoint test split | Otherwise the model memorises voices and WER looks better than it is |
| Clip filters (0.5-30 s, chars/sec sanity, dedupe) | Misaligned transcripts are the #1 cause of fine-tuning making things worse |
| Hindi-aware normaliser | Danda vs full stop, ३ vs 3, nukta, ZWJ are not real errors |
| Corpus-level WER + breakdown by script | Standard reporting, and shows *where* the gain is (Hindi vs Hinglish vs English) |
| Same decoder (faster-whisper) for before and after | Fair comparison: only the weights differ |
| Best checkpoint by val WER | Guards against over-fitting small datasets |

## Honest caveats to mention in interviews
- With 1 speaker (just you), the test set measures *your* voice. Say so, or recruit 3-5 friends.
- If references mix scripts, part of the "WER" is spelling disagreement, not recognition errors.
  Check `reports/asr/utterances.csv`.
