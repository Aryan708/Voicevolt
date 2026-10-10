"""Merge the LoRA adapter into Whisper and convert to CTranslate2 for faster-whisper.

    python finetune/export_model.py --base openai/whisper-small \
        --adapter runs/whisper-small-hinglish/adapter --out models/whisper-small-hinglish

Produces:
    <out>/hf    merged Hugging Face model (for the Hub / transformers)
    <out>/ct2   CTranslate2 model -> use it with  STT_MODEL=<out>/ct2  (or --model in scripts)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--quantization", default="int8_float16",
                   help="int8 (CPU) | int8_float16 | float16 (GPU)")
    p.add_argument("--push", help="optional Hugging Face repo id, e.g. you/whisper-small-hinglish")
    a = p.parse_args()

    import torch
    from ctranslate2.converters import TransformersConverter
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    hf_dir, ct2_dir = a.out / "hf", a.out / "ct2"
    base = WhisperForConditionalGeneration.from_pretrained(a.base, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(a.adapter)).merge_and_unload()
    processor = WhisperProcessor.from_pretrained(a.adapter)
    model.save_pretrained(hf_dir, safe_serialization=True)
    processor.save_pretrained(hf_dir)
    print(f"✓ merged model -> {hf_dir}")

    TransformersConverter(str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]
                          ).convert(str(ct2_dir), quantization=a.quantization, force=True)
    print(f"✓ faster-whisper model -> {ct2_dir}")
    print(f"  use it:  STT_MODEL={ct2_dir}   or   python scripts/ingest.py x.mp4 --model {ct2_dir}")

    if a.push:
        card = Path(__file__).parent / "MODEL_CARD.md"
        model.push_to_hub(a.push)
        processor.push_to_hub(a.push)
        if card.exists():
            from huggingface_hub import upload_file
            upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=a.push)
        print(f"✓ pushed to https://huggingface.co/{a.push}")


if __name__ == "__main__":
    main()
