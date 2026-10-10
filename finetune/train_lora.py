"""LoRA fine-tuning of Whisper on a manifest from prepare_data.py.

Fits a free Colab T4 (16 GB) for whisper-small in fp16; use --8bit for whisper-medium.

    python finetune/train_lora.py --manifest data/asr/manifest.jsonl \
        --base openai/whisper-small --language hindi --epochs 3 --out runs/whisper-small-hinglish

Only ~1-2% of parameters are trained (attention projections), so it trains fast,
the adapter is ~20-60 MB, and the base model's English ability is largely kept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.manifest import read_manifest, summary  # noqa: E402
from voicevault.eval.asr_metrics import cer, wer  # noqa: E402

SR = 16000
MAX_LABEL_TOKENS = 448  # Whisper decoder limit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--base", default="openai/whisper-small")
    p.add_argument("--language", default="hindi", help="Whisper language name or code")
    p.add_argument("--out", type=Path, default=Path("runs/whisper-lora"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--targets", default="q_proj,k_proj,v_proj,out_proj")
    p.add_argument("--8bit", dest="eightbit", action="store_true", help="load base in 8-bit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="tensorboard")
    return p.parse_args()


@dataclass
class SpeechCollator:
    """Pads log-mel features and labels; masks label padding with -100 so it is ignored."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # the model prepends the start token itself during training
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    a = parse_args()
    import torch
    from datasets import Audio, Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (BitsAndBytesConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor, set_seed)

    set_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    train = read_manifest(a.manifest, "train")
    val = read_manifest(a.manifest, "val")
    if not train or not val:
        raise SystemExit("manifest needs train and val splits (run prepare_data.py)")
    print("Data:", json.dumps(summary(train + val)))

    processor = WhisperProcessor.from_pretrained(a.base, language=a.language, task="transcribe")
    quant = BitsAndBytesConfig(load_in_8bit=True) if a.eightbit else None
    model = WhisperForConditionalGeneration.from_pretrained(
        a.base, quantization_config=quant, device_map="auto" if a.eightbit else None)
    model.generation_config.language = a.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if a.eightbit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
        target_modules=a.targets.split(",")))
    model.print_trainable_parameters()

    def to_ds(recs):
        ds = Dataset.from_list([{"audio": r["audio"], "text": r["text"]} for r in recs])
        ds = ds.cast_column("audio", Audio(sampling_rate=SR))

        def prep(ex):
            au = ex["audio"]
            ex["input_features"] = processor.feature_extractor(
                au["array"], sampling_rate=au["sampling_rate"]).input_features[0]
            ex["labels"] = processor.tokenizer(ex["text"]).input_ids
            return ex

        ds = ds.map(prep, remove_columns=["audio", "text"], num_proc=1)
        return ds.filter(lambda x: len(x["labels"]) <= MAX_LABEL_TOKENS)

    train_ds, val_ds = to_ds(train), to_ds(val)
    tok = processor.tokenizer

    def compute_metrics(pred):
        ids, labels = pred.predictions, pred.label_ids
        labels[labels == -100] = tok.pad_token_id
        hyps = tok.batch_decode(ids, skip_special_tokens=True)
        refs = tok.batch_decode(labels, skip_special_tokens=True)
        return {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}

    cuda = torch.cuda.is_available()
    args = Seq2SeqTrainingArguments(
        output_dir=str(a.out / "checkpoints"),
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=max(1, a.batch // 2),
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, warmup_steps=a.warmup,
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        eval_strategy="steps", eval_steps=a.eval_steps, save_steps=a.eval_steps,
        logging_steps=25, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        fp16=cuda and not a.eightbit, predict_with_generate=True, generation_max_length=225,
        remove_unused_columns=False, label_names=["labels"], report_to=a.report_to,
        dataloader_num_workers=2, seed=a.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=SpeechCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics, processing_class=processor.feature_extractor,
    )

    t0 = time.time()
    trainer.train()
    final = trainer.evaluate()
    adapter_dir = a.out / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    (a.out / "train_summary.json").write_text(json.dumps({
        "base": a.base, "language": a.language, "train_clips": len(train_ds),
        "val_clips": len(val_ds), "minutes": round((time.time() - t0) / 60, 1),
        "val_wer": final.get("eval_wer"), "val_cer": final.get("eval_cer"),
        "lora": {"r": a.lora_r, "alpha": a.lora_alpha, "targets": a.targets},
        "hparams": {"lr": a.lr, "epochs": a.epochs, "batch": a.batch * a.grad_accum},
    }, indent=2))
    print(f"\n✓ adapter saved to {adapter_dir}  (val WER {final.get('eval_wer'):.3f})")
    print(f"Next: python finetune/export_model.py --base {a.base} --adapter {adapter_dir}")


if __name__ == "__main__":
    main()
