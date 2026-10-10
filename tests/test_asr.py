"""Week 4 tests: Hindi-aware normalisation, WER/CER, manifests, report table."""
import warnings

import pytest

from finetune.evaluate_asr import comparison_table
from finetune.manifest import clean_records, read_manifest, split_by_speaker, write_manifest
from voicevault.eval.asr_metrics import align, cer, normalize, report, script_of, wer


# ------------------------------------------------------------- normalise

def test_normalize_hindi_punctuation_digits_nukta_zero_width():
    assert normalize("मीटिंग ३ बजे है।") == "मीटिंग 3 बजे है"
    assert normalize("ज़रूरी") == normalize("जरूरी")                  # nukta folded
    assert normalize("क्‍ष") == normalize("क्ष")                   # ZWJ removed
    assert normalize("Revenue: 4,200 Cr. — ok?") == "revenue 4200 cr ok"
    assert normalize("कल  की   meeting, cancel!") == "कल की meeting cancel"


def test_normalize_keeps_vowel_signs():
    # matras are combining marks; a naive \w-based strip would delete them
    assert normalize("मीटिंग") == "मीटिंग"


def test_script_detection():
    assert script_of("kal ki meeting") == "english"      # romanised Hinglish counts as Latin
    assert script_of("कल की मीटिंग") == "hindi"
    assert script_of("कल की meeting cancel") == "hinglish"


# ------------------------------------------------------------- WER / CER

def test_align_counts():
    a = align("the cat sat on the mat".split(), "the cat sit on mat now".split())
    assert (a.sub, a.dele, a.ins, a.hits) == (1, 1, 1, 4)
    assert a.pairs == [("sat", "sit")]


def test_wer_is_corpus_level_not_mean_of_utterances():
    refs = ["a b c d e f g h i j", "x"]
    hyps = ["a b c d e f g h i j", "y"]
    assert wer(refs, hyps) == pytest.approx(1 / 11)      # mean-of-utterances would say 0.5


def test_wer_ignores_formatting_differences_for_hindi():
    assert wer(["मीटिंग ३ बजे है।"], ["मीटिंग 3 बजे है"]) == 0.0
    assert wer(["मीटिंग ३ बजे है।"], ["मीटिंग 3 बजे है"], normalize_text=False) > 0


def test_cer_and_empty():
    assert cer(["abcd"], ["abxd"]) == pytest.approx(0.25)
    assert wer([], []) == 0.0


def test_report_breaks_down_by_script():
    refs = ["kal meeting hai", "कल मीटिंग है", "कल की meeting cancel"]
    hyps = ["kal meeting hai", "कल मीटिंग थी", "कल की meeting cancel"]
    r = report(refs, hyps)
    assert r["by_script"]["english"]["wer"] == 0
    assert r["by_script"]["hindi"]["wer"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["top_substitutions"][0] == (("है", "थी"), 1)


# --------------------------------------------------------------- manifest

def recs(n_spk=10, per=10):
    return [{"audio": f"clips/s{s}_{i}.wav", "text": f"utterance {s} {i} kal meeting",
             "duration": 3.0, "speaker": f"spk{s}"} for s in range(n_spk) for i in range(per)]


def test_clean_records_filters_bad_clips():
    bad = [{"audio": "a.wav", "text": "", "duration": 2},
           {"audio": "b.wav", "text": "hi", "duration": 45},
           {"audio": "c.wav", "text": "x" * 400, "duration": 2},
           {"audio": "d.wav", "text": "ok fine", "duration": 2, "speaker": "s"},
           {"audio": "e.wav", "text": "OK, fine!", "duration": 2.02, "speaker": "s"}]  # dup
    kept, dropped = clean_records(bad)
    assert [r["audio"] for r in kept] == ["d.wav"]
    assert dropped == {"empty_text": 1, "duration": 1, "text_too_long_for_audio": 1, "duplicate": 1}


def test_speaker_disjoint_split():
    out = split_by_speaker(recs(), val=0.1, test=0.2, seed=1)
    spk_split = {}
    for r in out:
        assert spk_split.setdefault(r["speaker"], r["split"]) == r["split"], "speaker leaked"
    splits = [r["split"] for r in out]
    assert splits.count("test") == 20 and splits.count("val") == 10
    assert out == split_by_speaker(recs(), val=0.1, test=0.2, seed=1)   # deterministic


def test_single_speaker_falls_back_with_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = split_by_speaker(recs(n_spk=1, per=50), val=0.1, test=0.1)
    assert any("splitting by utterance" in str(x.message) for x in w)
    assert [r["split"] for r in out].count("test") == 5


def test_manifest_roundtrip_resolves_relative_paths(tmp_path):
    path = tmp_path / "asr" / "manifest.jsonl"
    write_manifest(split_by_speaker(recs()), path)
    test = read_manifest(path, "test")
    assert test and all(r["audio"].startswith(str(tmp_path / "asr")) for r in test)


# ------------------------------------------------------------ report table

def test_comparison_table():
    base = {"label": "whisper-small", "wer": 0.40, "cer": 0.2, "rtf": 0.1,
            "by_script": {"hindi": {"wer": 0.35}, "hinglish": {"wer": 0.5}}}
    ft = {"label": "lora", "wer": 0.30, "cer": 0.15, "rtf": 0.1,
          "by_script": {"hindi": {"wer": 0.3}, "hinglish": {"wer": 0.32}}}
    t = comparison_table([base, ft])
    assert "| `lora` | 30.0% | 15.0% | -25.0% | 30.0% | 32.0% | 0.10 |" in t
    assert t.splitlines()[0].startswith("| Model | WER | CER")
