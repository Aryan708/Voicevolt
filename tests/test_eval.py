"""Week 5 tests: retrieval metrics + ablation, LLM-judge RAG eval, feedback -> SFT/DPO, latency."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import corpus
from voicevault.eval.feedback import build_preference_data, latency_report, percentile
from voicevault.eval.rag_eval import evaluate_rag, parse_json
from voicevault.eval.retrieval_eval import evaluate_retrieval, matches, retrieval_table, score_ranking
from voicevault.llm import FakeLLM
from voicevault.rag import RAGPipeline
from voicevault.schemas import Chunk

ROOT = Path(__file__).resolve().parent.parent

EVAL = [
    {"id": "q1", "question": "Why did the Chennai launch slip?", "language": "en",
     "relevant": [{"source": "sync.mp4", "start": 755, "end": 780}], "answer": "Vendor delays."},
    {"id": "q2", "question": "travel reimbursement limit rupees", "language": "en",
     "relevant": [{"source": "policy.pdf", "page": 3}]},
    {"id": "q3", "question": "scanned invoice 4471", "language": "en",
     "relevant": [{"source": "scan.pdf", "page": 1}]},
    {"id": "u1", "question": "What is the CEO's salary?", "unanswerable": True, "relevant": []},
]


def ch(**kw):
    return Chunk(id=kw.pop("id", "x"), source=kw.pop("source", "a.mp4"),
                 modality=kw.pop("modality", "video"), text="t", **kw)


# ------------------------------------------------------- retrieval metrics

def test_matches_page_time_and_source_only():
    assert matches(ch(source="d.pdf", modality="document", page=3), {"source": "d.pdf", "page": 3})
    assert not matches(ch(source="d.pdf", modality="document", page=2), {"source": "d.pdf", "page": 3})
    assert matches(ch(start=100, end=145), {"source": "a.mp4", "start": 140, "end": 200})
    assert matches(ch(start=100, end=145), {"source": "a.mp4", "start": 148, "end": 200})  # 5 s tolerance
    assert not matches(ch(start=100, end=145), {"source": "a.mp4", "start": 160, "end": 200})
    assert matches(ch(), {"source": "a.mp4"})
    assert not matches(ch(), {"source": "b.mp4"})


def test_score_ranking():
    ranked = [ch(id="1", source="x.mp4"), ch(id="2", source="a.mp4"), ch(id="3", source="b.mp4")]
    s = score_ranking(ranked, [{"source": "a.mp4"}, {"source": "b.mp4"}], k=3)
    assert s["hit"] == 1 and s["recall"] == 1 and s["mrr"] == pytest.approx(0.5)
    assert 0 < s["ndcg"] < 1
    assert score_ranking(ranked, [{"source": "zzz"}], k=3) == {"hit": 0, "recall": 0, "mrr": 0, "ndcg": 0}


def test_ablation_shows_hybrid_beats_single_retrievers(kb):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    res = evaluate_retrieval(retriever, EVAL, k=3)
    assert res["questions"] == 3                       # unanswerable excluded
    o = {n: c["overall"] for n, c in res["configs"].items()}
    assert o["hybrid (all)"]["hit"] == 1.0
    assert o["bm25"]["hit"] < 1.0                      # can't find the text-less scan
    assert o["hybrid (all)"]["mrr"] >= max(o["dense"]["mrr"], o["bm25"]["mrr"], o["visual"]["mrr"])
    table = retrieval_table(res)
    assert "| **hybrid (all)** |" in table and "MRR" in table


# ------------------------------------------------------------- RAG eval

def judge_responder(system, messages):
    p = messages[-1]["content"]
    if "atomic factual claims" in p:
        claims = [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]
        return "```json\n" + json.dumps({"claims": claims}) + "\n```"
    if "Reference answer" in p:
        return '{"score": 3}'
    if "CANNOT be answered" in p:
        return 'Sure! {"declined": true}'
    return '{"score": 5}'


def test_rag_eval_metrics(kb):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    rag = RAGPipeline(retriever, FakeLLM(), k=3)
    res = evaluate_rag(rag, FakeLLM(judge_responder), EVAL)
    s = res["summary"]
    assert s["faithfulness"]["mean"] == 0.5
    assert s["relevance"]["mean"] == 1.0
    assert s["correctness"] == {"mean": 0.5, "n": 1}
    assert s["refusal_ok"] == {"mean": 1.0, "n": 1}
    q1 = next(r for r in res["rows"] if r["id"] == "q1")
    assert q1["citation_precision"] == 0.5             # cites [1] (relevant) and [2] (not)
    assert q1["unsupported_claims"] == ["b"]


def test_judge_failure_is_tolerated(kb):
    store, indexer, retriever, tmp = kb
    indexer.index(corpus(tmp)[0])
    rag = RAGPipeline(retriever, FakeLLM(), k=3)
    res = evaluate_rag(rag, FakeLLM(lambda s, m: "I refuse to output JSON"), EVAL[:1])
    assert res["rows"][0]["faithfulness"] is None and "faithfulness" not in res["summary"]


def test_parse_json():
    assert parse_json('text ```json\n{"a": [1, {"b": 2}]}\n``` more') == {"a": [1, {"b": 2}]}
    with pytest.raises(ValueError):
        parse_json("no json here")


# --------------------------------------------------------- feedback / RLHF

def test_feedback_to_sft_and_dpo():
    fb = [
        {"question": "When is the launch?", "answer": "Q4, due to vendor delays [1].", "rating": 1},
        {"question": "when is the launch", "answer": "In March.", "rating": -1, "comment": "Wrong date"},
        {"question": "When is the launch?", "answer": "Q4, due to vendor delays [1].", "rating": 1},  # dup
        {"question": "Hiring plan?", "answer": "Two ML engineers [2].", "rating": 1},
        {"question": "Budget?", "answer": "No idea", "rating": -1, "comment": "wrong date"},
    ]
    d = build_preference_data(fb)
    assert d["dpo"] == [{"prompt": "When is the launch?", "chosen": "Q4, due to vendor delays [1].",
                         "rejected": "In March."}]
    assert len(d["sft"]) == 2
    assert d["issues"][0] == ("wrong date", 2)
    assert d["stats"]["thumbs_up_rate"] == 0.6


def test_latency_report_percentiles_and_generator_input():
    assert percentile([100, 200, 300, 400], 50) == 250
    logs = [{"mode": "voice", "total_ms": 1000 + i, "first_audio_ms": 700 + i, "stt_ms": 200}
            for i in range(10)] + [{"mode": "text", "total_ms": 5}, {"mode": "voice", "interrupted_ms": 300}]
    rep = latency_report(x for x in logs)
    assert rep["turns"] == 10 and rep["interrupted"] == 1
    assert rep["stages"]["first_audio_ms"]["p50"] == pytest.approx(704.5)


# -------------------------------------------------- CLI end-to-end (offline)

def test_evaluate_cli_end_to_end(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "policy.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 750, "Travel reimbursement is capped at 5000 rupees per trip for all staff.")
    c.save()
    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(json.dumps({"id": "q1", "question": "travel reimbursement cap rupees",
                                    "relevant": [{"source": "policy.pdf", "page": 1}]}) + "\n")
    env = {**os.environ, "EMBED_BACKEND": "hash", "LLM_PROVIDER": "fake", "VISION_PROVIDER": "none",
           "INDEX_DIR": str(tmp_path / "idx"), "REPORTS_DIR": str(tmp_path / "reports")}
    run = lambda *args: subprocess.run([sys.executable, *args], cwd=ROOT, env=env,  # noqa: E731
                                       capture_output=True, text=True, timeout=180)
    r = run("scripts/index.py", str(pdf))
    assert r.returncode == 0, r.stderr
    r = run("scripts/evaluate.py", "retrieval", "--eval-set", str(eval_set), "-k", "3")
    assert r.returncode == 0, r.stderr
    assert "| **hybrid (all)** | 1.000 |" in r.stdout
    assert (tmp_path / "reports" / "eval" / "retrieval.json").exists()
    assert not (ROOT / "reports" / "eval" / "retrieval.json").exists(), "test must not write into repo"
