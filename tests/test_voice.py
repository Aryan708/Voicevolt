"""Week 3 tests: sentence streaming, VAD endpointing, RAG answers, voice session + barge-in."""
import asyncio

import numpy as np

from conftest import corpus
from voicevault.llm import FakeLLM
from voicevault.rag import RAGPipeline, is_hindi, parse_citations, strip_citations
from voicevault.voice.agent import VoiceSession
from voicevault.voice.sentences import SentenceStreamer, split_stream
from voicevault.voice.stt import FakeSTT
from voicevault.voice.tts import FakeTTS
from voicevault.voice.vad import SAMPLE_RATE, EnergyVAD, UtteranceSegmenter


# ------------------------------------------------------------ sentences

def test_sentence_streamer_handles_abbreviations_decimals_and_hindi():
    text = ("Dr. Rao said revenue grew 12.5 percent in Q3 [1]. The Chennai launch slipped to Q4 "
            "because of vendors [2]! चेन्नई लॉन्च चौथी तिमाही तक टल गया। Done")
    tokens = [text[i:i + 3] for i in range(0, len(text), 3)]  # simulate token stream
    out = list(split_stream(iter(tokens)))
    assert out == [
        "Dr. Rao said revenue grew 12.5 percent in Q3 [1].",
        "The Chennai launch slipped to Q4 because of vendors [2]!",
        "चेन्नई लॉन्च चौथी तिमाही तक टल गया।",
        "Done",
    ]


def test_sentence_streamer_emits_early_and_cuts_runaway_text():
    s = SentenceStreamer(min_chars=10, max_chars=60)
    assert s.feed("First sentence is here. Sec") == ["First sentence is here."]
    long = "word " * 40
    pieces = s.feed(long)
    assert pieces and all(len(p) <= 60 for p in pieces)


# ----------------------------------------------------------------- VAD

def tone(seconds, amp=0.3, freq=220):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def silence(seconds, noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, noise * 32767, int(seconds * SAMPLE_RATE))).astype(np.int16)


def feed_all(seg, audio, chunk=1600):
    events, b = [], audio.tobytes()
    for i in range(0, len(b), chunk):
        events += seg.feed(b[i:i + chunk])
    return events


def test_vad_segments_one_utterance_with_preroll():
    seg = UtteranceSegmenter(vad=EnergyVAD(), silence_ms=600)
    audio = np.concatenate([silence(1.0), tone(1.2), silence(1.0)])
    ev = feed_all(seg, audio)
    assert [e.type for e in ev] == ["start", "end"]
    assert 1.2 <= ev[1].duration <= 1.8   # utterance + pre-roll + short tail


def test_vad_ignores_clicks_and_finds_two_utterances():
    seg = UtteranceSegmenter(vad=EnergyVAD(), silence_ms=500, min_utterance_ms=300)
    audio = np.concatenate([silence(0.8), tone(0.1), silence(0.8),   # click
                            tone(0.8), silence(0.8), tone(0.6, freq=330), silence(0.8)])
    ends = [e for e in feed_all(seg, audio) if e.type == "end" and e.audio is not None]
    assert len(ends) == 2


# ----------------------------------------------------------------- RAG

def test_citation_helpers():
    assert parse_citations("A [2] and B [1][2], not [x]") == [1, 2]
    assert strip_citations("Launch slipped [1]. Hiring [2] [3].") == "Launch slipped. Hiring."
    assert is_hindi("लॉन्च कब है?") and not is_hindi("when is launch?")


def test_rag_answer_with_citations_and_followup(kb):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    llm = FakeLLM()
    rag = RAGPipeline(retriever, llm, k=3)

    ans = rag.ask("Why did the Chennai launch slip?")
    assert ans.cited == [1, 2]
    assert ans.hits[0].chunk.id == "m1"
    assert ans.sources()[0]["citation"] == "sync.mp4 @ 12:30-13:15"
    system, msgs = llm.calls[-1]
    assert "[1] sync.mp4 @ 12:30-13:15 (video)" in msgs[-1]["content"]
    assert set(ans.timings) >= {"retrieval", "first_token", "total"}

    # follow-up triggers query rewriting with the history
    hist = [{"role": "user", "content": "Why did the Chennai launch slip?"},
            {"role": "assistant", "content": ans.text}]
    rag.ask("and what about hiring?", history=hist)
    rewrite_calls = [c for c in llm.calls if "Rewrite the follow-up" in c[0]]
    assert rewrite_calls and "Chennai" in rewrite_calls[-1][1][0]["content"]


def test_rag_voice_mode_and_language_rule(kb):
    store, indexer, retriever, tmp = kb
    indexer.index(corpus(tmp)[0])
    llm = FakeLLM()
    RAGPipeline(retriever, llm).ask("hiring plan?", voice=True)
    system = llm.calls[-1][0]
    assert "spoken aloud" in system and "same language" in system


# ------------------------------------------------------------- session

class Collector:
    def __init__(self):
        self.events = []

    async def __call__(self, ev):
        self.events.append(ev)

    def types(self):
        return [e["type"] for e in self.events]


def make_session(kb, tts=None, stt=None, **kw):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    sink = Collector()
    rag = RAGPipeline(retriever, FakeLLM(), k=3)
    sess = VoiceSession(rag, stt or FakeSTT(), tts or FakeTTS(), sink,
                        segmenter=UtteranceSegmenter(vad=EnergyVAD(), silence_ms=400),
                        metrics_path=tmp / "metrics.jsonl", **kw)
    return sess, sink, tmp


def test_text_turn_streams_tokens_audio_and_done(kb):
    tts = FakeTTS()
    sess, sink, tmp = make_session(kb, tts=tts)

    async def run():
        await sess.ask_text("Why did the Chennai launch slip?")
        await sess.wait()
    asyncio.run(run())

    t = sink.types()
    assert t[0] == "transcript" and t[1] == "sources" and t[-1] == "done"
    assert "token" in t and "audio" in t
    assert t.index("audio") < t.index("done")
    done = sink.events[-1]
    assert done["cited"] == [1, 2] and done["sources"][0]["chunk_id"] == "m1"
    assert all("[" not in s for s in tts.spoken), "citation markers must not be spoken"
    assert len(sess.history) == 2
    assert (tmp / "metrics.jsonl").read_text().count("first_audio_ms") == 1


def test_voice_turn_from_microphone_audio(kb):
    stt = FakeSTT("what is the hiring plan")
    sess, sink, _ = make_session(kb, stt=stt)

    async def run():
        audio = np.concatenate([silence(0.5), tone(1.0), silence(0.8)])
        b = audio.tobytes()
        for i in range(0, len(b), 3200):
            await sess.feed_audio(b[i:i + 3200])
        await sess.wait()
    asyncio.run(run())

    t = sink.types()
    assert t[:2] == ["vad", "vad"] and stt.calls == 1
    assert sink.events[-1]["type"] == "done"
    assert "stt_ms" in sink.events[-1]["metrics"]


def test_barge_in_cancels_current_answer(kb):
    slow_tts = FakeTTS(delay=0.4)
    sess, sink, _ = make_session(kb, tts=slow_tts, barge_in=True)

    async def run():
        await sess.ask_text("Why did the Chennai launch slip?")
        await asyncio.sleep(0.2)              # agent is mid-answer
        assert sess.busy
        b = tone(0.5).tobytes()               # user starts talking over it
        for i in range(0, len(b), 3200):
            await sess.feed_audio(b[i:i + 3200])
        assert not sess.busy
    asyncio.run(run())

    t = sink.types()
    assert "interrupt" in t
    assert sink.events[t.index("interrupt")]["reason"] == "barge_in"
    assert "done" not in t, "interrupted answer must not complete"
    assert sess.history == []


def test_barge_in_disabled_keeps_talking(kb):
    sess, sink, _ = make_session(kb, tts=FakeTTS(delay=0.1), barge_in=False)

    async def run():
        await sess.ask_text("Why did the Chennai launch slip?")
        await asyncio.sleep(0.05)
        b = tone(0.3).tobytes()
        for i in range(0, len(b), 3200):
            await sess.feed_audio(b[i:i + 3200])
        await sess.wait()
    asyncio.run(run())
    assert "interrupt" not in sink.types() and sink.types()[-1] == "done"
