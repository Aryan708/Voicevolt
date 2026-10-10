"""Terminal chat with your knowledge base (text in, cited answer out).

  python scripts/chat.py
  python scripts/chat.py "what did we decide about hiring?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicevault.llm import get_llm  # noqa: E402
from voicevault.rag import RAGPipeline  # noqa: E402
from voicevault.retrieval import HybridRetriever, open_store  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("-k", type=int, default=5)
    a = p.parse_args()

    store, temb, iemb = open_store()
    rag = RAGPipeline(HybridRetriever(store, temb, iemb), get_llm(), k=a.k)
    history: list[dict] = []
    print(f"VoiceVault chat · {rag.llm.name} · empty line to quit")
    try:
        while True:
            q = a.question or input("\nyou> ").strip()
            if not q:
                break
            print("vault> ", end="", flush=True)
            answer = None
            for ev in rag.stream(q, history):
                if ev["type"] == "sources" and ev["query"] != q:
                    print(f"(searching: {ev['query']})\n       ", end="", flush=True)
                elif ev["type"] == "token":
                    print(ev["text"], end="", flush=True)
                elif ev["type"] == "done":
                    answer = ev["answer"]
            print()
            for s in answer.sources():
                print(f"   [{s['n']}] {s['citation']}" + (f"  🖼 {s['image']}" if s["image"] else ""))
            print(f"   ⏱ retrieval {answer.timings['retrieval']*1000:.0f}ms · "
                  f"first token {answer.timings.get('first_token', 0)*1000:.0f}ms · "
                  f"total {answer.timings['total']*1000:.0f}ms")
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer.text}]
            if a.question:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        store.close()


if __name__ == "__main__":
    main()
