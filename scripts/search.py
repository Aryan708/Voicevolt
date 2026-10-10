"""Search the knowledge base.

  python scripts/search.py "what did we decide about the Chennai launch?"
  python scripts/search.py "bar chart of regional revenue" --modality video document
  python scripts/search.py                     # interactive mode
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicevault.retrieval import HybridRetriever, open_store  # noqa: E402


def show(hits, query: str) -> None:
    print(f"\n🔎 {query}")
    if not hits:
        print("  (no results)")
    for i, h in enumerate(hits, 1):
        ranks = " ".join(f"{k}#{v}" for k, v in sorted(h.ranks.items()))
        print(f"\n  {i}. {h.chunk.citation}   [{h.chunk.modality}]  score={h.score:.4f}  ({ranks})")
        if h.image:
            print(f"     🖼  {h.image}")
        body = h.chunk.text.strip().replace("\n", " ")
        print(textwrap.indent(textwrap.fill(body[:350] + ("…" if len(body) > 350 else ""),
                                            width=100), "     "))


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid multimodal search over VoiceVault.")
    p.add_argument("query", nargs="?")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--source", help="Only this file name")
    p.add_argument("--modality", nargs="+", choices=["audio", "video", "document", "image"])
    p.add_argument("--only", nargs="+", choices=["dense", "bm25", "visual"],
                   help="Use only some retrievers (for comparing / ablation)")
    a = p.parse_args()

    store, temb, iemb = open_store()
    r = HybridRetriever(store, temb, iemb)
    kw = dict(k=a.k, source=a.source, modality=a.modality)
    if a.only:
        kw["retrievers"] = tuple(a.only)

    try:
        if a.query:
            show(r.search(a.query, **kw), a.query)
        else:
            print("Interactive search - empty line to quit.")
            while q := input("\n> ").strip():
                show(r.search(q, **kw), q)
    finally:
        store.close()


if __name__ == "__main__":
    main()
