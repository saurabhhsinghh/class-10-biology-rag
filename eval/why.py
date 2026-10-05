"""Why does one question refuse? Print what the search actually returned.

    uv run python eval/why.py "aankh ki working samjhao"
    uv run python eval/why.py 15 "aankh ki working samjhao"     # show more

Reads only -- no model is called. Shows the top N chunks with score and text,
so you can see whether the search failed (the right thing is not in the list)
or the model did (it is in the list and the model refused anyway).

The optional leading number is how many chunks to print. It defaults to 10 --
what the model is actually given -- but raising it is worth doing whenever the
top 10 look one-sided: the search retrieves TOP_K_SEARCH (15) and hands over
only LLM_EXCERPTS (10), so ranks 11-15 are already in hand and were simply not
shown to the model. Both numbers live in ingest/config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
# config.py lives in ingest/. rag.py adds that path itself, but only once rag is
# imported -- and this file reads config first, so it has to say so explicitly.
sys.path.insert(0, str(ROOT / "ingest"))

from config import LLM_EXCERPTS, TOP_K_SEARCH  # noqa: E402
from rag import retrieve  # noqa: E402

args = sys.argv[1:]
limit = max(TOP_K_SEARCH, LLM_EXCERPTS)
if args and args[0].isdigit():
    limit = int(args.pop(0))

question = " ".join(args) or "aankh ki working samjhao"
chunks, converted = retrieve(question, limit=limit)

print(f"Q: {question}")
if converted != question:
    print(f"   converted to: {converted}")
print(f"   showing top {len(chunks)}  "
      f"(the model is given the first {LLM_EXCERPTS})")
print()
for rank, c in enumerate(chunks, 1):
    given = "  <- model sees this" if rank <= LLM_EXCERPTS else ""
    print(f"{rank:2d}. {c['score']:.3f}  {c['start_fmt']}{given}")
    print(f"    {c['text'][:220]}")
    print()
