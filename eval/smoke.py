"""Ask the app a handful of questions and look at what came back.

NOT the real evaluation -- that is eval/run_eval.py, with a scored question set.
This is the two-minute version you run after touching retrieval, to see whether
anything actually moved.

    uv run python eval/smoke.py

Read it in this order:
  1. `best score` -- did the search find anything at all? Below SCORE_THRESHOLD
     the app refuses before the model is even called.
  2. `timestamps` -- did the model choose to cite? No timestamps on a covered
     question means it refused, and the excerpts above are why.
  3. the answer text -- is it grounded in the lecture, or improvised?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from rag import ask  # noqa: E402

QUESTIONS = [
    "reflex action kaise hota hai",
    "aankh ki working samjhao",
    "neuron kya hota hai",
    "food chain kya hoti hai",
    "hormone kya kaam karta hai",
    "cricket ka score kya tha",       # NOT in the lecture -- must refuse
]


def main() -> None:
    for q in QUESTIONS:
        r = ask(q)
        print("=" * 78)
        print(f"Q: {q}")
        if r.get("converted") and r["converted"] != q:
            print(f"   searched as: {r['converted']}")
        print(f"   best {r.get('best_score', 0):.3f}   "
              f"refused={r['refused']}   "
              f"cites={[c['start_fmt'] for c in r['citations']]}")
        print()
        for line in r["answer"].splitlines():
            print(f"   {line}")
        print()


if __name__ == "__main__":
    main()
