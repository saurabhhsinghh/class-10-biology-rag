"""Experiment: what does Whisper give us if we force language="en"?

Run with:
    uv run python eval/langtest.py            # chunk 0
    uv run python eval/langtest.py --chunk 3  # a different one

The question
------------
A student types Roman Hinglish ("aankh ki working samjhao"). Our transcript is
77% Devanagari, because phase 2 forced language="hi". Those two do not match,
and we measured the cost: the same question scores 0.707 against Devanagari
text but only 0.420 against our romanised version.

An example project (see the conversation) hit the same question and answered it
BEFORE transcribing: they ran Whisper twice and picked the language flag that
kept their technical terms clean. With language="en" on Hinglish audio, Whisper
does not transliterate -- it TRANSLATES. The output is English prose.

So the question this script answers is narrow and concrete: for OUR video, is
the English output good enough to be the searchable text?

What to look for
----------------
1. Are the biology terms right? photosynthesis, neuron, nephron, reflex --
   these are the words a student actually types, and they must survive.
2. How much is actually translated vs. left alone? Whisper translates the
   Hindi it can and often mangles the rest into fluent-sounding nonsense.
   Read for meaning, not for grammar.
3. Does it invent things the teacher never said? Translation gives the model
   more room to hallucinate than plain transcription does.

This script writes NOTHING into data/. It is a read-only probe, and it caches
its Groq response so re-running costs no API call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ingest"))

# Windows terminals default to cp1252 here, which cannot print Devanagari --
# the very thing this script exists to show. Without this the comparison
# crashes on its first Hindi line.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from config import AUDIO_DIR, RAW_DIR, WHISPER_MODEL  # noqa: E402

CHUNKS_DIR = AUDIO_DIR / "chunks"
CACHE_DIR = ROOT / "eval" / "langtest_cache"

_DEV = re.compile(r"[ऀ-ॿ]")
_LAT = re.compile(r"[A-Za-z]")


def script_mix(text: str) -> str:
    """Rough Devanagari-vs-Latin split, for an at-a-glance comparison."""
    d = len(_DEV.findall(text))
    l = len(_LAT.findall(text))                              # noqa: E741
    total = d + l
    if not total:
        return "no letters at all"
    return f"{d / total * 100:3.0f}% Devanagari / {l / total * 100:3.0f}% Latin"


def transcribe(client, path: Path, language: str) -> list[dict]:
    with open(path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            file=(path.name, fh.read()),
            model=WHISPER_MODEL,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    raw = getattr(resp, "segments", None) or []
    out = []
    for seg in raw:
        get = seg.get if isinstance(seg, dict) else (lambda k, d=None: getattr(seg, k, d))
        out.append({
            "start": float(get("start", 0.0)),
            "end": float(get("end", 0.0)),
            "text": (get("text", "") or "").strip(),
        })
    return out


def fmt(seconds: float) -> str:
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--rows", type=int, default=12,
                    help="how many aligned rows to print (default 12)")
    args = ap.parse_args()

    idx = args.chunk
    audio = CHUNKS_DIR / f"chunk_{idx:03d}.mp3"
    hindi_json = RAW_DIR / f"chunk_{idx:03d}.json"
    if not audio.exists():
        sys.exit(f"ERROR: {audio} not found.")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY is not set in .env")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"en_chunk_{idx:03d}.json"

    if cache.exists():
        print(f"\n        using cached English transcript ({cache.name}) -- no API call")
        english = json.loads(cache.read_text(encoding="utf-8"))["segments"]
    else:
        print(f"\n        sending {audio.name} ({audio.stat().st_size / 1024 / 1024:.1f} MB) "
              f"to Groq with language=\"en\" ...")
        from groq import Groq
        english = transcribe(Groq(api_key=api_key), audio, "en")
        cache.write_text(
            json.dumps({"chunk_index": idx, "language": "en", "segments": english},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"        {len(english)} segments  (cached to {cache.name})")

    hindi = []
    if hindi_json.exists():
        hindi = json.loads(hindi_json.read_text(encoding="utf-8"))["segments"]

    # ---- headline numbers -------------------------------------------------
    en_text = " ".join(s["text"] for s in english)
    hi_text = " ".join(s["text"] for s in hindi) if hindi else ""

    print(f"\n{'':<14}{'segments':>9}  {'words':>6}   script mix")
    print(f"  {'HINDI (hi)':<12}{len(hindi):>9}  {len(hi_text.split()):>6}   "
          f"{script_mix(hi_text) if hi_text else '-'}")
    print(f"  {'ENGLISH (en)':<12}{len(english):>9}  {len(en_text.split()):>6}   "
          f"{script_mix(en_text)}")

    # ---- how much Hindi bled through --------------------------------------
    if hindi:
        hi_words = {w for w in hi_text.split() if _DEV.search(w)}
        en_words = en_text.split()
        # Any Devanagari at all in the English output means Whisper gave up on
        # translating that bit and left the original sitting there.
        leftover = sum(1 for w in en_words if _DEV.search(w))
        print(f"\n  Devanagari words surviving into the English output: "
              f"{leftover} of {len(en_words)} ({leftover / max(len(en_words), 1) * 100:.1f}%)")

    # ---- the bit that actually matters: can you read it? -------------------
    print(f"\n--- aligned view: first {args.rows} pairs ---")
    for i in range(min(args.rows, max(len(english), len(hindi)))):
        e = english[i]["text"] if i < len(english) else ""
        h = hindi[i]["text"] if i < len(hindi) else ""
        ts = fmt(english[i]["start"]) if i < len(english) else "?"
        print(f"\n  [{ts}]")
        print(f"    HI: {h[:150]}")
        print(f"    EN: {e[:150]}")

    # ---- the terms a student will actually type ---------------------------
    TERMS = ["photosynthesis", "neuron", "nephron", "reflex", "hormone",
             "ozone", "cornea", "retina", "enzyme", "chlorophyll",
             "artery", "vein", "plasma", "stomata", "thyroid", "insulin"]
    low = en_text.lower()
    found = [t for t in TERMS if t in low]
    print(f"\n  English technical terms present in this one chunk: "
          f"{', '.join(found) if found else '(none)'}")
    print("\n  NOTE: one chunk is 10 minutes of a 4-hour video. Absence here "
          "means nothing;\n  this is a look at the SHAPE of the output, not a "
          "coverage measure.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
