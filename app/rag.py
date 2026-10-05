"""Retrieval and answer generation -- the engine behind the UI.

Everything in here is deliberately free of Streamlit. The UI calls these
functions; if you later want a FastAPI version, or a command-line version, or
an evaluation harness, this file does not change.

Flow:
    ask(question)
        -> convert the Hindi words, leave the English ones
        -> search with both, keep the better match per chunk
        -> hand the excerpts to Gemini
        -> {answer, citations, refused}

WHY THE QUESTION IS CONVERTED BEFORE SEARCHING
----------------------------------------------
The transcript holds two kinds of writing and nothing in between: Hindi in
Devanagari, and English words left in Latin letters ("human eye", "reflex
action", "food chain"). That is how the speaker talked and how Whisper wrote it
down.

A question therefore matches the transcript only if its words are written the
way the transcript writes them:

    "aankh ki working samjhao"     0.420   "aankh" is Roman Hindi -- a form
    "aankh ke parts"               0.401   the transcript does not contain
    "eye ke parts batao"           0.603   anywhere
    "human eye ka structure"       0.607
    "आंख की संरचना समझाओ"           0.721

Same question, same meaning, 0.42 against 0.72. English words match and
Devanagari words match; Roman Hindi falls through, because it is a third script
the transcript never uses.

So the Hindi words are converted to Devanagari with the English left alone, and
BOTH that query and the original are searched -- the better score per chunk
wins. Searching twice cannot lose: a poor conversion simply scores lower and
the original query carries the result.

TWO EARLIER ATTEMPTS, AND WHY THEY FAILED
-----------------------------------------
Romanising the TRANSCRIPT. Cannot work. Whisper wrote "support" as सपोर्ट,
which throws the English spelling away -- Devanagari records sounds, not
spellings. Romanising it back gave "saport", a word no student types. The
information died at transcription time and no later processing brings it back.

Rewriting the question with an LLM, first try. This one was close, and the
failure is worth spelling out, because the instruction got exactly one thing
wrong. It said to write English technical words into Devanagari too, so
"reflex action kaise hota hai" came back as "रिफ्लेक्स एक्शन कैसे होता है" --
while the transcript says "reflex action", in English. The conversion had
destroyed the one word that was doing the matching. The fix is one line of
instruction: convert the Hindi, leave the English.

That attempt was also declared a success too early, on the strength of a single
score. "aankh ki working samjhao" hit 0.746, the best number this project had
seen, and it was taken as proof. It was not proof. Score is not correctness --
the 0.746 chunk turned out to be about lens dioptres, not about how the eye
works -- and one wording is not evidence. What gets measured now is where the
first genuinely CORRECT chunk lands. That is the number this conversion has to
move, and it is the only number that counts.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "ingest"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from config import (  # noqa: E402
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_EXCERPTS,
    LLM_FALLBACK_MODELS,
    LLM_MODEL,
    LLM_RETRIES,
    LLM_RETRY_SLEEP,
    MAX_CITATIONS,
    SCORE_THRESHOLD,
    TEMPERATURE,
    TOP_K_SEARCH,
    VIDEO_ID,
    use_local_hf_cache,
)

from prompts import (  # noqa: E402
    CONVERT_INSTRUCTION,
    REFUSAL,
    SYSTEM_PROMPT,
    build_user_message,
)


# ---------------------------------------------------------------------------
# resources -- loaded once, reused for the life of the process
# ---------------------------------------------------------------------------

_MODEL = None
_CLIENT = None
_QDRANT = None


def get_model():
    """The embedding model (~470 MB, several seconds to load)."""
    global _MODEL
    if _MODEL is None:
        use_local_hf_cache()
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def get_qdrant():
    global _QDRANT
    if _QDRANT is None:
        from qdrant_client import QdrantClient
        url = os.environ.get("QDRANT_URL")
        key = os.environ.get("QDRANT_API_KEY")
        if not url or not key:
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY missing from .env")
        _QDRANT = QdrantClient(url=url, api_key=key, timeout=30)
    return _QDRANT


def get_llm():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY missing from .env")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


def _call_llm(system_instruction: str, contents: str,
              fallback: str | None = None) -> tuple[str, str | None]:
    """Run one prompt against the model list, retrying transient failures.

    Returns (text, error). A 503 means Google is busy and it clears in seconds,
    so we walk the fallback models before giving up -- a student should not see
    a traceback because somebody else's traffic spiked.
    """
    import time

    from google.genai import types

    client = get_llm()
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=TEMPERATURE,
    )

    last: Exception | None = None
    for model in [LLM_MODEL, *LLM_FALLBACK_MODELS]:
        for _ in range(LLM_RETRIES):
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents, config=cfg,
                )
                text = (getattr(resp, "text", None) or "").strip()
                if text:
                    return text, None
                last = RuntimeError(f"{model} returned an empty answer")
            except Exception as exc:                            # noqa: BLE001
                last = exc
                # A daily quota does not clear in 1.5 seconds. Grinding through
                # the retries only makes the student wait for a page that is
                # already lost, so move to the next model immediately.
                if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                    break
            time.sleep(LLM_RETRY_SLEEP)

    if fallback is not None:
        return fallback, str(last)
    return "", str(last)


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------

# Conversions are remembered on disk. Two reasons, and the second is the one
# that forced it:
#
#   1. The same question asked twice costs nothing the second time.
#   2. The free Gemini tier allows 20 requests per model per day, and half the
#      app's calls are conversions. Re-running eval/smoke.py after an unrelated
#      change used to spend a dozen calls re-deriving answers we already had --
#      and on 2026-10-04 it spent the whole day's quota and every model
#      returned 429. A cache turns "run it again" from a cost into free.
#
# The file is gitignored: it is derived, and it holds whatever students typed.
_CONVERT_CACHE_FILE = ROOT / "data" / "processed" / "convert_cache.json"
_CONVERT_CACHE: dict[str, str] | None = None


def _cache_load() -> dict[str, str]:
    global _CONVERT_CACHE
    if _CONVERT_CACHE is None:
        try:
            _CONVERT_CACHE = json.loads(
                _CONVERT_CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Missing is normal on a first run; corrupt is not worth an error
            # page. Starting empty costs one extra API call and nothing else.
            _CONVERT_CACHE = {}
    return _CONVERT_CACHE


def _cache_save() -> None:
    if _CONVERT_CACHE is None:
        return
    try:
        _CONVERT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CONVERT_CACHE_FILE.write_text(
            json.dumps(_CONVERT_CACHE, ensure_ascii=False, indent=1),
            encoding="utf-8")
    except OSError:
        pass          # a cache that cannot be written is not worth failing over


def to_search_form(question: str) -> str:
    """The question with its Hindi words in Devanagari, English words untouched.

    Returns the question unchanged if the model is unreachable. Then `retrieve`
    searches one query instead of two, which costs a little accuracy and is
    much better than an error page.
    """
    cache = _cache_load()
    if question in cache:
        return cache[question]

    text, err = _call_llm(CONVERT_INSTRUCTION, question, fallback=question)
    if err:
        return question          # deliberately not cached, so the next try retries

    got = text.strip().strip('"').strip("'").strip()
    if not got:
        return question
    # The model occasionally adds a full stop or a trailing remark on a second
    # line. Neither belongs in a search string.
    got = got.splitlines()[0].strip().rstrip(".?!").strip() or question

    cache[question] = got
    _cache_save()
    return got


def retrieve(question: str, limit: int = TOP_K_SEARCH) -> tuple[list[dict], str]:
    """Search the transcript index. Returns (chunks best-first, converted query).

    Searches twice -- the question as typed, and the converted form from
    `to_search_form` -- and keeps the better score for each chunk. That cannot
    lose: if the conversion is poor its scores are simply lower and the
    original query carries the result. See the module docstring for why the
    conversion is needed at all.

    Every returned chunk carries `score`, `text`, and the timestamps the UI
    needs to seek the video.
    """
    converted = to_search_form(question)
    queries = [question] if converted == question else [question, converted]

    vectors = get_model().encode(queries, normalize_embeddings=True)

    best: dict[int, dict] = {}
    for qv in vectors:
        hits = get_qdrant().query_points(
            collection_name=COLLECTION_NAME,
            query=qv.tolist(),
            limit=limit,
            with_payload=True,
        ).points
        for h in hits:
            if h.id in best and h.score <= best[h.id]["score"]:
                continue
            best[h.id] = {
                "id": h.id,
                "score": float(h.score),
                **{k: v for k, v in h.payload.items() if k != "chunk_id"},
            }

    ranked = sorted(best.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:limit], converted


def _to_seconds(ts: str) -> int | None:
    """'2:20:55' -> 8363, '4:49' -> 289. None if it is not a timestamp.

    Both shapes are real. The transcript writes a timestamp without an hours
    field when the hour is zero ("4:49") and with one when it is not
    ("2:20:55"), so two fields always mean minutes:seconds -- in a four-hour
    lecture there is no "4:49" that could be four hours.
    """
    parts = ts.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


# A timestamp the model wrote, counted only when it sits inside square brackets.
# The model is asked to mark each sentence with the excerpt it came from, and
# bracket marks are the one convention it keeps to. A bare "2:20:55" in prose
# is not counted, because a number that looks like a time can appear for other
# reasons ("10:10" in a diagram caption, say).
_TS_IN_BRACKETS = re.compile(r"\[([^\]]{0,64})\]")
_TS = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")


def citations_from_answer(answer: str, chunks: list[dict],
                          max_n: int = MAX_CITATIONS) -> list[dict]:
    """The timestamps the ANSWER actually used, in the order it used them.

    This used to be "the top chunks by score", and that was quietly wrong. The
    model writes its answer from whichever excerpts fit the question, and those
    are not always the highest-scoring ones. For "food chain kya hoti hai" the
    buttons pointed at 2:02:52 -- a stretch about hormones -- while the answer
    itself was describing 3:33:53. Clicking a timestamp took the student
    somewhere the answer never mentioned. Pointing at the right minute is the
    entire product; getting it wrong is worse than showing no button.

    So we take the model at its word. It already marks each sentence with the
    excerpt it came from, and those marks ARE the citations. A timestamp that
    matches no excerpt we sent is dropped rather than shown -- if the model
    invented one, a dead button is the worse outcome.

    Matching is on seconds, not on the string, so "03:33:53" still finds
    3:33:53.
    """
    by_second: dict[int, dict] = {}
    for c in chunks:
        sec = _to_seconds(c.get("start_fmt", ""))
        if sec is not None:
            by_second.setdefault(sec, c)

    chosen: list[dict] = []
    seen: set[int] = set()
    for group in _TS_IN_BRACKETS.findall(answer):
        for ts in _TS.findall(group):
            sec = _to_seconds(ts)
            if sec is None or sec in seen or sec not in by_second:
                continue
            seen.add(sec)
            chosen.append(by_second[sec])
            if len(chosen) >= max_n:
                return chosen
    return chosen


# ---------------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------------

def ask(question: str) -> dict:
    """Answer one question.

    Returns {answer, citations, refused, chunks, converted, best_score}.
    `converted` is the search form of the question -- what the Hindi words were
    turned into before searching. The UI shows it, because when a search goes
    wrong the first thing worth checking is whether the question survived the
    conversion.
    """
    question = question.strip()
    if not question:
        return {"answer": REFUSAL, "citations": [], "refused": True,
                "chunks": [], "converted": ""}

    chunks, converted = retrieve(question)
    if not chunks:
        return {"answer": REFUSAL, "citations": [], "refused": True,
                "chunks": [], "converted": converted}

    best = chunks[0]["score"]
    if best < SCORE_THRESHOLD:
        # The coarse fence, and it is only a fence. Scores of covered and
        # uncovered questions overlap (see SCORE_THRESHOLD in config.py), so
        # this catches nonsense and nothing subtler. A genuine question that
        # scores weakly still goes to the model, because the model reading the
        # excerpts is the guard that actually works.
        return {"answer": REFUSAL, "citations": [], "refused": True,
                "chunks": chunks, "converted": converted, "best_score": best}

    answer, err = _call_llm(
        SYSTEM_PROMPT,
        build_user_message(question, chunks[:LLM_EXCERPTS]),
    )
    if err:
        # The raw error is not something to hand a Class 10 student -- a quota
        # failure is a screenful of JSON. The student gets a sentence; the
        # detail goes in `error` for the debug panel.
        return {
            "answer": "Couldn't build an answer just now — please try again in a bit. 🔌",
            "citations": [], "refused": False, "error": err,
            "chunks": chunks, "converted": converted, "best_score": best,
        }

    refused = REFUSAL[:20].lower() in answer.lower()
    return {
        "answer": answer,
        "citations": [] if refused else citations_from_answer(answer, chunks),
        "refused": refused,
        "chunks": chunks,
        "converted": converted,
        "best_score": best,
    }


def video_embed_url(start_seconds: int = 0, autoplay: bool = False) -> str:
    """The iframe URL for the lecture, optionally starting at a timestamp."""
    url = f"https://www.youtube.com/embed/{VIDEO_ID}?rel=0&modestbranding=1"
    if start_seconds:
        url += f"&start={int(start_seconds)}"
    if autoplay:
        url += "&autoplay=1"
    return url
