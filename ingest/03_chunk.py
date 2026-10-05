"""Phase 3 -- glue Whisper's segments together into chunks.

Run with:
    uv run python ingest/03_chunk.py

What this does
--------------
Phase 2 left us with 24 separate JSON files, one per 10-minute audio piece,
holding about 2,200 segments. A SEGMENT is wherever Whisper happened to break
-- sometimes 3 words, sometimes 80.

A 3-word piece is useless for search. There is nothing in it to match a
question against, and the timestamp it carries is too narrow to be a useful
citation. So we glue neighbouring segments together until the piece is roughly
TARGET_TOKENS long, then cut. That piece is a CHUNK.

Chunks overlap by OVERLAP_TOKENS, so a sentence that straddles a cut is still
whole inside at least one chunk.

Why we count with the REAL tokenizer, not words
-----------------------------------------------
The embedding model silently truncates anything past MAX_SEQ_LENGTH (128) --
no warning, no error. If we counted words, a chunk we believe is "100 words"
could be 160 real tokens, and the last third would simply never be embedded.
Nothing would fail. Retrieval would just be quietly worse, which is the worst
kind of bug to find. So we count with the exact tokenizer the model uses.

Output
------
    data/processed/chunks.jsonl      (committed -- the project's raw material)

Each chunk carries the Devanagari `text`, the seconds it starts and ends at,
and those same times written out (`start_fmt`) for the model and the UI to
quote. Nothing else -- an earlier version also stored a Romanised copy of every
chunk, so that search could run in Roman script. That approach is gone and the
reasons are in app/rag.py; carrying the field around afterwards would only have
raised the question of what still reads it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# This terminal defaults to cp1252, which cannot print Devanagari -- and the
# preview at the end of this script prints chunks.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (  # noqa: E402
    BLIP_GAP_SECONDS,
    BLIP_MAX_WORDS,
    CHUNK_TOKEN_CEILING,
    CHUNKS_FILE,
    EMBEDDING_MODEL,
    MAX_SEQ_LENGTH,
    OVERLAP_TOKENS,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_TOKENS,
    VIDEO_DURATION_SECONDS,
    use_local_hf_cache,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt(seconds: float) -> str:
    """Seconds -> h:mm:ss, or m:ss below an hour."""
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def load_tokenizer():
    """Load the EXACT tokenizer the embedding model will use.

    Only the tokenizer, not the model -- that is a few MB and needs no torch.
    """
    # HF_HOME must be set BEFORE transformers is imported -- it reads it once.
    use_local_hf_cache()

    from transformers import AutoTokenizer

    for name in (EMBEDDING_MODEL, f"sentence-transformers/{EMBEDDING_MODEL}"):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            print(f"        tokenizer: {name}")
            return tok
        except Exception:                                    # noqa: BLE001
            continue

    sys.exit(
        f"\nERROR: could not load the tokenizer for '{EMBEDDING_MODEL}'.\n"
        "  Check your internet connection (it is downloaded once, then cached).\n"
    )


# Any letter or digit, in any script -- Devanagari included.
_HAS_WORD = re.compile(r"[^\W_]", re.UNICODE)


def is_noise(text: str) -> bool:
    """True for a segment carrying no actual words (empty, or only ♪ or punctuation)."""
    return not _HAS_WORD.search(text)


def drop_isolated_blips(segments: list[dict]) -> list[dict]:
    """Remove Whisper's hallucinated specks.

    On non-speech audio Whisper does not stay quiet -- it invents something.
    The source video has a ~12-minute break at 2:29, and across it Whisper
    emitted one nonsense word every 30 seconds. Those specks are harmless
    individually but glue together into a chunk that spans 13 minutes of
    nothing, which then shows up in search results carrying a bogus timestamp.

    A speck has two tells, and we require both: it is very short, AND the next
    segment does not begin until long after it ends. Real teaching is neither.
    """
    keep: list[dict] = []
    for i, seg in enumerate(segments):
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        silence_after = (nxt["start"] - seg["end"]) if nxt else 0.0

        if (silence_after > BLIP_GAP_SECONDS
                and len(seg["text"].split()) <= BLIP_MAX_WORDS):
            continue
        keep.append(seg)
    return keep


def load_segments() -> list[dict]:
    """Flatten all data/raw/chunk_*.json into ONE list, in video order."""
    files = sorted(RAW_DIR.glob("chunk_*.json"))
    if not files:
        sys.exit(f"ERROR: no raw JSON in {RAW_DIR}. Run 02_transcribe.py first.")

    segments: list[dict] = []
    noise = repeats = 0

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for seg in data.get("segments", []):
            text = (seg.get("text") or "").strip()

            if not text or is_noise(text):
                noise += 1
                continue

            # Whisper's best-known failure: it gets stuck and emits the same
            # line over and over. Only IMMEDIATE repeats are dropped -- a phrase
            # genuinely said twice, with other speech in between, is kept.
            if segments and segments[-1]["text"] == text:
                repeats += 1
                continue

            segments.append({
                "text": text,
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "source": path.stem,
            })

    segments.sort(key=lambda s: (s["start"], s["end"]))
    before = len(segments)
    segments = drop_isolated_blips(segments)
    blips = before - len(segments)

    print(f"        segments : {len(segments)} kept  "
          f"({noise} noise, {repeats} repeats, {blips} hallucinated specks dropped)")
    return segments


def build_chunks(segments: list[dict], tokenizer) -> list[dict]:
    """Sliding window over segments, cutting at ~TARGET_TOKENS with overlap.

    Tokens are counted with the model's own tokenizer, on the exact text that
    will be embedded. That is the entire point of counting: the embedding model
    truncates past MAX_SEQ_LENGTH silently, so a chunk that is measured wrong
    is a chunk that is quietly half-embedded, and nothing anywhere fails.

    (An earlier version counted one text and embedded a different one, and 92
    of 660 chunks arrived at the model over-length. The ceiling has to guard
    the text that actually gets embedded, whichever text that is.)
    """
    counts = [len(tokenizer.encode(s["text"], add_special_tokens=False))
              for s in segments]

    n = len(segments)
    chunks: list[dict] = []
    i = 0

    while i < n:
        # Grow the window until it reaches the target. Two stops:
        #   - `j == i`  : always take at least one segment, even if that single
        #                 segment is longer than the target on its own.
        #   - ceiling   : never cross CHUNK_TOKEN_CEILING, or the embedding
        #                 model will silently chop the end off this chunk.
        j, tokens = i, 0
        while j < n:
            if j > i and tokens + counts[j] > CHUNK_TOKEN_CEILING:
                break
            tokens += counts[j]
            j += 1
            if tokens >= TARGET_TOKENS:
                break

        # The per-segment counts above are an estimate. Sub-word boundaries
        # shift when the pieces are joined with spaces, so the real count of
        # the joined text can differ. Measure it, and if it still overshoots,
        # shed segments from the tail until it fits. This is the guarantee --
        # everything above is just a good guess at where to cut.
        text = " ".join(s["text"] for s in segments[i:j])
        while j - i > 1 and \
                len(tokenizer.encode(text, add_special_tokens=False)) > CHUNK_TOKEN_CEILING:
            j -= 1
            text = " ".join(s["text"] for s in segments[i:j])

        window = segments[i:j]
        chunks.append({
            "id": len(chunks),
            "text": text,                 # Devanagari, as Whisper heard it
            "start": round(window[0]["start"], 2),
            "end": round(window[-1]["end"], 2),
            "start_fmt": fmt(window[0]["start"]),
            "end_fmt": fmt(window[-1]["end"]),
            "tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "n_segments": len(window),
        })

        if j >= n:
            break

        # Step the window BACK so the next chunk re-reads ~OVERLAP_TOKENS of
        # this one. `k > i + 1` guarantees forward progress: even if one
        # segment alone blows the overlap budget, we still advance by one.
        k, back = j, 0
        while k > i + 1 and back < OVERLAP_TOKENS:
            k -= 1
            back += counts[k]
        i = k

    return chunks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preview", type=int, default=3,
                    help="how many chunks to print at the end (default 3)")
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print(f"        target   : {TARGET_TOKENS} tokens, overlap {OVERLAP_TOKENS}, "
          f"ceiling {CHUNK_TOKEN_CEILING}")

    tokenizer = load_tokenizer()
    segments = load_segments()
    chunks = build_chunks(segments, tokenizer)

    if not chunks:
        sys.exit("ERROR: no chunks were built -- the transcript looks empty.")

    CHUNKS_FILE.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks),
        encoding="utf-8",
    )

    counts = [c["tokens"] for c in chunks]
    over = [c for c in chunks if c["tokens"] > MAX_SEQ_LENGTH]
    last_end = max(c["end"] for c in chunks)

    print(f"        chunks   : {len(chunks)}")
    print(f"        tokens   : min {min(counts)} / mean {sum(counts) // len(counts)} "
          f"/ max {max(counts)}")
    print(f"        coverage : 0:00 -> {fmt(last_end)}  "
          f"(video {fmt(VIDEO_DURATION_SECONDS)})")

    if over:
        print(f"\n  !! {len(over)} chunk(s) exceed {MAX_SEQ_LENGTH} tokens -- "
              f"the embedding model WILL truncate them.")
        for c in over[:5]:
            print(f"     chunk {c['id']}: {c['tokens']} tokens @ {c['start_fmt']}")
    else:
        print(f"\n        truncation check: none of the {len(chunks)} chunks "
              f"exceed {MAX_SEQ_LENGTH} tokens  -- OK")

    print(f"\nWrote {CHUNKS_FILE}")
    print(f"      {CHUNKS_FILE.stat().st_size / 1024:.0f} KB\n")

    if args.preview:
        print(f"--- first {args.preview} chunk(s) ---")
        for c in chunks[: args.preview]:
            print(f"\n  #{c['id']}  [{c['start_fmt']} - {c['end_fmt']}]  "
                  f"{c['tokens']} tokens, from {c['n_segments']} segments")
            print(f"      {c['text'][:150]}...")
        print()

    print("Next -- turn each chunk into numbers (embeddings):")
    print("      uv run python ingest/04_embed.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
