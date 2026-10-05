"""Phase 2 -- transcribe the audio chunks with Whisper, via Groq.

Run with:
    uv run python ingest/02_transcribe.py --limit 1     # test ONE chunk first
    uv run python ingest/02_transcribe.py               # then all 24

Why Groq rather than local Whisper
----------------------------------
Groq serves whisper-large-v3 on its free tier, on custom hardware -- far faster
than CPU inference and no model download. If you hit rate limits, the fallback
is local `faster-whisper` (see WHISPER_LOCAL_MODEL in config.py).

THE IMPORTANT PART -- timestamps
--------------------------------
Whisper knows nothing about your 4-hour video. It receives one 10-minute file
and returns timestamps starting at 0:00. Chunk 5 has no idea it begins at 40:00.

So every segment's start/end MUST be shifted by that chunk's offset before it is
stored. Get this wrong and every citation after the first chunk points to the
wrong moment -- off by exactly 600s per chunk index. It fails PLAUSIBLY (the
video jumps somewhere nearby but wrong), so you would blame the player rather
than the arithmetic.

Offsets are MEASURED with ffprobe, not assumed to be exactly 600s, because
ffmpeg's segmenter cuts on MP3 frame boundaries rather than on the second.

Output
------
    data/raw/chunk_000.json, chunk_001.json, ...      (gitignored)

Segments in those files carry ABSOLUTE timestamps (offset already applied), so
no downstream script ever has to think about offsets again. Existing output is
skipped, so a re-run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from config import (  # noqa: E402
    AUDIO_DIR,
    RAW_DIR,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CHUNKS_DIR = AUDIO_DIR / "chunks"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def probe_duration(path: Path) -> float:
    """Exact duration of an audio file in seconds, via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        sys.exit(
            "\nERROR: ffprobe is not on your PATH.\n"
            "  It ships with ffmpeg -- if ffmpeg works, open a NEW terminal.\n"
        )
    out = subprocess.run(
        [ffprobe, "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def compute_offsets(chunks: list[Path]) -> list[float]:
    """Cumulative start time of each chunk in the original video.

    Chunk 0 starts at 0.0. Chunk 1 starts at (length of chunk 0). And so on.
    Measured rather than assumed, so a chunk that is 600.07s long doesn't push
    every later timestamp out by 0.07s.
    """
    offsets: list[float] = []
    running = 0.0
    for c in chunks:
        offsets.append(running)
        running += probe_duration(c)
    return offsets


def transcribe_chunk(client, path: Path) -> list[dict]:
    """Send one chunk to Groq. Returns segments with RELATIVE timestamps."""
    with open(path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            file=(path.name, fh.read()),
            model=WHISPER_MODEL,
            language=WHISPER_LANGUAGE,      # force Hindi -- do not let it guess
            response_format="verbose_json",  # required for timestamps
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
    """Seconds -> h:mm:ss, or m:ss below an hour."""
    s = int(seconds)
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N chunks (use 1 to test first)")
    ap.add_argument("--force", action="store_true",
                    help="re-transcribe chunks whose JSON already exists")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY is not set. Fill it in at .env")

    chunks = sorted(CHUNKS_DIR.glob("chunk_*.mp3"))
    if not chunks:
        sys.exit(f"ERROR: no chunks found in {CHUNKS_DIR}. Run 01_download.py first.")

    if args.limit:
        chunks = chunks[: args.limit]

    print()
    print(f"        model  : {WHISPER_MODEL}  (via Groq)")
    print(f"        language: {WHISPER_LANGUAGE}  (forced)")
    print(f"        chunks : {len(chunks)} of {len(sorted(CHUNKS_DIR.glob('chunk_*.mp3')))}")
    print()

    # Measuring offsets for only the chunks we will process.
    all_chunks = sorted(CHUNKS_DIR.glob("chunk_*.mp3"))
    offsets = compute_offsets(all_chunks)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    from groq import Groq
    client = Groq(api_key=api_key)

    done, failed = 0, []

    for path in chunks:
        idx = int(path.stem.split("_")[1])
        out_path = RAW_DIR / f"chunk_{idx:03d}.json"
        offset = offsets[idx]

        if out_path.exists() and not args.force:
            print(f"[{idx:02d}]  skip -- already transcribed")
            done += 1
            continue

        print(f"[{idx:02d}]  offset {fmt(offset):>8}  ->  {path.name} ({path.stat().st_size/1024/1024:.1f} MB)")

        segments = None
        for attempt in range(3):
            try:
                segments = transcribe_chunk(client, path)
                break
            except Exception as exc:                      # noqa: BLE001
                wait = 2 ** attempt
                print(f"        ! attempt {attempt + 1} failed: {exc}")
                if attempt == 2:
                    failed.append((idx, str(exc)))
                else:
                    print(f"        retrying in {wait}s...")
                    time.sleep(wait)

        if segments is None:
            print("        giving up on this chunk -- re-run later to resume")
            continue

        # THE OFFSET CORRECTION: shift every timestamp into video time.
        for seg in segments:
            seg["start"] += offset
            seg["end"] += offset

        out_path.write_text(
            json.dumps({
                "chunk_index": idx,
                "chunk_file": path.name,
                "offset_seconds": offset,
                "language": WHISPER_LANGUAGE,
                "model": WHISPER_MODEL,
                "segments": segments,
            }, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        done += 1
        print(f"        {len(segments)} segments written")

    print(f"\nDone. {done} chunk(s) available in {RAW_DIR}")
    if failed:
        print(f"\n{len(failed)} chunk(s) failed:")
        for idx, err in failed:
            print(f"  chunk {idx:03d}: {err}")
        print("\nRe-run the same command to resume -- finished chunks are skipped.")

    # Show a taste of the actual transcript, so you can eyeball the quality.
    sample = sorted(RAW_DIR.glob("chunk_*.json"))
    if sample:
        data = json.loads(sample[0].read_text(encoding="utf-8"))
        print(f"\n--- first 5 segments of {sample[0].name} (offsets applied) ---")
        for seg in data["segments"][:5]:
            print(f"  [{fmt(seg['start'])}]  {seg['text'][:80]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
