"""Phase 1 -- download the lecture audio and split it into transcribable pieces.

Run with:
    uv run python ingest/01_download.py

What it does
------------
1. Asks yt-dlp for the AUDIO STREAM ONLY. Never the video -- a 4-hour video
   stream is hundreds of MB; the audio alone is roughly 55 MB.
2. Converts it with ffmpeg to 16 kHz mono MP3. 16 kHz is what Whisper wants,
   and mono halves the size for no loss of anything we care about.
3. Splits that into AUDIO_CHUNK_SECONDS pieces with ffmpeg, because Groq's
   upload limit is 25 MB -- far smaller than a 4-hour file.

Steps 2 and 3 shell out to `ffmpeg`, so it must be on your PATH. If you just
installed it, open a NEW terminal first.

Re-running is cheap: any step whose output already exists is skipped, unless
you pass --force.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Make `from config import ...` work when this file is run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SECONDS,
    AUDIO_DIR,
    AUDIO_SAMPLE_RATE,
    VIDEO_DURATION_SECONDS,
    VIDEO_URL,
)

RAW_STEM = AUDIO_DIR / "lecture_raw"     # yt-dlp writes lecture_raw.webm / .m4a / etc.
MP3_PATH = AUDIO_DIR / "lecture.mp3"     # the normalised 16 kHz mono file
CHUNKS_DIR = AUDIO_DIR / "chunks"        # the pieces we actually transcribe


def run(cmd: list[str]) -> None:
    """Run a subprocess, echoing the command so you can see what happened."""
    print("        $ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit(
            "\nERROR: ffmpeg is not on your PATH.\n"
            "  * If you just installed it, open a NEW terminal and run this again.\n"
            "  * Otherwise install it:  winget install --id Gyan.FFmpeg -e\n"
        )
    print(f"        ffmpeg: {exe}")
    return exe


def step_download(force: bool) -> Path:
    """Step 1 -- yt-dlp, audio stream only."""
    found = list(AUDIO_DIR.glob(RAW_STEM.name + ".*"))
    if found and not force:
        print(f"[1/3]   raw audio already downloaded: {found[0].name}")
        return found[0]

    print("[1/3]   downloading audio only -- this is the slow step, be patient...")
    run([
        sys.executable, "-m", "yt_dlp",
        "--format", "bestaudio/best",
        "--no-warnings",
        "--newline",
        "--output", str(RAW_STEM) + ".%(ext)s",
        VIDEO_URL,
    ])

    found = list(AUDIO_DIR.glob(RAW_STEM.name + ".*"))
    if not found:
        sys.exit("ERROR: yt-dlp finished but produced no audio file.")
    return found[0]


def step_convert(raw: Path, ffmpeg: str, force: bool) -> Path:
    """Step 2 -- ffmpeg, normalise to 16 kHz mono MP3."""
    if MP3_PATH.exists() and not force:
        print(f"[2/3]   already converted: {MP3_PATH.name}")
        return MP3_PATH

    print("[2/3]   converting to 16 kHz mono MP3...")
    run([
        ffmpeg, "-y", "-i", str(raw),
        "-ac", str(AUDIO_CHANNELS),
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-b:a", "64k",
        str(MP3_PATH),
    ])
    return MP3_PATH


def step_split(ffmpeg: str, force: bool) -> list[Path]:
    """Step 3 -- ffmpeg, cut into fixed-length pieces."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if force:
        for old in CHUNKS_DIR.glob("chunk_*.mp3"):
            old.unlink()

    if list(CHUNKS_DIR.glob("chunk_*.mp3")):
        print("[3/3]   chunks already exist")
    else:
        minutes = AUDIO_CHUNK_SECONDS / 60
        print(f"[3/3]   splitting into {AUDIO_CHUNK_SECONDS}s ({minutes:.0f} min) pieces...")
        run([
            ffmpeg, "-y", "-i", str(MP3_PATH),
            "-f", "segment",
            "-segment_time", str(AUDIO_CHUNK_SECONDS),
            "-c", "copy",                      # no re-encode: fast and lossless
            str(CHUNKS_DIR / "chunk_%03d.mp3"),
        ])

    return sorted(CHUNKS_DIR.glob("chunk_*.mp3"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="redo steps whose output already exists")
    args = ap.parse_args()

    if not VIDEO_URL:
        sys.exit("ERROR: VIDEO_URL is empty in ingest/config.py")

    print()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_ffmpeg()

    hours = VIDEO_DURATION_SECONDS / 3600
    expected = VIDEO_DURATION_SECONDS // AUDIO_CHUNK_SECONDS + 1
    print(f"        video : {VIDEO_DURATION_SECONDS} s ({hours:.2f} h)")
    print(f"        expect: about {expected} chunks\n")

    raw = step_download(args.force)
    step_convert(raw, ffmpeg, args.force)
    chunks = step_split(ffmpeg, args.force)

    total_mb = sum(c.stat().st_size for c in chunks) / 1024 / 1024
    print(f"\nDone. {len(chunks)} chunks, {total_mb:.1f} MB total.")
    print(f"      {CHUNKS_DIR}")
    if raw.exists():
        raw_mb = raw.stat().st_size / 1024 / 1024
        print(f"\nNote: {raw.name} ({raw_mb:.0f} MB) is the untouched download.")
        print("      It is gitignored and safe to delete once the chunks exist.")
    print("\nNext -- transcribe ONE chunk first, not all of them:")
    print("      uv run python ingest/02_transcribe.py --limit 1\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
