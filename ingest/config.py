"""Central configuration for the bio-rag pipeline.

Every tunable lives here, in one place. You will re-run the ingest scripts and
tweak these values many times during evaluation (Phase 9), and hunting a magic
number across five scripts is misery.

Values marked [VERIFIED] were confirmed by actually running them on this machine.
Values marked [TUNE] are sensible starting points -- expect to adjust them once
you have the eval set and can measure the effect.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"          # downloaded + split audio   (gitignored)
RAW_DIR = DATA_DIR / "raw"              # raw Whisper JSON per chunk (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"  # final chunks               (committed)

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

# ---------------------------------------------------------------------------
# Source video
# ---------------------------------------------------------------------------

# [VERIFIED 2026-10-03] "Class 10th Complete Biology in One Shot | Concept with
# Questions by Ashu Sir" -- channel: Science and Fun Education.
# Duration 3:53:30 (14,010 seconds). Confirmed via `yt-dlp --skip-download`.
#
# NOTE: this video has NO chapter markers. Chapter-based navigation would need
# topic boundaries auto-detected from the transcript -- a v2 feature, not v1.
VIDEO_URL = "https://www.youtube.com/watch?v=C_L_o8fI2qw"

# The 11-character video id -- needed for the embedded player and seek links.
VIDEO_ID = "C_L_o8fI2qw"

# Used to sanity-check the download and report progress.
VIDEO_DURATION_SECONDS = 14_010

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 16_000   # what Whisper wants; also keeps the file small
AUDIO_CHANNELS = 1           # mono
AUDIO_CHUNK_SECONDS = 600    # 10 minutes per piece (Groq's upload limit is 25 MB)

# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

WHISPER_LANGUAGE = "hi"      # force Hindi -- do not let it auto-detect
WHISPER_MODEL = "whisper-large-v3"        # Groq
WHISPER_LOCAL_MODEL = "small"             # faster-whisper fallback
WHISPER_LOCAL_COMPUTE = "int8"            # CPU-friendly

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Where Hugging Face keeps downloaded model files, so the ~470 MB model is
# fetched once and reused forever after. If this drive does not exist (on a
# Linux box or Hugging Face Spaces), the scripts silently fall back to the
# default cache location -- nothing breaks.
HF_CACHE_DIR = Path("E:/hf-cache")


def use_local_hf_cache() -> None:
    """Point Hugging Face at HF_CACHE_DIR.

    Must be called BEFORE importing transformers or sentence_transformers --
    they read HF_HOME once, at import.

    Skips if HF_HOME is already set (deployment sets its own), and skips if the
    drive does not exist, so this stays harmless on a Linux box or on Hugging
    Face Spaces where there is no E: drive.
    """
    import os

    if os.environ.get("HF_HOME"):
        return
    if HF_CACHE_DIR.parent.exists():
        os.environ["HF_HOME"] = str(HF_CACHE_DIR)

# [VERIFIED 2026-10-03] Confirmed by loading the model on this machine.
# The model SILENTLY TRUNCATES anything longer than MAX_SEQ_LENGTH -- no warning,
# no error. This is the single easiest way to silently wreck retrieval quality.
MAX_SEQ_LENGTH = 128
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# The chunks <-> embeddings contract
# ---------------------------------------------------------------------------
#
# embeddings.npy and chunks.jsonl are two halves of ONE table, joined by row
# number: row i is the vector for chunk i. Nothing inside a .npy records that,
# so a .npy left over from an older chunks.jsonl looks perfectly valid -- right
# shape, right dtype, no error -- while every row now belongs to a different
# chunk. Search still returns results. Every timestamp is just wrong.
#
# That is not hypothetical. On 2026-10-04 a stale embeddings.npy from an older
# chunking was on disk; 04_embed.py said "already exists -- skipping"; and those
# old vectors were uploaded against the new chunks. Both files had 660 rows, so
# nothing failed anywhere, and 05_upload.py's verification agreed -- it compares
# Qdrant against that same file. It would have shipped.
#
# So the .npy carries a sidecar recording WHAT built it, and both scripts ask
# this function whether the two still belong together.

EMBEDDINGS_META = PROCESSED_DIR / "embeddings.meta.json"


def chunks_fingerprint(chunks: list[dict]) -> str:
    """SHA-256 over the chunk ids and texts, in order.

    Text only, not timestamps: the text is what got embedded, so it is what
    decides whether an existing .npy still describes these chunks.
    """
    import hashlib

    h = hashlib.sha256()
    for c in chunks:
        h.update(str(c["id"]).encode("ascii"))
        h.update(b"\x1f")
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def read_embeddings_meta() -> dict:
    try:
        import json
        return json.loads(EMBEDDINGS_META.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_embeddings_meta(chunks: list[dict]) -> None:
    """Record which chunks and which model produced the .npy on disk."""
    import json

    EMBEDDINGS_META.write_text(json.dumps({
        "chunks_sha256": chunks_fingerprint(chunks),
        "chunks": len(chunks),
        "model": EMBEDDING_MODEL,
        "dim": EMBEDDING_DIM,
    }, indent=1), encoding="utf-8")


def embeddings_mismatch(chunks: list[dict]) -> str | None:
    """Why the .npy on disk does NOT belong to these chunks, or None if it does.

    Returning a reason rather than a bool, because "the vectors are stale" and
    "the model changed" call for different fixes and a bare False hides which.
    """
    meta = read_embeddings_meta()
    if not meta:
        return ("no embeddings.meta.json beside it, so there is no way to tell "
                "which chunks built it")
    if meta.get("chunks_sha256") != chunks_fingerprint(chunks):
        return "it was built from a different chunks.jsonl"
    if meta.get("model") != EMBEDDING_MODEL:
        return (f"it was built with {meta.get('model')}, "
                f"but config now says {EMBEDDING_MODEL}")
    if meta.get("dim") != EMBEDDING_DIM:
        return f"it was built at {meta.get('dim')} dims, not {EMBEDDING_DIM}"
    return None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# TARGET_TOKENS is deliberately ~78% of MAX_SEQ_LENGTH. The tokenizer used for
# chunking is not necessarily the same one the embedding model uses, so leave
# headroom rather than cutting it fine at 128.
TARGET_TOKENS = 100
OVERLAP_TOKENS = 30          # so a concept spanning a boundary isn't lost entirely

# Hard stop. TARGET_TOKENS is the GOAL, but a single long segment can overshoot
# it -- if we are at 95 tokens and the next segment is 79 tokens, we would land
# at 174, past MAX_SEQ_LENGTH, and the embedding model would silently truncate
# the tail. This ceiling wins over the target so that never happens.
CHUNK_TOKEN_CEILING = 120

# Whisper hallucinates a speck of text on non-speech audio -- music, applause,
# a break. The source video has a ~12-minute break at 2:29 where it emitted one
# nonsense word (झाल) every 30 seconds. Such a speck is always SHORT and always
# followed by a LONG silence; real teaching is neither. Chunk #440 was built
# entirely out of this junk before the filter existed.
BLIP_MAX_WORDS = 4           # a "segment" this short...
BLIP_GAP_SECONDS = 20.0      # ...followed by silence this long is not speech

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K_SEARCH = 15            # how many chunks come back from the vector search
LLM_EXCERPTS = 10            # how many of those are handed to the model

# [MEASURED 2026-10-04] Below this score we call the question not-covered.
#
# This is a FENCE, not the real guard. Six questions measured against the
# current pipeline:
#
#     "cricket ka score kya tha"     NOT in the lecture   0.313
#     "food chain kya hoti hai"      in the lecture       0.584
#     "aankh ki working samjhao"     in the lecture       0.735
#     "neuron kya hota hai"          in the lecture       0.766
#     "hormone kya kaam karta hai"   in the lecture       0.774
#     "reflex action kaise hota hai" in the lecture       0.779
#
# A gap of 0.27 between the off-topic question and the weakest genuine one --
# which LOOKS like a cutoff, and today a cutoff around 0.45 would score well on
# these six. It is not yet a cutoff, because six questions is not a population
# and one of them is off-topic. eval/run_eval.py exists to measure the real
# spread; until it has, the value stays low.
#
# What actually stops a bad answer is the model reading the excerpts and
# refusing (prompts.SYSTEM_PROMPT, rule 2). Cricket scored 0.313, cleared this
# fence, went to the model, and came back refused -- which is the design
# working, and the reason the fence does not need to be clever.
#
# So this sits below even the weakest genuine score and catches only outright
# nonsense. Refusing a question the lecture DOES cover tells a student their own
# syllabus does not exist; that is the worse failure of the two.
SCORE_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

# [VERIFIED 2026-10-03] "gemini-2.5-flash" now returns 404 for new API keys.
# Listed live with: client.models.list()
LLM_MODEL = "gemini-3.8-flash"

# Tried in order if the model above errors. Every name here was confirmed live
# in one run, so a dead name is a stale name -- re-check with models.list().
#
# Length matters for a second reason. The free tier allows 20 requests per
# MODEL per day, and this app makes two calls per question (one to convert the
# search form, one to answer). So this list's length is roughly the app's daily
# question budget divided by two.
#
# [MEASURED 2026-10-04] After a morning of testing, every model below returned
# 429 with "retry in 4h29m" -- the whole day's free quota, spent. That is fine
# for building the thing and hopeless for showing it to students. Add billing,
# or add models to this list, before that happens.
LLM_FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
LLM_RETRIES = 2                  # attempts per model before moving on
LLM_RETRY_SLEEP = 1.5            # seconds between attempts

TEMPERATURE = 0.1                # low -- we explicitly do not want creative drift
MAX_CITATIONS = 3                # timestamps shown per answer

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

COLLECTION_NAME = "bio_rag"
