"""Phase 4 -- turn each chunk into a list of numbers (an embedding).

Run with:
    uv run python ingest/04_embed.py
    uv run python ingest/04_embed.py --query "photosynthesis kya hota hai"

What an embedding is
--------------------
The model reads a piece of text and outputs 384 numbers. Those numbers place
the text at a point in 384-dimensional space, and the ONLY thing that matters
is that similar meanings land close together. "photosynthesis kya hai" and
"पौधे अपना खाना कैसे बनाते हैं" end up near each other even though they share
no characters at all.

That is what makes search possible without keyword matching: we compare
positions, not words.

Why ONE index, and what happens to the question instead
-------------------------------------------------------
A student types Roman Hinglish ("aankh ki working samjhao"). The transcript is
Devanagari, because Whisper wrote down what it heard. This model handles each
script well on its own and poorly across the gap:

    Devanagari chunk  +  Devanagari question     0.721
    Devanagari chunk  +  Roman question          0.420

An early attempt built a SECOND index, from a Romanised copy of the transcript,
so a Roman question would have a Roman index to match against. It is gone, and
the reason is worth keeping. Whisper had already written "support" as सपोर्ट,
which throws the English spelling away -- Devanagari records sounds, not
spellings. Romanising it back produced "saport", a word no student types. The
information died at transcription time and a second index cannot bring it back.

What replaced it moves the QUESTION instead, because then the transcript never
has to be touched at all: app/rag.py converts the question's Hindi words into
Devanagari and leaves its English words alone, then searches this one index
with both forms and keeps the better match per chunk.

Output
------
    data/processed/embeddings.npy        float32, (n_chunks, 384)
    data/processed/embeddings.meta.json  which chunks.jsonl built it

Row i == the chunk with id i. That contract is the whole point of the file: the
.jsonl and the .npy are two halves of one table joined by row number, so
reorder either and every answer points at the wrong timestamp. The join is
asserted before anything is written, and the sidecar records the hash of the
chunks so a stale .npy cannot be quietly reused later -- see _fingerprint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# This terminal defaults to cp1252, which cannot print Devanagari -- and the
# whole point of this script is to show you Devanagari chunks coming back.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (  # noqa: E402
    CHUNKS_FILE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDINGS_META,
    PROCESSED_DIR,
    embeddings_mismatch,
    use_local_hf_cache,
    write_embeddings_meta,
)

EMBEDDINGS = PROCESSED_DIR / "embeddings.npy"

# Build artifacts of the two-index version. Nothing reads them any more; they
# are only mentioned so their presence on disk is not a mystery.
_STALE = [PROCESSED_DIR / "embeddings_roman.npy",
          PROCESSED_DIR / "embeddings_dev.npy"]

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def load_model():
    """Load the embedding model. Downloads ~470 MB the first time, then cached."""
    # HF_HOME must be set BEFORE sentence_transformers is imported.
    use_local_hf_cache()

    from sentence_transformers import SentenceTransformer

    print(f"        cache    : {os.environ.get('HF_HOME') or '(hugging face default)'}")
    print(f"        loading  : {EMBEDDING_MODEL} ...")

    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    elapsed = time.time() - t0

    # get_embedding_dimension() is the 6.x name; keep the old one as a fallback
    # so this does not break on an older sentence-transformers.
    get_dim = getattr(model, "get_embedding_dimension", None) \
        or model.get_sentence_embedding_dimension
    dim = get_dim()
    max_seq = getattr(model, "max_seq_length", None)

    print(f"        loaded   : in {elapsed:.1f}s  (dim {dim}, max_seq_length {max_seq})")

    if dim != EMBEDDING_DIM:
        print(f"\n  !! config.py says EMBEDDING_DIM = {EMBEDDING_DIM}, "
              f"but the model says {dim}.")
        print("     Fix config.py before uploading to Qdrant, or the collection "
              "will be created with the wrong vector size.\n")

    return model


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        sys.exit(f"ERROR: {CHUNKS_FILE} not found. Run 03_chunk.py first.")

    chunks = [json.loads(line)
              for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not chunks:
        sys.exit(f"ERROR: {CHUNKS_FILE} is empty.")

    # The row-order contract. If ids are not exactly 0..n-1 in order, then
    # "row i == chunk i" is a lie and every timestamp downstream is wrong.
    for i, c in enumerate(chunks):
        if c["id"] != i:
            sys.exit(f"ERROR: chunks.jsonl is out of order at line {i}: "
                     f"expected id {i}, found {c['id']}. Re-run 03_chunk.py.")

    if "text" not in chunks[0]:
        sys.exit("ERROR: chunks.jsonl has no 'text' field -- it was built by an "
                 "older version of 03_chunk.py. Re-run 03_chunk.py.")

    print(f"        chunks   : {len(chunks)}")
    return chunks


# ---------------------------------------------------------------------------
# the interesting bit -- seeing it work
# ---------------------------------------------------------------------------

def show_retrieval(model, chunks, index, queries: list[str]) -> None:
    """Embed each query and show what the index returns.

    This is the first time the project actually DOES anything. Everything
    before this was preparation.

    Note what these queries are NOT: they are not what a student types. A
    student's Hindi words arrive in Roman script and are converted first (see
    app/rag.py). What is measured here is the index on its own, so the numbers
    mean "how good is the transcript index", with the question's script held
    constant.
    """
    for q in queries:
        qv = model.encode([q], normalize_embeddings=True)[0]
        sims = index @ qv
        top = sims.argsort()[::-1][:5]

        print(f"\n  Q: {q}")
        print(f"     {'':<4}{'score':>8}   chunk")
        for rank, i in enumerate(top, 1):
            c = chunks[i]
            print(f"     {rank}.  {float(sims[i]):>8.3f}   "
                  f"[{c['start_fmt']}] {c['text'][:52]}")


def self_check(model, chunks, index) -> bool:
    """Sanity check the plumbing without needing to know the video's content.

    Take a chunk's own text as the query. If the pipeline is wired correctly it
    must come back as its own best match -- a score of ~1.0, at rank 1. If it
    does not, the rows and the chunks have drifted out of alignment.
    """
    probe_i = len(chunks) // 2
    qv = model.encode([chunks[probe_i]["text"]], normalize_embeddings=True)[0]
    sims = index @ qv
    best = int(sims.argmax())
    good = best == probe_i
    print(f"  self-check : query = chunk #{probe_i} own text")
    print(f"               best match = chunk #{best}  (score {sims[best]:.4f})  "
          f"{'PASS' if good else 'FAIL -- alignment broken'}")
    return good


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query", action="append", default=[],
                    help="a question to try after embedding (repeatable). "
                         "With no --query, a self-check runs instead.")
    ap.add_argument("--force", action="store_true",
                    help="re-embed even if the .npy file already exists")
    args = ap.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print()
    stale = [p for p in _STALE if p.exists()]
    if stale:
        print("        note     : the old two-index build is still on disk and is "
              "no longer read: " + ", ".join(p.name for p in stale))
        print("                   Safe to delete. Rebuilt by the version before "
              "this one.\n")

    # Chunks first: it is cheap, and nothing below makes sense without them.
    chunks = load_chunks()

    # A .npy is only usable next to the exact chunks.jsonl it came from, and
    # nothing inside it says which that was. So "does the file exist" is the
    # wrong question -- ask instead whether it was built from THESE chunks.
    why = embeddings_mismatch(chunks) if EMBEDDINGS.exists() else "not built yet"
    fresh = why is None
    if EMBEDDINGS.exists() and not fresh:
        print(f"        ignoring : the .npy on disk -- {why}. Rebuilding it.")

    if fresh and not args.force and not args.query:
        print(f"        {EMBEDDINGS.name} is up to date -- skipping.")
        print("        Use --force to rebuild it, or pass --query to search what "
              "is already there.\n")
        return 0

    model = load_model()

    import numpy as np

    reuse = fresh and not args.force
    if reuse:
        # --query against vectors that are already on disk: no reason to spend
        # a minute re-encoding identical rows.
        print("        reusing  : the .npy file on disk (pass --force to rebuild)")
        index = np.load(EMBEDDINGS)
    else:
        texts = [c["text"] for c in chunks]
        print(f"        encoding : {len(texts)} chunks, batch {BATCH_SIZE}...")
        t0 = time.time()
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,  # unit length, so cosine == dot product
            show_progress_bar=False,
        )
        elapsed = time.time() - t0
        index = np.asarray(vectors, dtype="float32")
        print(f"        done     : {elapsed:.1f}s  "
              f"({elapsed / len(texts) * 1000:.0f} ms/chunk)")

    # ---- checks, then write ------------------------------------------------
    if not reuse:
        print()
    norms = np.linalg.norm(index, axis=1)
    checks = [
        ("shape matches", index.shape == (len(chunks), EMBEDDING_DIM)),
        ("no NaN / inf", bool(np.isfinite(index).all())),
        ("all unit length", bool(np.allclose(norms, 1.0, atol=1e-4))),
    ]
    if not reuse:
        print(f"  {index.shape}  expected ({len(chunks)}, {EMBEDDING_DIM})")
        for label, ok in checks:
            print(f"        {label:<18}: {'OK' if ok else 'FAILED'}")

    if not all(ok for _, ok in checks):
        sys.exit("\nERROR: embedding checks failed. Re-run with --force; if it "
                 "repeats, the model name is suspect.")

    if not reuse:
        np.save(EMBEDDINGS, index)
        write_embeddings_meta(chunks)
        print(f"        wrote {EMBEDDINGS.name}  "
              f"({EMBEDDINGS.stat().st_size / 1024 / 1024:.1f} MB)")
        print(f"              {EMBEDDINGS_META.name}  (which chunks built it)")

    print()
    ok = self_check(model, chunks, index)
    if args.query:
        show_retrieval(model, chunks, index, args.query)

    print("\nNext -- push the chunks and their vectors into Qdrant:")
    print("      uv run python ingest/05_upload.py\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
