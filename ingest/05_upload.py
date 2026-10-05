"""Phase 5 -- push the chunks and their vectors into Qdrant.

Run with:
    uv run python ingest/05_upload.py
    uv run python ingest/05_upload.py --keep     # don't wipe an existing collection

Why Qdrant at all, for a few hundred chunks
-------------------------------------------
Honestly: numpy could do this. A few hundred rows of 384 floats is about a
megabyte, and a dot product over them takes under a millisecond. Qdrant earns
its place when the corpus grows (a second video, a whole chapter series) or
when the app has to run somewhere that should not also be loading a 470 MB
embedding model just to hold its own copy of the index. It also lets the
deployed app search without shipping the .npy file at all.

Why one collection with one unnamed vector
------------------------------------------
The chunk is the unit of meaning, so it should be the unit of storage: a point
holds the chunk's text, its timestamps, and its vector, together. An earlier
version kept two vectors per point ("roman" and "dev") and searched them by
name; the romanisation experiment needed that and nothing else ever did. One
vector means a search needs no `using=` argument and there is no way to search
the wrong half by accident.

Output
------
    Qdrant collection named by COLLECTION_NAME in config.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (  # noqa: E402
    CHUNKS_FILE,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    PROCESSED_DIR,
    VIDEO_ID,
    embeddings_mismatch,
    use_local_hf_cache,
)

EMBEDDINGS = PROCESSED_DIR / "embeddings.npy"
UPSERT_BATCH = 128


def load_all():
    """Read chunks + vectors and prove they line up before uploading anything."""
    import numpy as np

    if not CHUNKS_FILE.exists():
        sys.exit(f"ERROR: {CHUNKS_FILE} not found. Run 03_chunk.py first.")
    if not EMBEDDINGS.exists():
        sys.exit(f"ERROR: {EMBEDDINGS.name} not found. Run 04_embed.py first.")

    chunks = [json.loads(line)
              for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    vectors = np.load(EMBEDDINGS)

    # The row-order contract, one last time. Uploading mismatched rows produces
    # a collection whose answers point at the wrong minute, and it looks
    # completely plausible in the UI -- which is exactly why it is checked here
    # rather than noticed later.
    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        sys.exit(f"ERROR: embeddings has shape {vectors.shape}, expected "
                 f"({len(chunks)}, {EMBEDDING_DIM}). Re-run 04_embed.py.")
    for i, c in enumerate(chunks):
        if c["id"] != i:
            sys.exit(f"ERROR: chunks.jsonl out of order at line {i}. "
                     f"Re-run 03_chunk.py.")

    # Shape alone proves nothing: a .npy from an older chunking can match on
    # rows and still hold vectors for entirely different text. The sidecar
    # knows which chunks built it, so ask that instead of guessing.
    why = embeddings_mismatch(chunks)
    if why:
        sys.exit(f"ERROR: embeddings.npy does not go with chunks.jsonl --\n"
                 f"  {why}.\n"
                 f"  Uploading it would put a confident, plausible timestamp on "
                 f"the wrong minute of a four-hour lecture, and nothing in the "
                 f"UI would look wrong.\n"
                 f"  Re-run 04_embed.py, then this script.")

    print(f"        chunks   : {len(chunks)}")
    print(f"        vectors  : {vectors.shape}")
    return chunks, vectors


def main() -> int:
    import numpy as np
    from qdrant_client import QdrantClient, models

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true",
                    help="do not delete an existing collection first")
    args = ap.parse_args()

    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        sys.exit("ERROR: QDRANT_URL / QDRANT_API_KEY are not set in .env")

    print()
    chunks, vectors = load_all()

    client = QdrantClient(url=url, api_key=api_key, timeout=60)

    exists = client.collection_exists(COLLECTION_NAME)
    if exists and not args.keep:
        # Wipe and rebuild rather than upsert over the top. A half-finished
        # earlier upload would otherwise leave stale points behind pointing at
        # timestamps that no longer exist, and nothing downstream could tell.
        print(f"        dropping : existing '{COLLECTION_NAME}' collection")
        client.delete_collection(COLLECTION_NAME)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"        created  : '{COLLECTION_NAME}' "
              f"(one vector, {EMBEDDING_DIM} dims, cosine)")

    # ---- upload ------------------------------------------------------------
    print(f"        uploading: {len(chunks)} points in batches of {UPSERT_BATCH}...")
    t0 = time.time()
    for start in range(0, len(chunks), UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, len(chunks))
        points = []
        for i in range(start, end):
            c = chunks[i]
            points.append(models.PointStruct(
                id=i,                     # == the chunk id == the row number
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": c["id"],
                    "text": c["text"],            # Devanagari, faithful
                    "start": c["start"],
                    "end": c["end"],
                    "start_fmt": c["start_fmt"],
                    "end_fmt": c["end_fmt"],
                    "video_id": VIDEO_ID,
                },
            ))
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        print(f"        {end:>4}/{len(chunks)}", end="\r")
    print(f"        uploaded : {len(chunks)} points in {time.time() - t0:.1f}s")

    # ---- verify, by asking it a real question ------------------------------
    # Uploading is not the same as uploading CORRECTLY. The scores below come
    # back from Qdrant over the network, and the `local` column is the same
    # query run against the .npy file on disk. Those two must agree to three
    # decimals: if they do, the round trip preserved both the vectors and their
    # row alignment, and any later disagreement is the app's fault, not the
    # upload's.
    info = client.get_collection(COLLECTION_NAME)
    print(f"        in qdrant: {info.points_count} points")
    if info.points_count != len(chunks):
        sys.exit(f"\nERROR: Qdrant holds {info.points_count} points but we sent "
                 f"{len(chunks)}. Re-run without --keep.")

    use_local_hf_cache()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)

    question = "human eye ke parts batao"
    qv = model.encode([question], normalize_embeddings=True)[0]
    local = vectors @ qv.astype("float32")

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qv.tolist(),
        limit=6,
        with_payload=True,
    ).points

    print(f"\n  verify     : '{question}'")
    print(f"  {'':<4}{'qdrant':>8}  {'local':>7}   chunk")
    mismatch = 0
    for rank, p in enumerate(hits, 1):
        same = abs(p.score - float(local[p.id])) < 1e-3
        mismatch += 0 if same else 1
        print(f"  {rank}.  {p.score:>8.3f}  {local[p.id]:>7.3f}   "
              f"[{p.payload['start_fmt']}] {p.payload['text'][:44]}")

    if mismatch:
        print(f"\n  !! {mismatch} score(s) disagree with the local .npy -- the "
              f"upload and the file are not the same vectors.")
        return 1

    print("\n        qdrant and the local .npy agree on every score  -- OK")
    print(f"\nDone. Collection '{COLLECTION_NAME}' is ready for the app.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
