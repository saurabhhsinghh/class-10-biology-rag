# Class 10 Biology Revision Assistant

Ask a question about a 4-hour Hinglish Biology revision lecture and get an answer
drawn **only** from that lecture — with clickable timestamps that jump the video to
the exact moment the topic was explained.

The retrieval is the answer. The timestamp is the proof.

---

## How it works

```
YouTube ─▶ audio ─▶ text + timestamps ─▶ 660 chunks ─▶ vectors ─▶ Qdrant
                        (one-time ingest, on your laptop)

question ─▶ search Qdrant ─▶ 10 excerpts ─▶ Gemini ─▶ answer + timestamps
                        (every question)
```

The transcript holds Hindi in Devanagari and English technical terms in Latin, so a
question typed in Roman Hinglish ("aankh ki working samjhao") matches neither. Each
question is therefore converted — Hindi words to Devanagari, English words left alone —
and **both** forms are searched, keeping the better score per chunk.

Timestamps come from the answer, not from the ranking: the model marks each sentence
with the excerpt it came from, and a timestamp matching no excerpt we sent is dropped
rather than shown.

---

## Running it

**Requirements:** Python 3.11, [uv](https://docs.astral.sh/uv/), `ffmpeg` on PATH
(only needed for re-ingesting), and a Qdrant collection that has already been built.

```bash
uv sync
uv run streamlit run app/app.py
```

Opens on <http://localhost:8501>. First load takes 30–60 s while the embedding model
loads.

### Secrets

Copy `.env.example` to `.env` and fill in four values. `.env` is gitignored — never
commit it.

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | console.groq.com — used for transcription |
| `GEMINI_API_KEY` | aistudio.google.com — writes the answers |
| `QDRANT_URL` | cloud.qdrant.io — free cluster endpoint |
| `QDRANT_API_KEY` | same cluster |

The app needs only `GEMINI_API_KEY`, `QDRANT_URL` and `QDRANT_API_KEY`. `GROQ_API_KEY`
is needed only to re-transcribe audio.

### Rebuilding the index

Only needed if you change the video or the chunking. Run in order; each script is
independently re-runnable and resumes where it stopped.

```bash
uv run python ingest/01_download.py     # yt-dlp + ffmpeg → 24 × 10-min chunks
uv run python ingest/02_transcribe.py   # Groq Whisper, with corrected offsets
uv run python ingest/03_chunk.py        # merge segments → chunks.jsonl
uv run python ingest/04_embed.py        # local embeddings → embeddings.npy
uv run python ingest/05_upload.py       # verify, then push to Qdrant
```

`03_chunk.py` is safe to re-run. `04`/`05` check a content fingerprint before reusing
anything on disk — a stale `embeddings.npy` would otherwise produce confident
timestamps pointing at the wrong minute, with nothing in the UI looking wrong.

---

## Deploying

Hosted on **Streamlit Community Cloud** — free, runs `streamlit run` natively, and
deploys straight from this GitHub repo.

Not Hugging Face Spaces, which is where this was originally headed. HF removed
Streamlit as a built-in SDK on 2025-04-30, so a Streamlit app there needs the Docker
SDK, and Docker Spaces require a paid plan. HF's free tier is Static Spaces, which
serve HTML only — no Python, so none of this can run there.

To deploy: <https://share.streamlit.io> → sign in with GitHub → **Create app** →
Deploy a public app from GitHub.

| Field | Value |
|---|---|
| Repository | `saurabhhsinghh/class-10-biology-revision-rag` |
| Branch | `main` |
| Main file path | `app/app.py` |
| Python version (Advanced settings) | `3.11` |

Then paste the three keys into **Advanced settings → Secrets**, at the top level, in
TOML:

```toml
GEMINI_API_KEY = "..."
QDRANT_URL = "..."
QDRANT_API_KEY = "..."
```

Keep them at the top level — a key nested under a `[section]` is not exposed as an
environment variable, and `rag.py` reads `os.environ`. `app.py` copies them across
from `st.secrets` on startup as a fallback.

Reboot the app after changing secrets; they are read once, at startup.

---

## Checking it still works

```bash
uv run python eval/test_citations.py                 # timestamp parsing, no API calls
uv run python eval/why.py "aankh ki working samjhao" # ranked chunks for one question
uv run python eval/smoke.py                          # 6 questions, end to end
```

`eval/why.py` marks which chunks the model actually receives, which separates a
retrieval failure from a prompt failure — those need opposite fixes.

---

## Known limits

- **One lecture.** Video id, duration and language live in `ingest/config.py`.
- **The Gemini free tier allows 20 requests per model per day, and a question costs
  two** (convert + answer). That is roughly 10 questions a day. Fallback models keep
  it from hard-failing, but this is a demo, not something that can serve a class.
- **Answers are grounded, which means they are sometimes thin.** If the lecture covers
  a topic only in fragments, the answer is fragments — it will not fill the gaps from
  general knowledge, by design.
- Some Whisper transcription artifacts survive in the transcript.
