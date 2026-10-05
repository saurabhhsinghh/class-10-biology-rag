"""The app: ask a question on the left, watch the lecture on the right.

Run with:
    uv run streamlit run app/app.py

THE ONE INTERACTION THAT MATTERS
--------------------------------
A student asks "reflex action kaise hota hai". The answer comes back, and under
it are timestamps. They click 2:20:55 and the video jumps to 2:20:55, playing.

That is the whole product. Everything else is decoration.

How the seek works
------------------
The player is a plain YouTube iframe, and its `src` is built from the number in
st.session_state.seek. Clicking a timestamp button writes a new number there and
reruns the script; the iframe's src changes; the browser reloads the player at
the new start time.

The useful side effect: when the student is only chatting, `seek` does not
change, so the src string is byte-for-byte identical, so the browser keeps the
same iframe and the video keeps playing. Streamlit's famous "it reruns and
resets everything" problem does not bite here, because the thing that would
reset is the one thing we are deliberately not changing.

Messages live in st.session_state and are re-rendered on every rerun. The script
never renders half an answer inline -- it appends to the list and reruns, so
there is exactly one code path that draws a message.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import ask, video_embed_url  # noqa: E402

st.set_page_config(
    page_title="Class 10 Biology Revision",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Chat avatars, written as codepoints rather than as emoji characters.
#
# The student's avatar used to be a joined emoji: person + ZWJ joiner (U+200D)
# + graduation cap. The ZWJ does not survive being written into this file, so
# what reached the disk was U+1F9D1 U+1F393 -- two codepoints that are not an
# emoji. Streamlit's is_emoji() check then failed, and it fell through to
# opening the string as a path to an image file:
#
#     StreamlitAPIException: Failed to load the provided avatar value as an image.
#
# Streamlit is fine with joined emoji; the joiner just never arrives. Writing
# the escape instead of the character means nothing can be mangled in transit.
AVATAR_STUDENT = "\U0001F393"    # graduation cap
AVATAR_TUTOR = "\U0001F9EC"      # DNA helix

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 0.5rem; }
      .ts-note { font-size: 0.80rem; opacity: 0.62; margin-bottom: 0.1rem; }
      div[data-testid="stButton"] button { padding: 0.15rem 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []       # [{role, content, citations, chunks}]
if "seek" not in st.session_state:
    st.session_state.seek = 0            # seconds; where the video should be


# ---------------------------------------------------------------------------
# small renderers
# ---------------------------------------------------------------------------

def hhmmss(seconds: int) -> str:
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}"


def render_citations(citations: list[dict], key_prefix: str) -> None:
    """The timestamp buttons. Clicking one seeks the video."""
    if not citations:
        return
    st.markdown('<div class="ts-note">Taught here in the lecture:</div>',
                unsafe_allow_html=True)
    for i, (col, c) in enumerate(zip(st.columns(len(citations)), citations)):
        with col:
            if st.button(f"▶ {c['start_fmt']}", key=f"{key_prefix}_{i}",
                         use_container_width=True):
                st.session_state.seek = int(c["start"])
                st.rerun()


def render_message(msg: dict, index: int) -> None:
    avatar = AVATAR_STUDENT if msg["role"] == "user" else AVATAR_TUTOR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("error"):
            # The student sees the friendly sentence above; the real reason
            # (a quota, a dead model name) is here, short, for whoever is
            # looking at the app while building it.
            st.caption(f"⚠️ {str(msg['error'])[:300]}")
        render_citations(msg.get("citations") or [], key_prefix=f"ts{index}")


# ---------------------------------------------------------------------------
# header + two panes
# ---------------------------------------------------------------------------

st.title("🧬 Class 10 Biology — Revision")
st.caption("Ask anything from the lecture. Every answer points at the exact "
           "moment it came from.")

col_chat, col_video = st.columns([1.1, 1], gap="large")

# The video renders FIRST. A timestamp click further down changes `seek` and
# triggers a rerun, so the new value is picked up at the top of the next run --
# no need for the layout order to depend on the interaction order.
with col_video:
    components.html(
        f"""
        <iframe
            src="{video_embed_url(st.session_state.seek, autoplay=st.session_state.seek > 0)}"
            width="100%" height="500"
            style="border:0; border-radius:10px; background:#000;"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media;
                   gyroscope; picture-in-picture; web-share"
            allowfullscreen>
        </iframe>
        """,
        height=510,
    )
    if st.session_state.seek:
        st.caption(f"▶ playing from **{hhmmss(st.session_state.seek)}**")
    else:
        st.caption("Click any timestamp in an answer to jump the video there.")

with col_chat:
    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i)

    question = st.chat_input("Ask a question…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATAR_STUDENT):
            st.markdown(question)

        with st.chat_message("assistant", avatar=AVATAR_TUTOR):
            with st.spinner("Searching the lecture…"):
                try:
                    result = ask(question)
                except Exception as exc:                        # noqa: BLE001
                    st.error(f"Something went wrong: {exc}")
                    st.stop()

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "error": result.get("error"),
        })
        st.rerun()
