"""The citation parser, on its own. Fast -- no model, no network, no Qdrant.

    uv run python eval/test_citations.py

This part broke once and the failure was invisible: the timestamps were in the
answer, the buttons just silently did not appear. "Silently empty" is the worst
kind of bug to leave uncovered, so it gets a test.

Two shapes matter. The transcript writes "4:49" when the hour is zero and
"2:20:55" when it is not. The first version of the parser only understood the
second, so every citation from the first half of the lecture vanished.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from rag import _to_seconds, citations_from_answer  # noqa: E402

CHUNKS = [
    {"start_fmt": "4:49", "start": 289},
    {"start_fmt": "8:13", "start": 493},
    {"start_fmt": "13:05", "start": 785},
    {"start_fmt": "48:09", "start": 2889},
    {"start_fmt": "2:20:55", "start": 8455},
    {"start_fmt": "3:33:53", "start": 12833},
]

failures = 0


def check(label: str, got, want) -> None:
    global failures
    if got == want:
        print(f"  ok    {label}")
    else:
        failures += 1
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")


print("_to_seconds -- both timestamp shapes")
check("4:49 is minutes:seconds", _to_seconds("4:49"), 289)
check("2:20:55 is h:m:s", _to_seconds("2:20:55"), 8455)
check("48:09 does not overflow", _to_seconds("48:09"), 2889)
check("0:00", _to_seconds("0:00"), 0)
check("prose is not a time", _to_seconds("hello"), None)
check("four fields is not a time", _to_seconds("1:2:3:4"), None)

print("\ncitations_from_answer")
check(
    "both shapes found, in the order the answer used them",
    [c["start_fmt"] for c in citations_from_answer(
        "Light cornea se aati hai [4:49]. Aqueous humor [8:13]. "
        "Hum 180 degree dekhte hain [13:05].", CHUNKS)],
    ["4:49", "8:13", "13:05"],
)
check(
    "the h:m:s shape works too",
    [c["start_fmt"] for c in citations_from_answer(
        "Reflex action [2:20:55]. Food chain [3:33:53].", CHUNKS)],
    ["2:20:55", "3:33:53"],
)
check(
    "an invented timestamp is dropped, not shown",
    [c["start_fmt"] for c in citations_from_answer(
        "Real [4:49]. Invented [9:99:99].", CHUNKS)],
    ["4:49"],
)
check(
    "a timestamp we never sent is dropped",
    [c["start_fmt"] for c in citations_from_answer(
        "Made up [1:11:11].", CHUNKS)],
    [],
)
check(
    "at most MAX_CITATIONS buttons",
    len(citations_from_answer(
        "[4:49] [8:13] [13:05] [48:09] [2:20:55] [3:33:53]", CHUNKS)),
    3,
)
check(
    "a bare timestamp in prose is not a citation",
    [c["start_fmt"] for c in citations_from_answer(
        "Ye 4:49 par aata hai, bina bracket ke.", CHUNKS)],
    [],
)
check(
    "no brackets at all is an empty list, not a crash",
    citations_from_answer("Koi timestamp nahi.", CHUNKS),
    [],
)

print()
if failures:
    print(f"{failures} FAILED")
    sys.exit(1)
print("all passed")
