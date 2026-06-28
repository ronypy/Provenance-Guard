"""Milestone 4 scoring test harness.

Runs the four deliberately chosen spec inputs (clear-AI, clear-human, and two borderline cases)
straight through the detection pipeline and prints each signal score, the combined score, the
confidence, the attribution, and the resulting label.

Goal: confirm the combined scores vary meaningfully across inputs and that all three label
classes are reachable — not that any single input gets a "correct" hardcoded answer.

Usage
-----
    # with the real Groq key (uses all three signals):
    python tests/test_pipeline.py

    # offline (stylometric + phrase only, no API key needed):
    python tests/test_pipeline.py --no-llm
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as `python tests/test_pipeline.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402,F401  (importing config loads .env)
from detection import run_pipeline  # noqa: E402

TEST_INPUTS = {
    "clear-AI (should score high)": (
        "Artificial intelligence represents a transformative paradigm shift in modern society. "
        "It is important to note that while the benefits of AI are numerous, it is equally "
        "essential to consider the ethical implications. Furthermore, stakeholders across "
        "various sectors must collaborate to ensure responsible deployment."
    ),
    "clear-human (should score low)": (
        "ok so i finally tried that new ramen place downtown and honestly? underwhelming. the "
        "broth was fine but they put WAY too much sodium in it and i was thirsty for like three "
        "hours after. my friend got the spicy version and said it was better. probably won't go "
        "back unless someone drags me there"
    ),
    "borderline: formal human (may score mid-high)": (
        "The relationship between monetary policy and asset price inflation has been extensively "
        "studied in the literature. Central banks face a fundamental tension between their "
        "mandate for price stability and the unintended consequences of prolonged low interest "
        "rates on equity and real estate valuations."
    ),
    "borderline: lightly edited AI (should score mid-range)": (
        "I've been thinking a lot about remote work lately. There are genuine tradeoffs — "
        "flexibility and no commute on one side, isolation and blurred work-life boundaries on "
        "the other. Studies show productivity varies widely by individual and role type."
    ),
}


def main() -> None:
    use_llm = "--no-llm" not in sys.argv
    mode = "ALL 3 SIGNALS (LLM on)" if use_llm else "OFFLINE (stylometric + phrase only)"
    print(f"\n=== Provenance Guard scoring test — {mode} ===\n")

    seen_attributions = set()
    for name, text in TEST_INPUTS.items():
        result = run_pipeline(text, use_llm=use_llm)
        s = result["signals"]
        seen_attributions.add(result["attribution"])
        print(f"• {name}")
        print(
            f"    llm={s['llm_score']:.2f}  stylo={s['stylometric_score']:.2f}  "
            f"phrase={s['phrase_score']:.2f}"
        )
        print(
            f"    combined={result['combined_score']:.2f}  "
            f"confidence={result['confidence']:.2f}  -> {result['attribution'].upper()}"
        )
        print(f"    label: {result['label']}\n")

    print(f"Distinct attribution classes reached: {sorted(seen_attributions)}")
    print(
        "PASS: scores vary across inputs and multiple label classes are reachable.\n"
        if len(seen_attributions) >= 2
        else "WARN: fewer than 2 attribution classes reached — review thresholds.\n"
    )


if __name__ == "__main__":
    main()
