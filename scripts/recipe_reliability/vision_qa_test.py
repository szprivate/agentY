"""Vision-QA test loop (a simple RL-style harness for the QA agent).

The vision-QA agent (``src.executor._vision_qa``) judges whether a generated
image matches the user's intent. This harness tests that judgement in isolation
- no ComfyUI, no diffusion models - by feeding it controlled (intent, image[,
input image]) cases with a KNOWN ground-truth verdict, then scoring how often
its verdict is correct.

Reward = fraction of correct verdicts (and low false-pass / false-fail rates).
When the agent misjudges, tune ``config/system_prompts/system_prompt.qaChecker.md``
(or the ``_vision_qa`` code) and re-run until it is reliable.

Run:
    python -m scripts.recipe_reliability.vision_qa_test
    python -m scripts.recipe_reliability.vision_qa_test --limit 1   # validate setup
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)
load_dotenv(os.path.join(_root, ".env"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from src.executor import _vision_qa  # noqa: E402

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_qa_assets")
_REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_qa_report.json")


def _font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _gen_images() -> dict[str, str]:
    """Generate controlled, unambiguous test images. Returns {name: path}."""
    from PIL import Image, ImageDraw
    os.makedirs(_ASSETS, exist_ok=True)

    def make(name: str, bg, draw=None) -> str:
        p = os.path.join(_ASSETS, name)
        img = Image.new("RGB", (768, 768), bg)
        if draw:
            draw(ImageDraw.Draw(img))
        img.save(p)
        return p

    return {
        "red_square": make("red_square.png", (220, 30, 30)),
        "green_square": make("green_square.png", (30, 180, 60)),
        "blue_circle": make("blue_circle.png", (255, 255, 255),
                            lambda d: d.ellipse([170, 170, 598, 598], fill=(40, 80, 220))),
        "red_yellow_circle": make("red_yellow_circle.png", (220, 30, 30),
                                  lambda d: d.ellipse([240, 240, 528, 528], fill=(245, 220, 40))),
        "text_hello": make("text_hello.png", (255, 255, 255),
                           lambda d: d.text((150, 300), "HELLO", fill=(0, 0, 0), font=_font(180))),
    }


def _cases(img: dict[str, str]) -> list[dict]:
    """(name, mode, intent, output, inputs, expected verdict)."""
    return [
        # ---- Generation: does the agent USE the intent? (same output, diff intent) ----
        {"name": "gen_red_square_match", "intent": "a solid red square filling the whole frame",
         "out": img["red_square"], "inputs": [], "expect": "PASS"},
        {"name": "gen_red_square_vs_dog", "intent": "a photograph of a dog in a park",
         "out": img["red_square"], "inputs": [], "expect": "FAIL"},
        {"name": "gen_blue_circle_match", "intent": "a blue circle on a white background",
         "out": img["blue_circle"], "inputs": [], "expect": "PASS"},
        {"name": "gen_blue_circle_vs_red_square", "intent": "a solid red square",
         "out": img["blue_circle"], "inputs": [], "expect": "FAIL"},
        {"name": "gen_text_hello_match", "intent": "an image with the word HELLO written large",
         "out": img["text_hello"], "inputs": [], "expect": "PASS"},
        # ---- Edit: does the agent receive input vs output and judge the change? ----
        {"name": "edit_recolor_green_ok", "intent": "change the color of this square to green",
         "out": img["green_square"], "inputs": [img["red_square"]], "expect": "PASS"},
        {"name": "edit_recolor_green_unchanged", "intent": "change the color of this square to green",
         "out": img["red_square"], "inputs": [img["red_square"]], "expect": "FAIL"},
        {"name": "edit_add_yellow_circle_ok", "intent": "add a yellow circle in the center",
         "out": img["red_yellow_circle"], "inputs": [img["red_square"]], "expect": "PASS"},
        # ---- Harder tier: adversarial cases that probe real discrimination ----
        # Right shape, wrong colour -> must FAIL (not just "there is a circle").
        {"name": "gen_shape_ok_colour_wrong", "intent": "a green circle on a white background",
         "out": img["blue_circle"], "inputs": [], "expect": "FAIL"},
        # Right colour, wrong shape -> must FAIL (not just "there is blue").
        {"name": "gen_colour_ok_shape_wrong", "intent": "a solid blue square filling the frame",
         "out": img["blue_circle"], "inputs": [], "expect": "FAIL"},
        # Compound intent, only half delivered -> must FAIL (missing element).
        {"name": "gen_compound_partial", "intent": "a red square next to a blue circle",
         "out": img["red_square"], "inputs": [], "expect": "FAIL"},
        # Compound intent fully delivered -> PASS (red field with a coloured circle inside).
        {"name": "gen_compound_match", "intent": "a red image with a round shape in the center",
         "out": img["red_yellow_circle"], "inputs": [], "expect": "PASS"},
        # Edit happened but hit the WRONG target colour -> must FAIL.
        {"name": "edit_wrong_target_colour", "intent": "change the color of this square to blue",
         "out": img["green_square"], "inputs": [img["red_square"]], "expect": "FAIL"},
        # Edit asked to add a circle but output is unchanged -> must FAIL.
        {"name": "edit_add_circle_unchanged", "intent": "add a yellow circle in the center",
         "out": img["red_square"], "inputs": [img["red_square"]], "expect": "FAIL"},
    ]


def _parse(verdict: str) -> str:
    v = (verdict or "").upper()
    m = re.search(r"OVERALL[^A-Z]*?(PASS|FAIL)", v)
    if m:
        return m.group(1)
    toks = re.findall(r"\b(PASS|FAIL)\b", v)
    return toks[-1] if toks else "UNKNOWN"


async def _run_case(c: dict) -> dict:
    inputs = [Path(p) for p in c["inputs"]] or None
    verdict = await _vision_qa(Path(c["out"]), {}, user_message=c["intent"],
                              input_image_paths=inputs)
    got = _parse(verdict)
    return {"name": c["name"], "mode": "edit" if c["inputs"] else "generation",
            "intent": c["intent"], "expect": c["expect"], "got": got,
            "correct": got == c["expect"], "verdict": verdict.strip()[:400]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    img = _gen_images()
    cases = _cases(img)[: args.limit] if args.limit else _cases(img)
    print(f"[qa-test] {len(cases)} case(s); assets in {_ASSETS}")

    results = []
    for c in cases:
        r = asyncio.run(_run_case(c))
        mark = "OK " if r["correct"] else "XX "
        print(f"  {mark}{r['name']:32} expect={r['expect']} got={r['got']}")
        if not r["correct"]:
            print(f"       verdict: {r['verdict'][:160].encode('ascii','replace').decode()}")
        results.append(r)
        json.dump({"results": results}, open(_REPORT, "w", encoding="utf-8"), indent=2)

    n = len(results)
    correct = sum(r["correct"] for r in results)
    false_pass = sum(1 for r in results if r["expect"] == "FAIL" and r["got"] == "PASS")
    false_fail = sum(1 for r in results if r["expect"] == "PASS" and r["got"] == "FAIL")
    print(f"\n[qa-test] accuracy {correct}/{n} = {correct / n:.0%} | "
          f"false-pass={false_pass} false-fail={false_fail}  -> {_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
