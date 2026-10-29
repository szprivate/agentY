"""Offline comparison: the OLD template matching (v1.8.10) against the NEW one.

OLD = as released in v1.8.10:
  * pinning: the words of a template's FILE name (minus a hand-kept stop list,
    plus the 3+-digit check of bc2402c) all appear in the request;
  * version check: agenty_core.utils.template_fit at v1.8.10 (family regexes over
    the request, the template's widget values and its name).
NEW = this branch:
  * pinning: agenty_core.utils.model_names.pin_templates - the model the request
    names, matched against the recipe DB's / catalog's model names to the version,
    and the job it asks for;
  * version check: template_fit reading versions from those model names.

No agentY turn, no ComfyUI: both read the template corpus and recipe DB on disk.

    .venv/Scripts/python.exe scripts/bench_template_matching.py [--out FILE.md]
        [--bench D:/AI/comfy_harness_benchmark] [--core D:/AI/agentY-core] [--old-tag v1.8.10]

Prompts: every case of the harness benchmark (baseline, complex, extended) plus the
labelled wordings in EXTRA below. A case's correct templates are the ones whose
MAIN model file matches the case's model pattern (unet_name / ckpt_name) and whose
recipe media is the case's output; an EXTRA wording says its own.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# (prompt, expected pin, correct templates) - both as regexes over template names,
# the labels a person gives. Expected pin None: nothing should be pinned. A pin is
# correct when it is non-empty and every pinned template matches the expected regex.
EXTRA: list[tuple[str, str | None, str | None]] = [
    # versions written every way
    ("make a ltx2.5 text to video of waves", r"ltx2_5_t2v", r"^video_ltx2_5_t2v$"),
    ("LTX-2.5 t2v, a fox in the snow", r"ltx2_5_t2v", r"^video_ltx2_5_t2v$"),
    ("text to video with LTX v2.5, local", r"^video_ltx2_5_t2v$", r"^video_ltx2_5_t2v$"),
    ("LTX 2.3 image-to-video of this photo", r"ltx_?2_3.*(i2v|ia2v|image_to_video)|image_to_video_ltx_2_3",
     r"^video_ltx2_3_i2v$"),
    ("an LTX-2 image to video of the street", r"ltx2_i2v|api_ltxv_image_to_video", r"^video_ltx2_i2v$"),
    ("wan2.2 text to video, a dancer", r"wan_?2_2.*t2v|text_to_video_wan_2_2", r"^video_wan2_2_14B_t2v$"),
    ("Wan 2.2 image to video of the photo", r"wan_?2_2|^image_to_video$|wan_dancer",
     r"^video_wan2_2_14B_i2v$"),
    ("wan 2.1 text to video", r"text_to_video_wan$|wan2\.1.*t2v|wan_vace_14B_t2v", r"^text_to_video_wan$"),
    ("WAN-2.2 first and last frame video", r"wan2_2_14B_flf2v|key-frames|looped_restyler",
     r"^video_wan2_2_14B_flf2v$"),
    ("Qwen-Image-Edit-2511: replace the sky", r"(?i)2511|multiangle|^image_edit$|angle|systms", r"^image_qwen_image_edit_2511$"),
    ("qwen image edit 2511, edit this photo: make it night", r"(?i)2511|multiangle|^image_edit$|angle|systms", r"^image_qwen_image_edit_2511$"),
    ("Qwen Image 2512 text to image of a castle", r"2512", r"^image_qwen_Image_2512$"),
    ("qwen image edit 2509 edit: add a hat", r"2509|angles|image_to_real|light_migration", r"^image_qwen_image_edit_2509$"),
    # wrong / unknown versions: no pin
    ("wan 2.4 text to video of a forest", None, None),
    ("LTX 3 text-to-video of a forest", None, None),
    ("Qwen Image 2600 text to image", None, None),
    # sampler words and loose numbers
    ("a portrait, 8 steps, cfg 1, seed 2, 1024x1024", None, None),
    ("render 97 frames at 24 fps, 2 steps, denoise 0.5", None, None),
    ("seed 2.0, steps 30, sampler euler", None, None),
    ("use 2 steps and a 2512 px long edge", None, None),
    # no model at all
    ("a red fox in the snow", None, None),
    ("a cinematic photo, 35mm film grain, lens flare", None, None),
    ("upscale this image 2x using a model", None, None),
    ("remove the background of this photo", None, None),
    ("make a 10 second video of a cat", None, None),
    # brands / families
    ("flux dev text to image of a cat", r"flux_?1?_?dev|flux_dev", r"^flux_dev_full_text_to_image$"),
    ("SDXL text to image of a lighthouse", r"sdxl", r"^image_sdxl_simple$"),
    # basic_switch_node and story_refs_only are Z-Image Turbo text-to-image graphs too
    # (main file z_image_turbo_bf16), as the harness label of EX03 counts them.
    ("a z-image turbo image of a dog, 8 steps", r"z_?image_turbo|zimage_turbo|^basic_switch_node$|^story_refs_only$",
     r"^image_z_image_turbo$"),
    ("FLUX.2 Klein image edit: change the jacket to red", r"klein|all_in_one", None),
    # written out by name
    ("use image_qwen_image_2512_with_2steps_lora for this", r"^image_qwen_image_2512_with_2steps_lora$",
     r"^image_qwen_image_2512_with_2steps_lora$"),
    ("run video_ltx2_5_t2v with 'a kite over dunes'", r"^video_ltx2_5_t2v$", r"^video_ltx2_5_t2v$"),
]

# Harness cases whose model pattern is looser than the version they name (the
# pattern for "LTX-2" also matches LTX 2.3 files): the main model file must also
# match this for a template to count as correct.
CASE_FILE_RE = {
    "CX01_Flux_Dev_Text2Img": r"(?i)flux1", "CX09_Flux_Redux_Style": r"(?i)flux1",
    "EX06_Flux_Canny_Control": r"(?i)flux1", "EX01_Flux_Kontext_Edit": r"(?i)flux1",
    "CX10_LTX2_Text2Video": r"(?i)ltx-2-19b", "EX11_LTX2_Image2Video": r"(?i)ltx-2-19b",
}


# ---------------------------------------------------------------------------
# the two matchers
# ---------------------------------------------------------------------------

def _git_show(repo: Path, tag: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{tag}:{path}"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def old_matchers(tag: str, core: Path):
    """(pin(text, catalog) -> list|None, template_fit module) as released in *tag*."""
    src = _git_show(ROOT, tag, "src/pipeline.py")
    a = src.index("_ORCH_STOP_TOKENS: frozenset")
    b = src.index("# Pipeline callable")
    ns: dict = {"re": re, "json": json}
    exec(src[a:src.rfind("# ----", a, b)], ns)          # stop list, _split_identifier, _brand_tokens
    m = src.index("    def _match_named_templates(self, user_text: str)")
    end = src.index("\n    def ", m + 10)
    exec(textwrap.dedent(src[m:end]), ns)

    def pin(text: str, catalog: list[str]):
        fake = types.SimpleNamespace(_template_brand_index=lambda: {n: set(ns["_brand_tokens"](n)) for n in catalog})
        out = ns["_match_named_templates"](fake, text)
        return list(out[1]) if out else None

    tf_src = _git_show(core, tag, "agenty_core/utils/template_fit.py")
    tf = types.ModuleType("old_template_fit")
    exec(compile(tf_src, "old_template_fit.py", "exec"), tf.__dict__)
    return pin, tf


def new_matchers():
    from agenty_core.utils import model_names as M
    from agenty_core.utils import template_fit as tf
    vocab = M.vocabulary()
    api = _api_templates()
    # Pipeline._wants_local's words ("local", "no api", ...), without importing the pipeline.
    rx = re.compile(r"local|offline|on my machine|on my gpu|on my pc|without api|no api|"
                    r"self hosted|open source|opensource|no partner|without partner")
    local_re = lambda t: bool(rx.search(re.sub(r"[\s_-]+", " ", (t or "").lower())))  # noqa: E731

    def pin(text: str, catalog: list[str]):
        low = text.lower()
        literal = sorted(n for n in catalog if len(n) >= 6 and re.search(
            r"(?<![\w.-])" + re.escape(n.lower()) + r"(?![\w-])", low))
        if literal:
            return literal
        out = M.pin_templates(text, catalog=catalog, vocab=vocab)
        if not out:
            return None
        names = list(out["templates"])
        if local_re(text):
            names = [n for n in names if n not in api] or names
        return names
    return pin, tf


def _api_templates() -> set:
    from agenty_core.paths import corpus_root
    db = json.loads((corpus_root() / "config" / "workflow_recipes.json").read_text(encoding="utf-8"))
    api, local = set(), set()
    for t in db.get("tasks") or []:
        for m in t.get("models") or []:
            (api if m.get("uses_api_nodes") else local).update(m.get("member_files") or [])
    return api - local


_FACTS: dict = {}


def version_rejects(tf, prompt: str, catalog: list[str]) -> set:
    """The catalog templates *tf*'s check marks wrong for another model version."""
    want = tf.request_wants(prompt)
    cache = _FACTS.setdefault(id(tf), {})
    out = set()
    for name in catalog:
        if name not in cache:
            cache[name] = tf.facts_for_template(name)
        facts = cache[name]
        if facts is None:
            continue
        v = tf.check(want, facts)
        if any("the request names" in r for r in v.get("reasons") or []):
            out.add(name)
    return out


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------

def bench_cases(bench: Path) -> list[dict]:
    sys.path.insert(0, str(bench))
    cases = []
    for mod in ("baseline_tests", "complex_tests", "extended_tests"):
        m = __import__(f"datasets.{mod}", fromlist=["SUITE"])
        for c in m.SUITE:
            cases.append({"id": c.id, "prompt": c.prompt, "case": c, "suite": mod.split("_")[0]})
    return cases


def correct_for(case, main_files: dict, media_of: dict, also: str | None = None) -> set | None:
    """Templates whose main model file matches the case's model pattern, of its output media."""
    if case.initial_workflow:
        return None
    pats = [v for cls, inp, v in (case.expected_patterns or [])
            if re.search(inp, "unet_name") or re.search(inp, "ckpt_name")]
    if not pats:
        return None
    groups = [g for g in (case.required_any or [])] + [case.required_nodes or []]
    video = any(x in ("SaveVideo", "CreateVideo", "VHS_VideoCombine", "SaveWEBM") for g in groups for x in g)
    want = "video" if video else "image"
    out = set()
    for name, files in main_files.items():
        if media_of.get(name) not in (None, want):
            continue
        if any(re.search(p, f) and (not also or re.search(also, f)) for p in pats for f in files):
            out.add(name)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="D:/AI/comfy_harness_benchmark")
    ap.add_argument("--core", default=str(ROOT.parent / "agentY-core"))
    ap.add_argument("--old-tag", default="v1.8.10")
    ap.add_argument("--out", default=str(ROOT / "output" / "bench_template_matching.md"))
    a = ap.parse_args(argv)

    from agenty_core.tools.comfyui import get_workflow_catalog
    from agenty_core.utils import model_names as M
    catalog = [str(n) for n in json.loads(get_workflow_catalog() or "{}")]
    entries = M.build_entries(M._corpus_dirs(), json.loads(
        (M._recipe_db_path()).read_text(encoding="utf-8")))
    main_files = {n: e.get("files") or [] for n, e in entries.items() if n in set(catalog)}
    media_of = {n: M._TASK_MEDIA.get(e.get("task") or "") for n, e in entries.items()}

    old_pin, old_tf = old_matchers(a.old_tag, Path(a.core))
    new_pin, new_tf = new_matchers()

    rows = []
    for c in bench_cases(Path(a.bench)):
        corr = correct_for(c["case"], main_files, media_of, CASE_FILE_RE.get(c["id"]))
        # A case that edits a graph it is given should pin nothing; one with a model
        # pattern should pin only templates whose main model matches it.
        rows.append({"id": c["id"], "prompt": c["prompt"], "correct": corr or None,
                     "expect": (lambda ps, s=frozenset(corr): set(ps) <= s) if corr else None,
                     "label": "none" if c["case"].initial_workflow else ("set" if corr else "unknown")})
    for i, (p, expect, must) in enumerate(EXTRA, 1):
        rows.append({"id": f"W{i:02d}", "prompt": p,
                     "correct": {n for n in catalog if re.search(must, n)} if must else None,
                     "expect": (lambda ps, rx=expect: all(re.search(rx, n) for n in ps)) if expect else None,
                     "label": "set" if expect else "none"})

    keys = ("correct_pin", "wrong_pin", "missed_pin", "no_pin_ok", "pin_unlabelled",
            "correct_rejected", "correct_allowed", "correct_known")
    totals = {k: dict.fromkeys(keys, 0) for k in ("old", "new")}
    for r in rows:
        for k, pin, tf in (("old", old_pin, old_tf), ("new", new_pin, new_tf)):
            pinned = pin(r["prompt"], catalog)
            rej = version_rejects(tf, r["prompt"], catalog)
            if r["label"] == "set":
                verdict = ("missed pin" if not pinned else
                           "correct pin" if r["expect"](pinned) else "wrong pin")
            elif r["label"] == "none":
                verdict = "wrong pin" if pinned else "no pin (ok)"
            else:
                verdict = "pin (no label)" if pinned else "no pin (ok)"
            t = totals[k]
            t[{"correct pin": "correct_pin", "wrong pin": "wrong_pin", "missed pin": "missed_pin",
               "no pin (ok)": "no_pin_ok", "pin (no label)": "pin_unlabelled"}[verdict]] += 1
            corr = r["correct"] or set()
            wrongly = sorted(corr & rej)
            allowed = None
            if corr:
                t["correct_known"] += 1
                allowed = any(n not in rej and (not pinned or n in pinned) for n in corr)
                t["correct_allowed"] += int(allowed)
                t["correct_rejected"] += len(wrongly)
            r[k] = {"pinned": pinned, "verdict": verdict, "rejected": len(rej), "wrongly": wrongly,
                    "allowed": allowed}

    def cell(x):
        pins = ", ".join(x["pinned"][:3]) + (f" +{len(x['pinned']) - 3}" if x["pinned"] and len(x["pinned"]) > 3 else "") \
            if x["pinned"] else "-"
        allowed = "" if x["allowed"] is None else ("yes" if x["allowed"] else "**NO**")
        wrong = f"; wrongly rejects {', '.join(x['wrongly'])}" if x["wrongly"] else ""
        return f"{x['verdict']}: {pins} (rejects {x['rejected']}{wrong}) | {allowed}"

    lines = ["# Template matching: v1.8.10 (old) vs recipe model names (new)", "",
             f"{len(rows)} prompts: {len(rows) - len(EXTRA)} harness cases + {len(EXTRA)} labelled wordings. "
             "Offline: no agentY turn, no ComfyUI.", "",
             "## Totals", "", "| | old | new |", "|---|---|---|"]
    labels = [("correct_pin", "correct pins"), ("wrong_pin", "wrong pins"), ("missed_pin", "missed pins"),
              ("no_pin_ok", "no pin, none expected"), ("pin_unlabelled", "pins on unlabelled cases"),
              ("correct_rejected", "correct templates wrongly rejected as the wrong version"),
              ("correct_allowed", "cases whose correct template is still allowed")]
    for key, label in labels:
        o, n = totals["old"][key], totals["new"][key]
        if key == "correct_allowed":
            lines.append(f"| {label} | {o} / {totals['old']['correct_known']} | {n} / {totals['new']['correct_known']} |")
        else:
            lines.append(f"| {label} | {o} | {n} |")
    lines += ["", "A pin is correct when every pinned template is one of the expected ones. "
              "\"rejects N\" counts the catalog templates the version check marks NOT FOR THIS REQUEST "
              "for another model version. \"allowed\" = a correct template is neither version-rejected "
              "nor left out of the pin.", "",
              "## Per prompt", "", "| id | prompt | old: pin (rejects) | old allowed | new: pin (rejects) | new allowed |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        p = r["prompt"].replace("|", "/")
        p = p if len(p) <= 90 else p[:87] + "..."
        lines.append(f"| {r['id']} | {p} | {cell(r['old'])} | {cell(r['new'])} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
