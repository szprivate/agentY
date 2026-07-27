# QA briefings

A **briefing** is what agentY's QA agent judges finished images and videos against.
Drop one here as `<name>.md` and it becomes available everywhere by that name.

```
config/qa/
  house-style.md          ← the criteria (this file's contents, verbatim)
  house-style.refs/       ← optional mood / reference images
    grade_reference.png
    lighting_01.jpg
```

Use it by name:

* in the chat panel — `/qa house-style`
* on the canvas — an `agentY hook` with `purpose: qa` whose directive cites
  `@house-style` (you can add extra criteria around the citation)

## Writing one

Write for a reader who can see the image but cannot read your mind. Each line
should be checkable on its own — the QA agent reports pass/fail **per criterion**,
and a failed criterion is fed back verbatim when it re-generates, so a vague line
produces a vague fix.

Good:

```markdown
- Skin tones stay warm; no orange or magenta cast.
- Hands: exactly five fingers, no fusing or extra digits.
- Depth of field is shallow — background clearly separated from the subject.
- Text in the image, if any, is legible and correctly spelled.
```

Less useful: `- looks good`, `- professional quality`, `- matches the brand`.

Reference images do the work that words can't: put the grade, lighting or
character look you mean into `<name>.refs/` and say *"matches the reference
grade"* rather than trying to describe it.

## Note

`README.md` is skipped — it is not offered as a briefing name.

Settings that control **how** QA runs (retries, caps, the judging model) live in
Settings ▸ qa and Settings ▸ llm ▸ pipeline ▸ qa_checker. This folder only
decides **what** is enforced.
