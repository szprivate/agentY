[DRY RUN — this turn builds the workflow and generates nothing.]

Do the request exactly as you would for a real run: build the workflow with
`prepare_workflow` (or make the canvas edit that was asked for), then call
`signal_workflow_ready`. The ONLY difference is that no graph is submitted to
ComfyUI — the built workflow is filed for the user to inspect instead.

So the product of this turn is a **built workflow**, not a description of one.
Do not stop to ask, offer options, or explain what you would build: build it.
Ask only if the request genuinely cannot be built without an answer.

Finish with a short report: what was built, from which template, the model, the
output — and anything that looked wrong while building it.
