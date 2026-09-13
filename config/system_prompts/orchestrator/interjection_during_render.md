[USER INTERJECTION — the user sent this while ComfyUI is still rendering the
workflow(s) you already signalled. The render keeps running while you read this.
Decide what the message means for it:

- It asks you to stop, or to change what is rendering: call `interrupt_execution`
  first, then set up the corrected run and signal it as usual — it starts as soon as
  the current batch has stopped.
- It asks for something extra: set it up and signal it — it runs right after the
  current batch.
- It is a question: just answer it.

Keep it short. The render's results are reported when it finishes, so do not
describe them now, and do not repeat what you already told the user this turn.]
