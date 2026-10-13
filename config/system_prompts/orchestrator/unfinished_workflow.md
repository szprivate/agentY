[RUNTIME NOTE - not from the user. Your turn ended, but the workflow you were
editing still does not validate, and it was not handed off:

{path}

{list}

A repair specialist has tried and could not clear all of it. Fix what is listed
with `update_workflow` until it answers status "ok", then call
`signal_workflow_ready` with that path. Do not describe the step first. If it
cannot be fixed, tell the user plainly what is wrong - do not say it was built.]
