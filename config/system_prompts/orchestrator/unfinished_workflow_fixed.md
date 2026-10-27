[RUNTIME NOTE - not from the user. Your turn ended with the workflow you were
editing not validating, and it was not handed off:

{path}

It failed on:
{list}

A repair specialist has fixed the file in place and it validates now. Call
`signal_workflow_ready` with that path now, then tell the user what was built.
Do not edit it again unless the request needs something the repair left out.
Do NOT run it: a run is only for what the user asked for in this turn.]
