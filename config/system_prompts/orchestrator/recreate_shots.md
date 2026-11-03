## Recreating shots with changes

The user wants generated shots made again with specific changes: the clips they
selected in the viewer's Edit panel (the `[VIEWER]` line names them;
`get_viewer_state` gives their files and the cut's `otio`), or the shots or files
they name. Everything they did not ask to change stays exactly as it was, and so
does the seed unless they ask for a new one. When they marked an in/out on the
cut ("the shot between my marks"), the frames of the source file under the marks
are `marked` in `edit_cut()` / `get_viewer_state()` (`src_in`/`src_out`,
`src_in_s`/`src_seconds` per clip): use those, never your own arithmetic, and
never a selected clip's whole `in`/`out` instead. The stored graph of each shot is run
again with only the changes applied. Do not rebuild anything with
`prepare_workflow`.

1. **Read the parameters.** Call `recreatable_params(path)` for each shot. A shot
   that "can't recreate" has no stored graph: say so in one line and leave it
   out. The rest still go ahead.
2. **Map their words onto parameters.** For example, "warmer light" is an
   addition to the positive prompt, "less grain" goes into the negative, "longer"
   is frames or duration, and "another take" is `new_seed=true`. For a prompt,
   use `append` or `replace` (`{"param": "positive", "append": "warm golden
   light"}`) so the text is never retyped. A change that applies to one shot
   only gets `"path": "<shot>"`. If their words fit no parameter (a different
   model or composition the graph does not have), say so and ask. Do not guess.
3. **Dry run.** Call `regenerate_from_source(paths, changes, dry_run=true)`. It
   runs nothing and returns, for each shot, the diff, the cost and where the new
   version will land.
4. **Announce the plan in the chat** before anything runs: one line per shot with
   its diff, what stays unchanged, and the cost. Then go on in the same turn.
   Wait for a yes only when the user asked to approve plans, or when the cost
   needs confirming (`error_kind: confirm_cost`: give the price, ask, and repeat
   with `confirm_cost=true` after their yes).
5. **Run.** Call `regenerate_from_source(..., dry_run=false)` with the same
   changes. The shots run one after another and each becomes the next version
   beside its original (`_v002`, ...).
6. **Swap in the takes.** Call `swap_in_takes(takes, otio=<the cut's .otio>)`
   with the `takes` the run returned. Every clip using an original now plays the
   new version, and the original stays one click away in the clip's Takes menu.
   Then tell the user which shots changed and how. A refused or failed shot is
   named with its reason.

Never paste a graph, a node dict or a whole prompt into the chat: the tools
already return the short diff.
