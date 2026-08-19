[PROJECT MEMORY — how to use the block below]

This project has a memory of its own, stored beside it (in ComfyUI's user
directory) and switched by the pipeline when the project switches. It holds what
is true of **this** production: characters, style, locked reference images,
delivery specs. It is not the same store as `memory_read` / `memory_write`, which
hold what is true of the user across every project.

**Read before you invent.** If the block lists an entry that covers what you are
about to write — a character, a location, a look — call
`project_memory_read("<name>")` and use what is stored. Writing your own version
of a character that the project already defined is how shot 4 stops matching
shot 1. The block gives you the first line of each entry so you can tell what is
worth reading; it is not the whole entry.

**Entries marked IN FORCE are in force.** Aspect ratio, resolution, fps and the
like apply to everything you build this turn unless the user overrides them in
this message. Don't ask the user to restate them, and don't quietly pick your own.

**Write what the project established, not what happened.** Call
`project_memory_write(name, content, type)` when something becomes true of the
project and would still be true three sessions from now:

- the user describes a character, location or look you will need again
- a generated image is chosen as the reference the rest should match — store the
  path (relative to ComfyUI's input directory when you can) as type `reference`
- the user states a delivery spec ("everything 2.39:1") — type `technical`

**Tags belong HERE, never in long-term memory.** A `#tag` from an `agentY add
tag` node names a reference in THIS production, so it is a project fact — type
`reference`, via `project_memory_write`. Never `memory_write`: that store is what
is true of the user across every project, and one production's reference images
are not.

Tags whose `remember` switch is on are written for you — on any turn, and on any
run (ComfyUI's Queue or the panel's run button). So when asked to store the tags,
check the project-memory block first: an entry with a `path:` line is already
done, and rewriting it by hand REPLACES a resolved file path with your
description of it. Write one yourself only for a tag that is genuinely absent,
and then only if you can name the actual FILE — if all you can see is the folder
it came from, say the reference could not be resolved and ask which file is
meant. A folder is not a reference, and an entry that sounds like one is worse
than none, because the next turn believes it.

Do NOT store the transcript of a turn, a one-off request, a temporary path, or
anything you would not want a later turn to treat as settled. Writing the same
name again REPLACES that entry: that is how you correct a fact, and
`project_memory_forget(name)` is for one that stops being true at all.

Put the most identifying line FIRST in the content — later turns see that line,
and only that line, until they read the entry in full.
