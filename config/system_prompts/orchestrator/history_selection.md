[HISTORY SELECTION - what the user has picked in {where}. When they say "the selected one(s)", "these", "this render", "the one I have open", they mean these files, not the canvas.]
{list}
Each path is a file on disk: use it as an input exactly like a #tag's file (stage it with `upload_image`, then it is the LoadImage value), look at it with `view_image`. `is:` is what agentY recorded the file to be. The pick can change during the turn: `get_history_selection` reads it again.
