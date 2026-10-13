[RUNTIME NOTE - not from the user. Your turn ended, but the graph on the canvas is
not finished: nodes you added or rewired this turn have required inputs with
nothing wired into them, so the graph cannot run. Nothing has been handed to the
user yet.

{list}

Wire each of them now with `edit_canvas_graph` (a `connect` op per input; add a
node first where nothing in the graph outputs the type), and check that the
result no longer lists `still_unwired`. Do not describe the step first. If an
input is deliberately left for the user to connect, say which and why in one
sentence instead.]
