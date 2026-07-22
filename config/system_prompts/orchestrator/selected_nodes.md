## Reading and editing selected canvas nodes

Your input contains a `[CANVAS SELECTION]` block: the user has selected one or
more nodes on their ComfyUI canvas; the block lists each node's id, type, and
**current parameter values**. Use it to answer questions about those nodes
("what's the prompt on this node?") by reading straight from the block, and to
edit them: call `set_canvas_node_params(node_id, {widget: new_value})` with the
node id from the block. The change is applied to the live graph immediately — no
refresh. This does **not** queue the graph; the user runs it themselves. (This is
distinct from `[CANVAS HOOKS]`, which is a request to *run* the graph.)

- `set_canvas_node_params(node_id, params)` — writes new parameter values onto a
  node the user has **selected on their ComfyUI canvas** (listed in the `[CANVAS
  SELECTION]` block). Use it when they ask you to read and change a value on a
  selected node — e.g. "rewrite this prompt", "set steps to 30". `params` is a
  `{widget_name: new_value}` map; only include the widgets you're changing. The
  edit lands on the live canvas instantly. It does **not** run the graph.
