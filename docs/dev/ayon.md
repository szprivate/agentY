# agentY and Ayon (working design note)

This is what the studio's Ayon ComfyUI addon does, read from its code, and what agentY can and cannot do through it. The
addon is not changed: agentY uses what it already exposes. Written against the installed addon
`comfyui_0.0.28+bpc.1` (a fork of [ynput/ayon-comfyui](https://github.com/ynput/ayon-comfyui), Apache-2.0).

Test context: server `https://bepic.ayon.app`, project `i002_ai_rnd`, folder `agentRND/agentRND_0010`, task `ai`.

## 1. The short version

- Ayon does not run inside ComfyUI. An **Ayon host process** (Python, Qt, with `ayon_core` and pyblish) starts ComfyUI as a
  child process and shows it in a browser tab, wrapped in a tiny web page it serves itself. Every Ayon tool (Loader,
  Creator, Publisher, Workfiles, Scene Inventory) is a Qt window of that host process.
- The only link between the Ayon host and the canvas is a websocket **from the browser tab to the host**. The host
  pushes node edits down it. There is no way for another program to push edits down it.
- What the Loader and the Creator leave on the canvas is **plain node data**: JSON in a text field of an `AYON ...` node.
  That data can be written without the Ayon host, and agentY does (`ayon_load`, `ayon_mark_publish`).
- The **publish** runs pyblish inside the Ayon host process, behind a Qt window. Nothing can start it from outside.
  `ayon_publish` opens that window and says so; the user presses Publish. That is the one open item (section 8).

## 2. How ComfyUI is launched through Ayon

Chain, in order:

1. The user starts the **ComfyUI** application from the Ayon launcher, in a task (project / folder / task). The
   applications addon prepares the environment: `AYON_PROJECT_NAME`, `AYON_FOLDER_PATH`, `AYON_TASK_NAME`,
   `AYON_WORKDIR`, `AYON_APP_NAME`, `AYON_SERVER_URL`, and the user's login for the child process.
2. The addon's launch hook (`hooks/pre_launch_args.py`, app group `comfyui`) replaces the command with
   `ayon run api/launch_script.py`. So the "application" the launcher starts is the addon's own Python script, running in
   the Ayon launcher's Python, not ComfyUI.
3. `launch_script.py` -> `launch_logic.main()`: installs the `ComfyUIHost` (registers the loader / creator / pyblish
   plugin paths), starts a Qt app and shows the **profile dialog** ("Select launch profile"): a *local* profile (start
   ComfyUI here) or a *remote* profile (an already running ComfyUI at some address).
4. Local profile: `_subproc_launch_ComfyUI()` starts ComfyUI as a child process, in a new PowerShell console:
   `python -s <base_folder>\main.py --disable-auto-launch --port <comfy_launch_port> <launch_args> --extra-model-paths-config <temp yaml>`.
   - Python: the profile's custom Python, or `<base_folder>\..\python_embeded\python.exe` for a "Windows portable" build
     (profile flag `comfy_is_windows_portable`, **default on**), or a managed venv built from the base folder's
     `requirements.txt` under `addons_resources\comfyui_envs`.
   - `PYTHONPATH` is set to the base folder, and the child inherits the **whole environment of the Ayon process**. That is
     why the Ayon context is readable from the ComfyUI process (agentY's `ayon_context` reads only the five `AYON_*`
     names it needs, never the rest).
   - The packaged ComfyUI plugin (`_comfyui_plugin/ayon_menu`) is not copied into `custom_nodes`. A temp yaml (deleted after
     20 s) adds its folder as an extra `custom_nodes` path. Before the start, the launcher rewrites `ayon_menu/consts.py`
     (backend port) and `js/lib/consts.js` (the static page address) inside the installed addon folder.
   - Launch arguments such as `--user-directory`, `--input-directory`, `--output-directory` come from the profile's
     `launch_args` list. The running ComfyUI on this machine (`D:\ai\comfyui`, port 8188) has exactly such arguments
     (`W:/i002_ai_rnd/02_build/comfy/<user>/...`), so the studio profile most likely points at the same install.
5. Then it starts the three servers of the host (section 3) and opens the browser on the static page. The browser tab is
   the Ayon "session". **Closing that tab ends the session**: the host notices the missing tab and kills the ComfyUI
   process tree (`taskkill /T /F`). If ComfyUI dies on its own, the heartbeat fails and the host shuts itself down.

**Where it is configured.** Not in files on the machine. The addon's settings are on the Ayon server (studio and project
level, addon `comfyui`): the list of local launch profiles (`comfy_launch_port`, `comfy_base_folder_win/lin/osx`,
`launch_profile.launch_args_*`, `comfy_is_windows_portable`, `dev_omit_packaged_ayon_comfyui_plugin`, extra model
directories), the remote profiles, and the three ports `http_server_port`, `server_pulse_port`, `frontend_port`. The
launcher fetches them at start; the machine keeps no copy (`%LOCALAPPDATA%\Ynput\AYON` holds addons, the launcher state and
`executables.json`, nothing of the addon's settings). I could not read the studio's values (that needs a login), so the
numbers below are the addon's defaults, from the upstream `server/local_settings.py`; the studio fork may differ.

| setting | default | meaning |
|---|---|---|
| `comfy_launch_port` | **8188** | the port ComfyUI is started on |
| `http_server_port` | 5454 | the static page the browser tab shows (ComfyUI sits in an iframe on it) |
| `frontend_port` | 55056 | the host's WSRPC server (the browser tab connects to it) |
| `server_pulse_port` | 55055 | a websocket server inside ComfyUI's backend (the host connects, a heartbeat) |

**The port clash.** The ComfyUI agentY normally drives is `D:\ai\comfyui` on port 8188, and 8188 is also the profile's
default. Ayon cannot start a second ComfyUI on a port that is taken: it would print the bind error in its console, the
heartbeat would connect to the *old* ComfyUI (there is no ayon_menu plugin in it, so no server on 55055 either), and the
session would be broken. Two ways around it:

- **One ComfyUI (recommended).** Stop the running ComfyUI (or agentY's launcher) and start ComfyUI only from Ayon. agentY's
  `COMFYUI_URL` stays `http://127.0.0.1:8188` and now points at the Ayon-launched one. The same install loads
  agentY-comfyuiConnect too (it is in `custom_nodes`), so the agentY panel and the `/agent/*` routes are there.
- **Two ComfyUIs.** Set the Ayon profile's `comfy_launch_port` to another port (8189) on the server. The `ayon_*` tools
  talk to the session's ComfyUI (they read its address from the static page), agentY's other tools stay on 8188. Output
  from one is not visible to the other unless both use the same input / output directories. Not recommended.

Never launch the Ayon session while the ComfyUI on 8188 is busy with a render you care about.

## 3. The RPC

```
[Ayon host process, Python + Qt]                       [Browser tab]
  RPCServerThread  ws://127.0.0.1:55056/ws/  <------>  static page http://127.0.0.1:5454   (proxy.js, WSRPC client)
                                                          |  postMessage {rpc, id, method, params}
  WSClientThread   ws://127.0.0.1:55055/ws   ------>   ComfyUI backend (ayon_menu/ws_server.py, heartbeat only)
                                                          iframe -> ComfyUI frontend (ayon_menu.js, RPCServer)
```

Three hops: **host <-> static page** is WSRPC over a websocket; **static page <-> ComfyUI frontend** is `postMessage`
between the page and its iframe (the page repeats each route into the iframe, and the iframe's menu actions go the other
way through a polled `pop_process` queue); the **heartbeat** (host as client, ComfyUI's backend as server) carries no
messages, it only tells the host when ComfyUI is gone.

**Transport.** [wsrpc-aiohttp](https://pypi.org/project/wsrpc-aiohttp/) JSON over websocket, text frames. Per frame:

```
call      {"id": 3, "method": "ayonComfyUI.pingAyonMenu", "params": {"message": "x"}}
result    {"result": "x", "id": 3}
error     {"error": {"type": "NotImplementedError", "message": "..."}, "id": 3}
keepalive {"id": 1, "method": "ping", "params": {"seq": 12.5}}   -> the client must answer {"id": 1, "result": {"seq": 12.5}}
```

`params` as an object becomes keyword arguments. The server pings every client every 30 s and closes one that does not
echo. No authentication on the websocket: it listens on 127.0.0.1 only.

**Browser -> host** (route class `ayonComfyUI`, all in `api/rpc_server.py`; these are the only methods a client can call):

| method | params | does |
|---|---|---|
| `pingAyonMenu` | `message` | returns `message` |
| `requestToolByName` | `tool_name`: `loader` `create` `publisher` `workfiles` `sceneinventory` | schedules the Qt tool in the host's Qt thread (a window opens on the artist's machine); returns `"<tool> scheduled in qt_thread"` |
| `requestSaveByName` | `file_name`, `workfile_contents` | writes the contents over `<AYON_WORKDIR>/<file_name>.json` if it exists, else opens Workfiles |

**Host -> browser** (the host calls these with the `call_on_origin` decorator, `rpc_stub.py`; the static page repeats them
into the iframe; implemented in `ayon_menu.js`; all with a 10 s timeout except `getPublishNodeImages`, which waits for the
graph to cook):

`getWorkfile`, `loadWorkfile(workfile_json, workfile_name)`, `updateTab(new_name)`, `getImprintContext`,
`setImprintContext(imprint_info)`, `addPublishNode(instance_json, node_type)`, `removePublishNodes(ids_to_remove, node_type)`,
`getPublishNodes(node_type)`, `getPublishNodeImages(id_for_images, node_type)`, `addLoadProductNode(container_json, node_type)`,
`removeLoadProductNodes`, `updateLoadProductNode(container_json, node_type)`.

**How the host picks its client.** `get_client_from_origin()` returns the *first* connected client whose `Origin` header is
the static page (`http://127.0.0.1:5454`) and sends every host -> browser call to it. So: (1) only one browser tab is the
Ayon session, (2) a second client that did not send that Origin is never given these calls, which is why agentY's own
client sends no Origin and can only use the three browser -> host methods above, and (3) a second tab on the static page
would compete for the calls (the "single tab" limit).

## 4. How load, create and publish actually happen

**Load** (Loader window -> `plugins/load/load_image.py`, `load_video.py`, `load_model.py`; Qt, in the host):

1. The artist picks a representation in the Loader. The host finds its file(s) on disk (anatomy roots, for a sequence all
   frames).
2. `upload_input_images` posts each file to the ComfyUI backend `POST /upload/image` (`type=input`,
   `subfolder=AYON/<namespace>_<name>`); the response `{name, subfolder, type}` is kept.
3. `containerise()` builds the container dict: `{"schema": "ayon:container-3.0", "id": "ayon.load.container", "name",
   "namespace", "loader", "image_upload_info": [...], "representation": <id>, "container_uuid", "container_name"}` and adds
   it to the `containers` list of the AYON Context node (`getImprintContext` / `setImprintContext`). This list is what the
   Scene Inventory shows.
4. `addLoadProductNode` creates an `AYON Load Image` / `AYON Load Video` / `AYON Load 3D Model` node at the centre of the
   view and writes the container JSON into its `ayon_container_info` text field, then queues just that node so a preview
   shows. The node's code reads the uploaded file from ComfyUI's input folder.

*Needs the Ayon host:* steps 1 (anatomy, representation lookup) and the Qt picking. *Plain data:* steps 2-4. agentY does 2-4
itself and does step 1 with the Ayon server API (`ayon_load`).

**Create** (Creator window -> `plugins/create/create_image.py`, `create_video.py`, `create_model.py`; Qt, in the host):

1. The artist picks a creator and a variant. The creator builds the instance: product name `<prefix>_<unique>_<variant name>`,
   `creator_attributes` (`keep_metadata`, `prefix`, `unique_name`, `compression_level` / `formatcodec` /
   `fallback_format`, `force_recook_on_publish`), the folder / task / project, and an `instance_id`. Stored as
   `{"id": "ayon.create.instance", "productType", "productBaseType", "productName", "active", "creator_identifier",
   "variant", "folderPath", "task", "projectName", "instance_id", "creator_attributes", "publish_attributes"}`.
2. `addPublishNode` creates an `AYON Image Save` / `AYON Video Save` / `AYON 3D Model Save` node and writes that JSON
   into its `ayon_info` field (and `recook`). The user wires an image / video / mesh into it.
3. The instance is added to the `instances` list of the AYON Context node. That list is what the Publisher shows.

*Needs the Ayon host:* the Creator window and the naming template. *Plain data:* 2 and 3 (`ayon_mark_publish`).

**Publish** (Publisher window, pyblish, in the host):

1. The Publisher collects instances from the context node, and the workfile instance.
2. `CollectImage` (and the video / model collectors) call `getPublishNodeImages`: the browser finds the save node by
   `instance_id`, re-runs the graph up to it when `recook` is on, and answers with `/api/view?...` URLs of the saved files.
3. The collector downloads them to a staging directory and builds representations (`png` + thumbnail; review tag).
4. Validators, extractors and integrators run; versions, representations and files are written to the Ayon server and the
   publish folder. `CollectWorkfile` / `IncrementWorkfile` save and increment the workfile (JSON under `AYON_WORKDIR`;
   it asks the browser for the workflow with `getWorkfile`).

*Needs the Ayon host, all of it* (pyblish, anatomy, the server session of the logged-in user). No part can be done by
writing node data.

## 5. What agentY can call, and from where

agentY's host is a separate Python process with its own venv, not Ayon's; it does not import `ayon_core`. Three ways to
reach the Ayon side exist:

| route | reaches | used for |
|---|---|---|
| WSRPC client to `ws://127.0.0.1:55056/ws/` (no Origin) | `pingAyonMenu`, `requestToolByName`, `requestSaveByName` | detect the session, open the Loader / Creator / Publisher windows |
| ComfyUI's own HTTP API at the address the static page names | `/upload/image`, `/agent/set_node_params` (agentY-comfyuiConnect), `/system_stats` | upload loaded files, fill widgets of existing nodes |
| agentY's panel on the canvas (`edit_graph` patch) | add / wire / set nodes in the live graph, in place | place AYON nodes and wire them |

Not reachable: anything host -> browser (`addPublishNode`, `getImprintContext` ...) as such. A client cannot receive those
calls, and impersonating the tab's Origin to steal them is not on the table. The same results are obtained by editing the
nodes directly, because the host's calls only write widget values.

The Ayon **server** is not reached by these tools at all: see section 6 for the split with the official ayon-mcp server.

**Where the context comes from.** Two readings, both reported by `ayon_context`:
- *launch context:* `AYON_PROJECT_NAME`, `AYON_FOLDER_PATH`, `AYON_TASK_NAME`, `AYON_WORKDIR` from the ComfyUI process
  (allow-listed names only; needs `psutil`).
- *canvas context:* the AYON Context node's `ayon_context_info` text, which also holds the registered instances and
  containers. agentY's panel posts the whole graph with each message, so inside agentY it is always at hand. From
  agentY-mcp (no panel) the user selects the AYON Context node and the tool reads the selection.

## 6. Two layers: ayon-mcp for the server, these tools for the live session

| layer | covers | how |
|---|---|---|
| **ayon-mcp** (Ynput's official MCP server, `https://<server>/api/mcp`, key in the `x-api-key` header; its own guardrails: `confirm_mutation`, `AYON_MCP_READ_ONLY`) | everything on the Ayon **server**: projects, folders, tasks, products, versions, representations, GraphQL, comments, events, settings | the MCP preset (added separately) |
| **the `ayon_*` tools** (this section) | everything in the live **ComfyUI session**: which context it runs in, loading a file into the canvas, marking an output for publish, opening the Publisher | the addon's RPC, ComfyUI's HTTP API, the agentY panel |

They meet at small hand-offs. To load a product: ask ayon-mcp for the product's latest version and its representation
(file path with roots resolved, representation id), then call `ayon_load(path, product, representation_id=...)`. To
check a publish: `ayon_publish` returns the product names it opened the Publisher for; after the user presses Publish,
ayon-mcp finds the new version of those products and can add a comment to it. Frame range, fps, resolution and task
status are server data: `ayon_context` names the project / folder / task and says to ask ayon-mcp. The tools do not
query the server and need no key; `AYON_API_KEY` / `AYON_SERVER_URL` (Settings -> Add API key -> Ayon) are for ayon-mcp.

## 7. The tools (agenty_core/tools/ayon.py, pack `ayon`)

| tool | does | needs |
|---|---|---|
| `ayon_context()` | session check (static page + ping), launch context, canvas context (instances, containers) | the session |
| `ayon_load(path, product, representation_id="", folder_path="", version=0, kind="", load_node_id="", dry_run=False)` | upload the file to `input/AYON/...`, fill an AYON Load node, register the container | the file readable from this machine |
| `ayon_mark_publish(node_id="", kind="image", variant="Main", prefix="ayon", unique_name="ayon", product_name="", output=0, save_node_id="", force_recook=False)` | add an AYON Save node with the creator's instance JSON, wire the node's output into it, register the instance | agentY panel for add + wire; otherwise returns the node data and a to-do |
| `ayon_publish()` | check instances are registered, open the Publisher, return the product names | RPC |
| `ayon_open_tool(tool_name)` | open loader / create / publisher / workfiles / sceneinventory | RPC |

`AYON_COMFYUI_STATIC_URL` (default `http://127.0.0.1:5454`) is only for a changed `http_server_port`.

Design rules the tools follow: say "not launched through Ayon" when the static page does not answer; never write a node
whose result they cannot confirm; say what is registered and what is not (an instance that is not on the AYON Context
node is invisible to the Publisher, and the result says so); never fake a Qt action. Writing a node never goes through
`/agent/load_workflow`: that reloads the tab under another name (`agent bake`), and the Ayon workfile is found by the tab
name.

## 8. Risks and open items

**Risks**
- **One tab.** The host serves one browser client. A second tab on the static page, or reloading the tab, is picked up
  differently by `get_client_from_origin` (first match). Reloading is fine; opening a second tab is not. Closing the tab
  ends the session and kills ComfyUI.
- **ComfyUI restarts end the session.** The heartbeat fails after about 15 s, the host stops its servers and exits. Do not
  restart the ComfyUI an Ayon session runs on; start a new session from the launcher.
- **The tab name is the workfile.** `Ayon Save Workfile` finds `<AYON_WORKDIR>/<tab name>.json`. Anything that reloads the
  tab under another name (`/agent/load_workflow`, "open workflow in canvas") breaks that. The ayon tools do not.
- **Registration.** `ayon_mark_publish` and `ayon_load` rewrite the AYON Context node's text. If the user edits instances in
  the Ayon Creator at the same moment, the last writer wins. Within one agentY turn the tools merge with what they wrote
  earlier (the panel's snapshot is older than the edit).
- **Publishing without the UI is not possible** through this RPC: pyblish and the server session live in the Qt process.
  A "publish" from agentY that only writes files into the publish folder would put unregistered files on disk and skip
  the validators; it is not offered.
- **Reading the launch context** uses `psutil` on the ComfyUI process, restricted to five names. If a future launcher
  hides them, `ayon_context` falls back to the canvas context and says so.
- **Untested live:** every tool was tested against recorded RPC frames and mocked ComfyUI / canvas calls only. Run
  `ayon_load(dry_run=True)` first (section 9).

**Open items** (need a change on the Ayon side, or the user)
- Headless publish: needs an RPC route in the studio's addon fork that runs `create_context` + `publish_in_test`-style
  publishing in the host, or a server-side farm publish. Until then the user presses Publish.
- Choosing a representation by hand in the Loader, and the Creator's product-name template, stay Qt.
- Wiring and adding nodes outside agentY (agentY-mcp): returns data to paste. A small `/agent/edit_graph` route in
  agentY-comfyuiConnect would close this; not built here.

## 9. Live test (needs the user)

Do these steps yourself; agentY never starts the Ayon launcher.

1. Stop the ComfyUI on port 8188 (or the whole agentY launcher), so the port is free.
2. Start the **Ayon launcher**, pick project `i002_ai_rnd`, folder `agentRND/agentRND_0010`, task `ai`, application
   **ComfyUI**, and Launch.
3. In the profile dialog choose the local profile for `D:\ai\comfyui`. Wait for the browser tab on
   `http://127.0.0.1:5454`. Keep that single tab open. Check the **AYON** entry in the sidebar.
4. Restart the agentY host (:5000) so it loads the new tools. (The service-user key is only needed for ayon-mcp.)
5. From `D:\AI\agentY`:
   ```
   .venv\Scripts\python.exe scripts\ayon_smoke.py
   .venv\Scripts\python.exe scripts\ayon_smoke.py --path W:\...\imageMain_v003.png --product imageMain   # a dry-run load
   ```
   It prints the context and the dry-run load. It changes nothing on the canvas and uploads nothing.
6. Then in the agentY panel, with the same tab: "what Ayon context am I in", "load imageMain from this shot",
   "publish the result of the VAE Decode node as an image, variant Main", "open the publisher".
