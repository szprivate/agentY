# agentY AYON add-on

An AYON add-on (`agenty`) that makes a ComfyUI session started from the AYON launcher work with agentY without
the manual steps in `docs/dev/ayon.md`, section 9. It works next to ynput's ComfyUI add-on
([ayon-comfyui](https://github.com/ynput/ayon-comfyui)). The `ayon_*` agent tools (in agentY-core) are unchanged and do
not need this add-on.

## What it does

**Before ComfyUI starts** (a pre-launch hook for the `comfyui` application group, run after the ComfyUI add-on's own
hook):

1. **The port clash.** The ComfyUI add-on starts ComfyUI on its launch profile's port (8188 by default), which the
   ComfyUI agentY normally drives may already hold. If anything holds it, the new ComfyUI cannot bind, the session attaches to the old
   one and breaks. The hook checks every local launch profile's port (ports a *remote* profile points at on this
   machine are skipped, since that profile wants the ComfyUI there) and, as the setting says:
   - **block** (default): the launch stops with a message saying which ComfyUI holds the port;
   - **stop_if_idle**: closes the old ComfyUI when nothing is queued or running on it, else blocks;
   - **ignore**: launches anyway.

   An AYON ComfyUI session that is still open (its page on port 5454 answers) always blocks: closing its browser
   tab is how a session ends cleanly.
2. **The agentY host.** If the agentY host does not answer at its address (`http://127.0.0.1:5000`), it is started
   from the agentY folder (`run_agent.ps1` in a console of its own on Windows, `run_agent.sh` elsewhere, output in
   `.logs/ayon_host_start.log`). It gets a clean environment: none of the launcher's Python, Qt or `AYON_*`
   variables, so no login token of the artist reaches it. A host that runs already is left alone. If the
   session page is on a port other than 5454, the host is started with `AYON_COMFYUI_STATIC_URL` set for the `ayon_*`
   tools. A failure here never stops the launch.

**In the tray:** an **agentY** entry. Host running: opens its log viewer. Host down: starts it.

It does not change the ComfyUI add-on, publish anything, or talk to the AYON server.

## Settings (Studio / Project settings -> agentY)

| setting | default | |
|---|---|---|
| Enabled | on | |
| agentY folder (Windows / Linux / MacOS) | empty | the checkout with `run_agent.ps1`; empty = never start the host |
| agentY host address | `http://127.0.0.1:5000` | a host on another machine is never started from here |
| agentY host on ComfyUI launch | start it when it is not running | or never |
| Extra arguments for run_agent | none | e.g. `-NoUpdate` (Windows), `--no-update` |
| When a ComfyUI already runs on the launch port | block | or close it when idle, or launch anyway |

## Build and install

```
cd ayon_addon
python create_package.py            # -> package/agenty-0.1.0.zip
```

On the AYON server: **Studio settings -> Bundles -> Install addons**, upload the zip, add `agenty` to a bundle
(next to `comfyui`, `applications`, `core`), then set the agentY folder in the add-on's settings. `create_package.py` is
ynput's standard packer from [ayon-addon-template](https://github.com/ynput/ayon-addon-template), unchanged.

Layout:

```
package.py                    name / version / client_dir
server/                       BaseServerAddon + the settings model
client/ayon_agenty/
  addon.py                    AYONAddon + tray action, hands its hooks folder to the launcher
  hooks/pre_launch_agenty.py  the pre-launch hook (comfyui group)
  lib.py                      the logic, no AYON imports (tests/test_ayon_addon.py in agentY)
```

The add-on's version (`package.py`) moves on its own, not with the agentY release number: the server keeps
uploaded versions side by side and a bundle names one.

## Not tested live

Tested against fake ComfyUI and agentY servers, with stand-ins for the launcher's `ayon_core` / `ayon_applications`
modules, and by building the package. It has not run in an AYON launcher yet. On the first real launch, check the
launcher's log for the `agentY:` lines.
