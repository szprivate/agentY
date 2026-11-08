"""AYON add-on package: agentY in the AYON launcher.

Build with ``python create_package.py`` (writes ``package/agenty-<version>.zip``) and upload the zip
on the AYON server (Studio settings -> Bundles -> Install addons). See README.md.
"""

name = "agenty"
title = "agentY"
version = "0.1.0"

# Client code the AYON launcher imports.
client_dir = "ayon_agenty"

# Works next to ynput's ComfyUI add-on (ayon-comfyui, settings layout of 0.0.28). It is not required: without
# it the launch hook simply never runs, and the tray entry still works.
ayon_required_addons = {}
ayon_compatible_addons = {}
