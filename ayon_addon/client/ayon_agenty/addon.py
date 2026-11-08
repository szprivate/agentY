"""The client side of the agentY add-on."""

from __future__ import annotations

import os
import threading
import webbrowser

from ayon_core.addon import AYONAddon, ITrayAction

from . import lib
from .version import __version__

AGENTY_ADDON_ROOT = os.path.dirname(os.path.abspath(__file__))


class AgentYAddon(AYONAddon, ITrayAction):
    """agentY next to ComfyUI sessions launched from AYON.

    * a pre-launch hook for the ``comfyui`` application group (hooks/pre_launch_agenty.py);
    * a tray entry "agentY": starts the agentY host when it is down, opens its log when it runs.
    """

    name = "agenty"
    label = "agentY"
    version = __version__

    def initialize(self, settings):
        self.settings = lib.normalize((settings or {}).get(self.name))
        self.enabled = bool(self.settings.get("enabled", True))

    # -- launch hooks: offered for every application; the hook itself only runs for the comfyui group
    def get_launch_hook_paths(self, app=None):
        return [os.path.join(AGENTY_ADDON_ROOT, "hooks")]

    # -- tray
    def tray_init(self):
        return

    def on_action_trigger(self):
        threading.Thread(target=self._tray_action, name="agentY-tray", daemon=True).start()

    def _tray_action(self):
        url = self.settings["agent_server_url"].rstrip("/")
        if lib.host_healthy(url):
            webbrowser.open(url + "/agentY/log_viewer")
            return
        # From the tray there is no task, so studio settings apply; start it even when the launch-time
        # setting says "never": the artist asked for it here.
        state = lib.ensure_host({**self.settings, "start_host": "if_not_running"}, log=self.log.info)
        messages = {
            "started": "Starting the agentY host. The agentY panel in ComfyUI connects once it is up.",
            "running": "The agentY host is running.",
            "remote": f"The agentY host at {url} does not answer, and it is not on this machine.",
            "no_root": "Set the agentY folder in the agentY add-on settings to start the host from here.",
            "missing": "run_agent was not found in the agentY folder set in the agentY add-on settings.",
        }
        self.execute_in_main_thread(lambda: self.show_tray_message("agentY", messages.get(state, state)))
