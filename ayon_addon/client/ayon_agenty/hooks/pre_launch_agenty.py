"""Before a ComfyUI session starts from the AYON launcher: free its port, and have the agentY host up.

Runs after the ComfyUI add-on's own hook (order 20), in the launcher. See ayon_agenty/lib.py.
"""

from __future__ import annotations

from typing import ClassVar

from ayon_applications import ApplicationLaunchFailed, LaunchTypes, PreLaunchHook

from ayon_agenty import lib


class AgentYComfyUIPrelaunch(PreLaunchHook):
    app_groups: ClassVar[set] = {"comfyui"}
    order: ClassVar[int] = 25
    launch_types: ClassVar[set] = {LaunchTypes.local}

    def _project_settings(self) -> dict:
        settings = self.data.get("project_settings")
        if settings:
            return settings
        from ayon_core.settings import get_project_settings, get_studio_settings
        project = self.data.get("project_name")
        return get_project_settings(project) if project else get_studio_settings()

    def execute(self) -> None:
        project_settings = self._project_settings()
        settings = lib.normalize(project_settings.get("agenty"))
        if not settings["enabled"]:
            return
        comfy = project_settings.get("comfyui") or {}
        try:
            for note in lib.check_comfyui_ports(comfy, settings["comfyui_port_busy"], log=self.log.info):
                self.log.info("agentY: %s", note)
        except lib.PortBusy as exc:
            raise ApplicationLaunchFailed(str(exc)) from None
        try:
            state = lib.ensure_host(settings, comfy, log=self.log.info)
            self.log.debug("agentY host: %s", state)
        except Exception:  # noqa: BLE001 - the session works without the agent; never fail the launch on it
            self.log.warning("agentY: could not start the agentY host", exc_info=True)
