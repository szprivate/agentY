"""agentY add-on settings (studio and project level)."""

from typing import Any

from ayon_server.settings import BaseSettingsModel, SettingsField


def _start_host_enum() -> list[dict[str, str]]:
    return [
        {"value": "if_not_running", "label": "Start it when it is not running"},
        {"value": "never", "label": "Never (start it yourself)"},
    ]


def _port_busy_enum() -> list[dict[str, str]]:
    return [
        {"value": "block", "label": "Stop the launch and say why"},
        {"value": "stop_if_idle", "label": "Close the old ComfyUI when its queue is empty"},
        {"value": "ignore", "label": "Launch anyway"},
    ]


class MultiplatformPath(BaseSettingsModel):
    windows: str = SettingsField("", title="Windows")
    linux: str = SettingsField("", title="Linux")
    darwin: str = SettingsField("", title="MacOS")


class AgentYSettings(BaseSettingsModel):
    """agentY next to ComfyUI sessions launched from AYON."""

    enabled: bool = SettingsField(True, title="Enabled")

    agenty_root: MultiplatformPath = SettingsField(
        default_factory=MultiplatformPath,
        title="agentY folder",
        description=(
            "The agentY checkout on the artist's machine (the folder with run_agent.ps1 / run_agent.sh), "
            "e.g. D:\\AI\\agentY. Needed to start the agentY host; empty = never start it."
        ),
    )

    agent_server_url: str = SettingsField(
        "http://127.0.0.1:5000",
        title="agentY host address",
        description="Where the agentY host answers (the agent_server_url of agentY's settings).",
    )

    start_host: str = SettingsField(
        "if_not_running",
        title="agentY host on ComfyUI launch",
        enum_resolver=_start_host_enum,
        description=(
            "When ComfyUI is launched from AYON: start the agentY host if it does not answer at the address "
            "above. A host that already runs is left alone."
        ),
    )

    host_args: list[str] = SettingsField(
        default_factory=list,
        title="Extra arguments for run_agent",
        description="Passed to run_agent.ps1 / run_agent.sh, e.g. -NoUpdate (Windows) or --no-update.",
    )

    comfyui_port_busy: str = SettingsField(
        "block",
        title="When a ComfyUI already runs on the launch port",
        enum_resolver=_port_busy_enum,
        description=(
            "AYON cannot start ComfyUI on a port another ComfyUI holds: the session would attach to the old "
            "one and break. Checked against the ports of the ComfyUI add-on's local launch profiles. "
            "'Close the old ComfyUI' only does so when nothing is queued or rendering on it."
        ),
    )


DEFAULT_VALUES: dict[str, Any] = {
    "enabled": True,
    "agenty_root": {"windows": "", "linux": "", "darwin": ""},
    "agent_server_url": "http://127.0.0.1:5000",
    "start_host": "if_not_running",
    "host_args": [],
    "comfyui_port_busy": "block",
}
