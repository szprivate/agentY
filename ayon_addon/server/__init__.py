"""agentY add-on, server side: only the settings."""

from typing import Type

from ayon_server.addons import BaseServerAddon

from .settings import DEFAULT_VALUES, AgentYSettings


class AgentYAddon(BaseServerAddon):
    """Settings for agentY next to ComfyUI sessions started from the AYON launcher."""

    settings_model: Type[AgentYSettings] = AgentYSettings

    async def get_default_settings(self) -> AgentYSettings:
        settings_model_cls = self.get_settings_model()
        return settings_model_cls(**DEFAULT_VALUES)
