"""ElevenLabs' play_audio is not given to the agent.

It played a file on the speakers of the machine agentY runs on and answered only
once the sound had ended, so a turn waited for the whole clip and a user at a
browser elsewhere heard nothing. A sound any tool makes is shown in the chat
panel as a player instead (src/utils/media_sink.py -> the panel's sound card).
"""

import unittest
from pathlib import Path
from unittest import mock

from strands import tool

from src.tools import mcp_on_demand as od
from src.tools import mcp_tools


@tool(name="elevenlabs__text_to_speech")
def _tts(text: str) -> str:
    """Speak."""
    return "File saved as: /tmp/x.mp3"


@tool(name="elevenlabs__play_audio")
def _play(input_file_path: str) -> str:
    """Play."""
    return "played"


@tool(name="other__play_audio")
def _other_play(input_file_path: str) -> str:
    """Another server's tool of the same name."""
    return "played"


class _Client:
    def __init__(self, tools):
        self._tools = tools

    def list_tools_sync(self):
        return list(self._tools)


def _names(tools):
    return [t.tool_name for t in tools]


class HiddenTools(unittest.TestCase):
    def test_play_audio_dropped_for_elevenlabs(self):
        self.assertEqual(_names(mcp_tools.visible_tools("elevenlabs", [_tts, _play])),
                         ["elevenlabs__text_to_speech"])

    def test_a_renamed_elevenlabs_server_too(self):
        @tool(name="ElevenLabs_2__play_audio")
        def _p(x: str) -> str:
            """Play."""
            return ""
        self.assertEqual(mcp_tools.visible_tools("ElevenLabs_2", [_p]), [])

    def test_other_servers_keep_theirs(self):
        self.assertEqual(_names(mcp_tools.visible_tools("other", [_other_play])),
                         ["other__play_audio"])

    def test_startup_list_leaves_it_out(self):
        cfg = {"servers": {"elevenlabs": {"enabled": True}}}
        with mock.patch.object(mcp_tools, "load_mcp_config", return_value=cfg), \
                mock.patch.dict(mcp_tools._CLIENTS, {"elevenlabs": _Client([_tts, _play])}, clear=True), \
                mock.patch.dict(mcp_tools._LISTS, {}, clear=True), \
                mock.patch.dict(mcp_tools._STATUS, {}, clear=True):
            got = mcp_tools.load_mcp_tools()
            self.assertEqual(_names(got), ["elevenlabs__text_to_speech"])
            self.assertEqual(mcp_tools._STATUS["elevenlabs"], "connected (1)")

    def test_on_demand_list_leaves_it_out(self):
        client = _Client([_tts, _play])
        with mock.patch.dict(mcp_tools._CLIENTS, {"elevenlabs": client}, clear=True), \
                mock.patch.dict(mcp_tools._LISTS, {}, clear=True):
            self.assertEqual(_names(od.server_tools("elevenlabs")), ["elevenlabs__text_to_speech"])

    def test_prompt_says_sounds_play_in_the_panel(self):
        prompt = (Path(__file__).resolve().parents[1] / "config" / "system_prompts"
                  / "system_prompt.orchestrator.md").read_text(encoding="utf-8")
        self.assertIn("You never play a sound yourself", prompt)


if __name__ == "__main__":
    unittest.main()
