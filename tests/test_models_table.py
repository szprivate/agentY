"""The orchestrator's prompt points at the model tools instead of listing every model.

The table of installed models was ~10.7k tokens with ~400 models: more than half
of the prompt, sent with every step, for something the agent looks up with
check_model / find_local_models anyway. models_table_in_prompt = true restores it.
"""

import unittest
from unittest import mock

from src import agent

MODELS = {"available": {"loras": ["WAN22\\a_lora.safetensors"], "vae": ["WAN21\\wan_2.1_vae.safetensors"]}}


def _prompt(setting):
    settings = {} if setting is None else {"models_table_in_prompt": setting}
    with mock.patch.object(agent, "_settings", return_value=settings), \
         mock.patch.object(agent, "_models", return_value=MODELS):
        return agent._load_system_prompt("orchestrator")


class ModelsInThePrompt(unittest.TestCase):

    def test_by_default_it_names_the_tools_not_the_models(self):
        text = _prompt(None)
        self.assertNotIn("{{MODEL_TABLE}}", text)
        self.assertNotIn("| shortname", text)
        self.assertNotIn("a_lora.safetensors", text)
        self.assertIn("## Models", text)
        self.assertIn("find_local_models", text.split("## Models")[-1])
        self.assertIn("check_model", text.split("## Models")[-1])

    def test_the_setting_puts_the_table_back(self):
        text = _prompt(True)
        self.assertIn("| shortname", text)
        self.assertIn("WAN22\\a_lora.safetensors", text)

    def test_the_pointer_is_a_fraction_of_the_table(self):
        self.assertLess(len(agent._MODELS_POINTER), 600)


if __name__ == "__main__":
    unittest.main()
