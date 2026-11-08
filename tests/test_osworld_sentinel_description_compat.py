"""Parser-description fixtures only; no GUI, model or native scoring."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from osworld_sentinel_description_compat import sentinel_description_compatible_agent


class Original:
    def parse_response(self, response):
        pyautogui_code = response
        low_level_instruction = ""
        if not low_level_instruction and len(pyautogui_code) > 0:
            action_type = pyautogui_code[0].split(".", 1)[1].split("(", 1)[0]
            low_level_instruction = f"Performing {action_type} action"
        return low_level_instruction, pyautogui_code


@pytest.mark.parametrize(
    "code", [["DONE"], ["FAIL"], ["WAIT"], ["pyautogui.click(1, 2)"], []]
)
def test_only_description_changes_original_actions_are_identical(code):
    method = Original.parse_response
    compatible = sentinel_description_compatible_agent(Original)()
    description, actions = compatible.parse_response(code)
    assert actions is code and Original.parse_response is method
    if code and code[0] in {"DONE", "FAIL", "WAIT"}:
        with pytest.raises(IndexError):
            Original().parse_response(code)
        assert description == f"Performing {code[0]} action"
    else:
        assert (description, actions) == Original().parse_response(code)


def test_unrecognized_upstream_method_refused():
    class Other:
        def parse_response(self, response):
            return response

    with pytest.raises(ValueError, match="exact pinned"):
        sentinel_description_compatible_agent(Other)
