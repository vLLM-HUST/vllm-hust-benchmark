"""Fix only upstream's cosmetic description of native sentinel actions.

The pinned parser already creates DONE/FAIL/WAIT. Its final description fallback
indexes a missing dot for a sentinel without an Action: line. Copy that exact
method with one guarded description expression; do not repair model actions.
"""

import inspect
import textwrap

NEEDLE = 'action_type = pyautogui_code[0].split(".", 1)[1].split("(", 1)[0]'


def sentinel_description_compatible_agent(original):
    method = original.parse_response
    source = textwrap.dedent(inspect.getsource(method))
    if source.count(NEEDLE) != 1:
        raise ValueError("exact pinned upstream description expression required")
    patched = source.replace(
        NEEDLE, NEEDLE + ' if "." in pyautogui_code[0] else pyautogui_code[0]'
    )
    namespace = {}
    # Only a caller-verified original method is compiled; no model output or
    # task program enters this code. Preserve original globals and parser body.
    exec(
        compile(patched, method.__code__.co_filename, "exec"),
        method.__globals__,
        namespace,
    )

    class SentinelDescriptionCompatibleAgent(original):
        parse_response = namespace[method.__name__]

    return SentinelDescriptionCompatibleAgent
