from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/sample_ascend_peak_hbm.py"
SPEC = spec_from_file_location("sample_ascend_peak_hbm", SCRIPT)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_hbm_usage() -> None:
    output = """
| 2     910B2               | OK            | 92.1                 38                      0    / 0                |
| 0                         | 0000:81:00.0  | 0                    0    / 0                3417 / 65536          |
| 7     910B2               | OK            | 175.0                51                      0    / 0                |
| 0                         | 0000:42:00.0  | 100                  0    / 0                61203/ 65536          |
"""
    assert MODULE.parse_hbm_usage(output) == {
        2: (3417, 65536),
        7: (61203, 65536),
    }
