import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "mod_custody", SCRIPTS / "migrate_mod_release_custody.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("name", ["../escape", "/absolute", ".", "..", ""])
def test_unsafe_names(name):
    with pytest.raises(ValueError):
        module.validate_asset({"name": name, "digest": "sha256:abc"})


def test_missing_digest():
    with pytest.raises(ValueError):
        module.validate_asset({"name": "archive.tgz"})


def test_exact_scope_and_visibility():
    assert sum(map(len, module.APPROVED.values())) == 5
    with pytest.raises(ValueError):
        module.check_visibility(
            {"draft": True, "prerelease": True}, {"draft": False, "prerelease": True}
        )
