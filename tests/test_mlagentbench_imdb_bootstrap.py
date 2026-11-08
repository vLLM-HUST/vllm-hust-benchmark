import os
from pathlib import Path
import subprocess
import sys


def test_child_python_inherits_missing_packages_without_shadowing_base(tmp_path):
    compatibility = tmp_path / "compat"
    supplemental = tmp_path / "supplemental"
    compatibility.mkdir()
    supplemental.mkdir()
    (compatibility / "imdb_bootstrap_fixture.py").write_text("value = 17\n")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(
            Path(__file__).resolve().parents[1]
            / "configs/workload-native/mlagentbench-imdb-bootstrap"
        ),
        MLAGENTBENCH_COMPATIBILITY_DIR=str(compatibility),
        MLAGENTBENCH_SUPPLEMENTAL_DIR=str(supplemental),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, imdb_bootstrap_fixture as f; "
            "assert f.value == 17; "
            "assert sys.path[-2:] == sys.argv[1:]",
            str(compatibility.resolve()),
            str(supplemental.resolve()),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
