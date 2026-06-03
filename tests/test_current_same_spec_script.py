from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_RUNNER_SCRIPT = REPO_ROOT / "scripts/run-current-ascend-same-spec.sh"


def test_current_same_spec_runner_reuses_selected_ascend_device() -> None:
    script_text = CURRENT_RUNNER_SCRIPT.read_text(encoding="utf-8")

    assert (
        'CURRENT_DEVICE_PREFERENCE_FILE=${CURRENT_DEVICE_PREFERENCE_FILE:-${GOAL_BASELINE_DEVICE_PREFERENCE_FILE:-}}'
        in script_text
    )
    assert '[same-spec-current] reusing Ascend device from preference file:' in script_text
    assert 'export ASCEND_VISIBLE_DEVICES="$visible_devices"' in script_text
    assert 'export ASCEND_RT_VISIBLE_DEVICES="$rt_visible_devices"' in script_text