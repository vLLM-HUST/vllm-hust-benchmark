"""HMP protocol fixtures only; no guest or actual socket operations."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import osworld_owned_qemu_provider as provider


def profile():
    return {
        "scope": "operator-owned-disposable-qemu",
        "monitor_socket": "/control/monitor.sock",
        "snapshot_name": "qualified_baseline",
        "ports": {"server": 15000, "chromium": 19222, "vnc": 0, "vlc": 0},
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "shared-production"),
        ("monitor_socket", "relative/socket"),
        ("snapshot_name", "unsafe;command"),
    ],
)
def test_unsafe_operator_profile_rejected(field, value):
    data = profile()
    data[field] = value
    with pytest.raises(ValueError):
        provider.validate_profile(data)


def test_optional_ports_may_be_disabled_but_controller_must_be_real():
    data = profile()
    assert provider.validate_profile(data)["ports"]["vnc"] == 0
    data["ports"]["server"] = 0
    with pytest.raises(ValueError, match="ports"):
        provider.validate_profile(data)


def test_baseline_restore_happens_once_and_no_overwrite(monkeypatch):
    monkeypatch.setattr(provider.Path, "is_socket", lambda self: True)
    monkeypatch.setattr(provider.Path, "is_symlink", lambda self: False)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(provider, "urlopen", lambda *args, **kwargs: Response())
    instance = provider.OwnedQemuProvider(profile())
    commands = []

    def command(value):
        commands.append(value)
        return (
            "\n-- qualified_baseline 3.2GiB\n"
            if value == "info snapshots"
            else "(qemu)"
        )

    monkeypatch.setattr(instance, "command", command)
    instance.start_emulator("operator-owned-qemu", True, "Ubuntu")
    instance.start_emulator("operator-owned-qemu", True, "Ubuntu")
    assert commands == ["info snapshots", "loadvm qualified_baseline", "cont", "cont"]
    assert instance.get_ip_address("operator-owned-qemu") == "127.0.0.1:15000:19222:0:0"
    with pytest.raises(RuntimeError, match="overwritten"):
        instance.save_state("operator-owned-qemu", "qualified_baseline")
    with pytest.raises(ValueError, match="declared baseline"):
        instance.revert_to_snapshot("operator-owned-qemu", "another")
    instance.stop_emulator("operator-owned-qemu")
    assert commands[-1] == "stop"
