"""Explicit disposable-QEMU provider port for the original DesktopEnv interface.

No VM creation, baseline overwrite, Docker socket, task rewrite or local action
execution. The operator must provision and qualify the named baseline separately.
Only this opt-in provider uses HMP; the default author Docker provider is unchanged.
"""

from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def validate_profile(profile):
    if (
        not isinstance(profile, dict)
        or profile.get("scope") != "operator-owned-disposable-qemu"
    ):
        raise ValueError("explicit operator-owned disposable QEMU scope required")
    monitor = profile.get("monitor_socket", "")
    if (
        not isinstance(monitor, str)
        or not monitor.startswith("/")
        or len(monitor.encode()) >= 108
    ):
        raise ValueError(
            "short absolute Linux monitor socket path required; bind it to /control/monitor.sock"
        )
    if ".." in Path(monitor).parts:
        raise ValueError("contained explicit monitor socket path required")
    snapshot = profile.get("snapshot_name", "")
    if not isinstance(snapshot, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", snapshot
    ):
        raise ValueError("explicit safe existing baseline snapshot name required")
    ports = profile.get("ports", {})
    if not isinstance(ports, dict) or set(ports) != {
        "server",
        "chromium",
        "vnc",
        "vlc",
    }:
        raise ValueError(
            "explicit server/chromium/vnc/vlc port map required; use0 for unused optional ports"
        )
    for name, value in ports.items():
        minimum = 1 if name in {"server", "chromium"} else 0
        if type(value) is not int or not minimum <= value <= 65535:
            raise ValueError("valid loopback ports required")
    return {"monitor_socket": monitor, "snapshot_name": snapshot, "ports": dict(ports)}


class OwnedQemuProvider:
    def __init__(self, profile):
        self.profile = validate_profile(profile)
        self.initial = True
        path = Path(self.profile["monitor_socket"])
        if path.is_symlink() or not path.is_socket():
            raise ValueError("existing direct owned monitor socket required")

    def command(self, value):
        with socket.socket(socket.AF_UNIX) as monitor:
            monitor.settimeout(180)
            monitor.connect(self.profile["monitor_socket"])

            def prompt():
                data = b""
                while b"(qemu)" not in data:
                    block = monitor.recv(65536)
                    if not block:
                        raise RuntimeError("Owned monitor closed")
                    data += block
                return data.decode(errors="replace")

            prompt()
            monitor.sendall(value.encode() + b"\r")
            result = prompt()
        if "Error:" in result or "unknown command" in result.lower():
            raise RuntimeError(
                "Owned QEMU command failed; retain original private diagnostic"
            )
        return result

    def start_emulator(self, path_to_vm, headless, os_type):
        if path_to_vm != "operator-owned-qemu":
            raise ValueError("owned provider identity differs")
        if self.initial:
            snapshot = self.profile["snapshot_name"]
            listing = self.command("info snapshots")
            if not re.search(
                r"\n\s*(?:\d+|--)\s+" + re.escape(snapshot) + r"\s+", listing
            ):
                raise ValueError("qualified baseline not present on all VM disks")
            self.command("loadvm " + snapshot)
            self.initial = False
        self.command("cont")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                with urlopen(
                    f"http://127.0.0.1:{self.profile['ports']['server']}/platform",
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        return
            except (HTTPError, URLError, TimeoutError):
                time.sleep(1)
        raise TimeoutError(
            "Original controller readiness deadline; no automatic VM restart"
        )

    def get_ip_address(self, path_to_vm):
        ports = self.profile["ports"]
        return "127.0.0.1:" + ":".join(
            str(ports[name]) for name in ["server", "chromium", "vnc", "vlc"]
        )

    def revert_to_snapshot(self, path_to_vm, snapshot_name):
        if (
            path_to_vm != "operator-owned-qemu"
            or snapshot_name != self.profile["snapshot_name"]
        ):
            raise ValueError("only the declared baseline may be restored")
        self.command("loadvm " + snapshot_name)
        return path_to_vm

    def save_state(self, *_args):
        raise RuntimeError("Baseline cannot be overwritten during native tasks")

    def stop_emulator(self, path_to_vm):
        if path_to_vm != "operator-owned-qemu":
            raise ValueError("owned provider identity differs")
        self.command("stop")  # Release pauses only the explicitly owned guest.
