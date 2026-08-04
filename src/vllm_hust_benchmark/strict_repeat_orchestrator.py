"""Fail-closed host-side orchestration for one strict benchmark repeat.

This module deliberately runs on the host, not inside the benchmark container.
Container-side ``npu-smi`` output is never used for admission or evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


STRICT_SCHEMA = "strict-execution-evidence/v1"
CLEANUP_SCHEMA = "cleanup-chain-attestation/v1"
FAILURE_SCHEMA = "strict-repeat-failure/v1"
DRY_RUN_SCHEMA = "strict-repeat-dry-run/v1"
CONTAINER_ID_RE = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
DEVICE_ROW = re.compile(r"^\|\s*(\d+)\s+\S+\s+\|")
HBM_CELL = re.compile(r"(\d+)\s*/\s*(\d+)")
PROCESS_ROW = re.compile(r"^\|\s*(\d+)\s+(?:\d+\s+)?(\d+)\s+\S+")


class GateFailure(RuntimeError):
    """A condition that makes the repeat ineligible."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evidence_record(repeat_dir: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(repeat_dir)),
        "sha256": sha256_file(path),
    }


def build_strict_evidence(
    *,
    hostname: str,
    startup_id: str,
    target_id: str,
    side: str,
    container_id: str,
    service_port: int,
    runtime_image_digest: str,
    immutable_inputs: dict[str, str],
    devices: list[int],
    acquired_at: str,
    released_at: str,
    snapshots: list[dict[str, Any]],
    ownership: list[dict[str, Any]],
    hbm_samples: dict[str, str],
    peak_hbm_mb: int | float,
    cleanup: dict[str, str],
) -> dict[str, Any]:
    """Build the exact payload consumed by the official attestation validator."""
    return {
        "schema_version": STRICT_SCHEMA,
        "hostname": hostname,
        "startup_instance_id": startup_id,
        "target_id": target_id,
        "side": side,
        "container_id": container_id,
        "service_port": service_port,
        "runtime_image_digest": runtime_image_digest,
        "immutable_inputs": immutable_inputs,
        "lease": {
            "physical_npu_ids": devices,
            "acquired_at": acquired_at,
            "released_at": released_at,
        },
        "pre_start_snapshots": snapshots,
        "ownership": ownership,
        "hbm_samples": hbm_samples,
        "peak_hbm_mb": peak_hbm_mb,
        "cleanup": cleanup,
    }


def parse_hbm_usage(output: str) -> dict[int, tuple[int, int]]:
    """Parse physical NPU id -> (used MiB, capacity MiB)."""
    lines = output.splitlines()
    parsed: dict[int, tuple[int, int]] = {}
    for index, line in enumerate(lines[:-1]):
        match = DEVICE_ROW.match(line)
        if not match:
            continue
        cells = lines[index + 1].split("|")
        if len(cells) < 3:
            continue
        matches = HBM_CELL.findall(cells[-2])
        if matches:
            used, capacity = matches[-1]
            parsed[int(match.group(1))] = (int(used), int(capacity))
    return parsed


def parse_compute_pids(output: str) -> dict[int, list[int]]:
    """Parse the process table in host ``npu-smi info`` output."""
    in_process_table = False
    parsed: dict[int, list[int]] = {}
    for line in output.splitlines():
        if "Process id" in line or "Process ID" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        match = PROCESS_ROW.match(line)
        if not match:
            continue
        device, pid = (int(value) for value in match.groups())
        if pid > 0:
            parsed.setdefault(device, []).append(pid)
    return {device: sorted(set(pids)) for device, pids in parsed.items()}


def port_listener_pids(output: str, port: int) -> list[int]:
    listeners: set[int] = set()
    for line in output.splitlines():
        if not re.search(rf":{port}(?:\s|$)", line):
            continue
        listeners.update(int(value) for value in re.findall(r"pid=(\d+)", line))
        if not re.findall(r"pid=(\d+)", line):
            listeners.add(-1)
    return sorted(listeners)


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


class RootCommands:
    """Run allowlisted read-only host commands through root or ``sudo -n``."""

    def __init__(self) -> None:
        self.prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        if self.prefix:
            probe = subprocess.run(
                [*self.prefix, "true"], capture_output=True, text=True, check=False
            )
            if probe.returncode:
                raise GateFailure(f"sudo -n is unavailable: {probe.stderr.strip()}")

    def run(self, argv: Sequence[str], *, check: bool = True) -> CommandResult:
        result = subprocess.run(
            [*self.prefix, *argv],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if check and result.returncode:
            raise GateFailure(
                f"host command failed ({shlex.join(argv)}): {result.stderr.strip()}"
            )
        return CommandResult(result.stdout, result.stderr, result.returncode)


class ResourceLease:
    """Non-blocking locks for every card, port, and target container."""

    def __init__(
        self,
        root: Path,
        *,
        startup_id: str,
        container_id: str,
        devices: list[int],
        port: int,
        repeat_dir: Path,
    ) -> None:
        self.root = root
        self.startup_id = startup_id
        self.container_id = container_id
        self.devices = devices
        self.port = port
        self.repeat_dir = repeat_dir
        self.handles: list[Any] = []
        self.acquired_at = ""
        self.released_at = ""
        self.global_path = root / "leases" / f"{startup_id}.json"

    def bind_container_id(self, container_id: str) -> None:
        if not CONTAINER_ID_RE.fullmatch(container_id):
            raise GateFailure("owned container ID is not immutable")
        self.container_id = container_id
        self._write_global(active=True)

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        resources = [f"npu-{device}" for device in self.devices]
        resources.extend((f"port-{self.port}", f"container-{self.container_id}"))
        for resource in sorted(resources):
            path = self.root / f"{resource}.lock"
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                self.release()
                raise GateFailure(f"resource lease is busy: {resource}") from error
            self.handles.append(handle)
        self.acquired_at = utc_now()
        self._write_global(active=True)

    def conflicts(self) -> list[str]:
        conflicts: list[str] = []
        lease_dir = self.root / "leases"
        if not lease_dir.is_dir():
            return conflicts
        for path in lease_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                conflicts.append(str(path))
                continue
            if payload.get("startup_instance_id") == self.startup_id:
                continue
            if not payload.get("active"):
                continue
            overlap = set(payload.get("physical_npu_ids") or {}) & set(self.devices)
            if (
                overlap
                or payload.get("service_port") == self.port
                or payload.get("container_id") == self.container_id
            ):
                conflicts.append(str(path))
        return sorted(conflicts)

    def mark_released(self) -> str:
        self.released_at = utc_now()
        self._write_global(active=False)
        for handle in reversed(self.handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        self.handles.clear()
        return self.released_at

    def release(self) -> None:
        for handle in reversed(self.handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        self.handles.clear()

    def _write_global(self, *, active: bool) -> None:
        payload = {
            "schema_version": "strict-host-resource-lease/v1",
            "startup_instance_id": self.startup_id,
            "container_id": self.container_id,
            "physical_npu_ids": self.devices,
            "service_port": self.port,
            "repeat_dir": str(self.repeat_dir),
            "acquired_at": self.acquired_at,
            "released_at": self.released_at or None,
            "active": active,
        }
        atomic_json(self.global_path, payload)
        if self.repeat_dir.is_dir():
            atomic_json(self.repeat_dir / "resource-lease.json", payload)


@dataclass
class Snapshot:
    captured_at: str
    hbm: dict[int, int]
    compute_pids: dict[int, list[int]]
    summary: dict[str, Any]


class StrictRepeatOrchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repeat_dir = args.repeat_dir.resolve()
        try:
            relative_repeat = self.repeat_dir.relative_to(args.repo_host_path.resolve())
        except ValueError as error:
            raise GateFailure(
                "repeat directory must be inside the mounted benchmark repository"
            ) from error
        self.container_repeat_dir = (
            Path("/workspace/vllm-hust-benchmark") / relative_repeat
        )
        self.devices = args.physical_npu
        self._validate_target_scope()
        self.startup_id = uuid.uuid4().hex
        self.hostname = socket.gethostname()
        self.root = RootCommands()
        self.container_name = f"{args.container_name_prefix}-{self.startup_id}"
        self.container_id = ""
        self.lease = ResourceLease(
            args.lease_dir,
            startup_id=self.startup_id,
            container_id=self.container_name,
            devices=self.devices,
            port=args.service_port,
            repeat_dir=self.repeat_dir,
        )
        self.ownership: list[dict[str, Any]] = []
        self.host_pids: list[int] = []
        self.hbm_path = self.repeat_dir / "hbm-samples.jsonl"
        self.host_peak_path = self.repeat_dir / "strict-host-peak-hbm.json"
        self.per_device_peaks = {device: 0 for device in self.devices}
        self.hbm_sample_count = 0

    def _validate_target_scope(self) -> None:
        registry_path = Path(__file__).parent / "data" / "official_targets.json"
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            targets = registry["targets"]
            target = next(
                item for item in targets if item.get("target_id") == self.args.target_id
            )
        except (
            KeyError,
            OSError,
            StopIteration,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            raise GateFailure(
                f"official target is not registered: {self.args.target_id}"
            ) from error
        chip_count = int((target.get("hardware") or {}).get("chip_count") or 0)
        if chip_count != len(self.devices):
            raise GateFailure(
                f"physical NPU scope does not match target chip count: {len(self.devices)} != {chip_count}"
            )
        expected_digest = str(
            (target.get("baseline_runtime") or {}).get("runtime_image_digest") or ""
        )
        if not IMAGE_DIGEST_RE.fullmatch(expected_digest):
            raise GateFailure(
                "official target does not pin an immutable runtime image digest"
            )
        if expected_digest != self.args.runtime_image_digest:
            raise GateFailure(
                f"runtime image does not match official target: "
                f"{self.args.runtime_image_digest} != {expected_digest}"
            )

    def _resolve_container(self, container: str) -> str:
        result = self.root.run(["docker", "inspect", container])
        try:
            payload = json.loads(result.stdout)
            record = payload[0]
            container_id = str(record["Id"])
            image = str(record["Image"])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise GateFailure("target container inspect is malformed") from error
        if not CONTAINER_ID_RE.fullmatch(container_id):
            raise GateFailure("target container did not resolve to a full immutable ID")
        if image != self.args.runtime_image_digest:
            raise GateFailure(
                f"runtime image mismatch: {image} != {self.args.runtime_image_digest}"
            )
        return container_id

    def _docker_create_argv(self) -> list[str]:
        argv = [
            "docker",
            "create",
            "--rm",
            "--name",
            self.container_name,
            "--label",
            f"vllm-hust.strict-startup-id={self.startup_id}",
            "--network",
            "none",
            "--ipc",
            "host",
            "--workdir",
            "/tmp",
            "--env",
            "ASCEND_VISIBLE_DEVICES=" + ",".join(map(str, range(len(self.devices)))),
            "--env",
            "ASCEND_RT_VISIBLE_DEVICES=" + ",".join(map(str, range(len(self.devices)))),
            "--env",
            "VLLM_HUST_STRICT_HOST_ORCHESTRATED=1",
            "--env",
            (
                "VLLM_HUST_STRICT_HOST_PEAK_HBM_FILE="
                f"{self.container_repeat_dir}/strict-host-peak-hbm.json"
            ),
            "--mount",
            (
                f"type=bind,src={self.args.repo_host_path.resolve()},"
                "dst=/workspace/vllm-hust-benchmark"
            ),
            "--mount",
            (
                f"type=bind,src={self.args.shared_data_host_path.resolve()},"
                "dst=/data/shared_models,readonly"
            ),
            "--mount",
            (
                f"type=bind,src={self.args.shared_data_host_path.resolve()},"
                "dst=/data/shared_datasets,readonly"
            ),
            "--mount",
            "type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly",
        ]
        for logical, physical in enumerate(self.devices):
            argv.extend(["--device", f"/dev/davinci{physical}:/dev/davinci{logical}"])
        for device in ("davinci_manager", "devmm_svm", "hisi_hdc"):
            argv.extend(["--device", f"/dev/{device}:/dev/{device}"])
        argv.extend([self.args.runtime_image_digest, *self.args.command])
        return argv

    def _create_owned_container(self) -> None:
        for path, description in (
            (self.args.repo_host_path, "benchmark repository"),
            (self.args.shared_data_host_path, "shared model/data root"),
            (Path("/usr/local/Ascend/driver"), "Ascend driver mount"),
        ):
            if not path.resolve().exists():
                raise GateFailure(f"{description} is missing: {path}")
        for device in self.devices:
            if not Path(f"/dev/davinci{device}").exists():
                raise GateFailure(f"physical NPU device node is missing: {device}")
        for device in ("davinci_manager", "devmm_svm", "hisi_hdc"):
            if not Path(f"/dev/{device}").exists():
                raise GateFailure(f"Ascend management device is missing: /dev/{device}")
        result = self.root.run(self._docker_create_argv())
        candidate = result.stdout.strip()
        if CONTAINER_ID_RE.fullmatch(candidate):
            self.container_id = candidate
        self.container_id = self._resolve_container(candidate)
        self.lease.bind_container_id(self.container_id)
        inspect = self.root.run(["docker", "inspect", self.container_id])
        try:
            inspect_record = json.loads(inspect.stdout)[0]
        except (IndexError, TypeError, json.JSONDecodeError) as error:
            raise GateFailure("owned container inspect is malformed") from error
        self._validate_owned_container_inspect(inspect_record)
        inspect_path = self.repeat_dir / "owned-container-create-inspect.json"
        inspect_path.write_text(inspect.stdout, encoding="utf-8")
        atomic_json(
            self.repeat_dir / "owned-container-identity.json",
            {
                "container_id": self.container_id,
                "container_name": self.container_name,
                "runtime_image_digest": self.args.runtime_image_digest,
                "create_argv": self._docker_create_argv(),
                "inspect": evidence_record(self.repeat_dir, inspect_path),
                "physical_to_logical_npu": {
                    str(physical): logical
                    for logical, physical in enumerate(self.devices)
                },
            },
        )

    def _validate_owned_container_inspect(self, record: dict[str, Any]) -> None:
        config = record.get("Config") or {}
        host_config = record.get("HostConfig") or {}
        labels = config.get("Labels") or {}
        if record.get("Id") != self.container_id:
            raise GateFailure("owned container inspect ID mismatch")
        if record.get("Image") != self.args.runtime_image_digest:
            raise GateFailure("owned container OCI config digest mismatch")
        if labels.get("vllm-hust.strict-startup-id") != self.startup_id:
            raise GateFailure("owned container label mismatch")
        if host_config.get("AutoRemove") is not True:
            raise GateFailure("owned container is not configured with --rm")
        if host_config.get("NetworkMode") != "none":
            raise GateFailure("owned container network mode is not none")
        if host_config.get("IpcMode") != "host":
            raise GateFailure("owned container IPC mode is not host")

        actual_devices = {
            (item.get("PathOnHost"), item.get("PathInContainer"))
            for item in (host_config.get("Devices") or [])
        }
        expected_devices = {
            (f"/dev/davinci{physical}", f"/dev/davinci{logical}")
            for logical, physical in enumerate(self.devices)
        }
        expected_devices.update(
            (f"/dev/{name}", f"/dev/{name}")
            for name in ("davinci_manager", "devmm_svm", "hisi_hdc")
        )
        if actual_devices != expected_devices:
            raise GateFailure(
                f"owned container device mapping mismatch: {actual_devices}"
            )

        mounts = {
            (item.get("Source"), item.get("Destination")): bool(item.get("RW"))
            for item in (record.get("Mounts") or [])
        }
        repo_key = (
            str(self.args.repo_host_path.resolve()),
            "/workspace/vllm-hust-benchmark",
        )
        shared_key = (
            str(self.args.shared_data_host_path.resolve()),
            "/data/shared_models",
        )
        shared_dataset_key = (
            str(self.args.shared_data_host_path.resolve()),
            "/data/shared_datasets",
        )
        driver_key = ("/usr/local/Ascend/driver", "/usr/local/Ascend/driver")
        if mounts.get(repo_key) is not True:
            raise GateFailure(
                "owned container repository mount is missing or read-only"
            )
        if mounts.get(shared_key) is not False:
            raise GateFailure(
                "owned container shared data mount is missing or writable"
            )
        if mounts.get(shared_dataset_key) is not False:
            raise GateFailure("owned container dataset mount is missing or writable")
        if mounts.get(driver_key) is not False:
            raise GateFailure("owned container driver mount is missing or writable")

    def _host_snapshot(self, number: int) -> Snapshot:
        directory = self.repeat_dir / f"pre-start-{number}"
        directory.mkdir(parents=True, exist_ok=True)
        captured_at = utc_now()
        npu = self.root.run(["npu-smi", "info"])
        npu_list = self.root.run(["npu-smi", "info", "-l"])
        containers = self.root.run(["docker", "ps", "-aq"])
        ids = [value for value in containers.stdout.splitlines() if value]
        inspect = (
            self.root.run(["docker", "inspect", *ids])
            if ids
            else CommandResult("[]\n", "", 0)
        )
        ss = self.root.run(["ss", "-H", "-lntup"])
        network_sockets = self.root.run(["ss", "-H", "-ntup"])
        who = self.root.run(["who"], check=False)
        login_sessions = self.root.run(
            ["loginctl", "list-sessions", "--no-legend"], check=False
        )
        tmux = self.root.run(["tmux", "ls"], check=False)
        screen = self.root.run(["screen", "-ls"], check=False)
        processes = self.root.run(["ps", "-eo", "pid=,ppid=,user=,lstart=,args="])
        fd_holders: dict[int, list[int]] = {}
        fuser_raw: dict[str, dict[str, Any]] = {}
        for device in self.devices:
            fuser = self.root.run(["fuser", "-v", f"/dev/davinci{device}"], check=False)
            holders = sorted(
                {int(value) for value in re.findall(r"\b\d+\b", fuser.stdout)}
            )
            fd_holders[device] = holders
            fuser_raw[str(device)] = {
                "stdout": fuser.stdout,
                "stderr": fuser.stderr,
                "returncode": fuser.returncode,
            }

        npu_path = directory / "npu-smi.txt"
        npu_path.write_text(
            "$ sudo -n npu-smi info\n"
            + npu.stdout
            + "\n$ sudo -n npu-smi info -l\n"
            + npu_list.stdout,
            encoding="utf-8",
        )
        inspect_path = directory / "docker-inspect-and-host-context.json"
        atomic_json(
            inspect_path,
            {
                "docker_inspect": json.loads(inspect.stdout),
                "listening_sockets": ss.stdout,
                "network_sockets": network_sockets.stdout,
                "who": who.stdout,
                "login_sessions": login_sessions.stdout,
                "tmux": tmux.stdout,
                "screen": screen.stdout,
                "processes": processes.stdout,
                "device_fuser": fuser_raw,
            },
        )

        hbm_full = parse_hbm_usage(npu.stdout)
        missing = [device for device in self.devices if device not in hbm_full]
        if missing:
            raise GateFailure(f"host npu-smi omitted requested devices: {missing}")
        hbm = {device: hbm_full[device][0] for device in self.devices}
        compute = parse_compute_pids(npu.stdout)
        all_compute_pids = sorted({pid for pids in compute.values() for pid in pids})
        if all_compute_pids:
            self._capture_pid_context(directory, all_compute_pids)
        external_pids = sorted(
            pid for device in self.devices for pid in compute.get(device, [])
        )
        listener_pids = port_listener_pids(ss.stdout, self.args.service_port)
        session_pattern = re.compile(
            rf"({re.escape(self.args.target_id)}|:{self.args.service_port}\b)", re.I
        )
        session_conflicts = [
            line
            for text in (tmux.stdout, screen.stdout, processes.stdout)
            for line in text.splitlines()
            if session_pattern.search(line) and str(os.getpid()) not in line
        ]
        lease_conflicts = self.lease.conflicts()
        summary = {
            "captured_at": captured_at,
            "physical_npu_ids": self.devices,
            "external_compute_pids": external_pids,
            "external_container_ids": [],
            "lease_conflicts": lease_conflicts,
            "port_conflicts": listener_pids,
            "session_conflicts": session_conflicts,
            "device_fd_holders": {
                str(device): holders for device, holders in fd_holders.items()
            },
            "stable": False,
            "npu_smi": evidence_record(self.repeat_dir, npu_path),
            "container_inspect": evidence_record(self.repeat_dir, inspect_path),
        }
        if external_pids:
            raise GateFailure(f"requested NPU has compute PIDs: {external_pids}")
        if listener_pids:
            raise GateFailure(f"service port is occupied: {listener_pids}")
        if any(fd_holders.values()):
            raise GateFailure(f"requested NPU has device FD holders: {fd_holders}")
        if session_conflicts:
            raise GateFailure("benchmark-related host session conflict")
        if lease_conflicts:
            raise GateFailure(f"resource lease conflict: {lease_conflicts}")
        if any(value > self.args.max_idle_hbm_mb for value in hbm.values()):
            raise GateFailure(f"idle HBM exceeds threshold: {hbm}")
        return Snapshot(captured_at, hbm, compute, summary)

    def _capture_pid_context(self, directory: Path, pids: list[int]) -> None:
        records = []
        for pid in sorted(set(pids)):
            cgroup = Path(f"/proc/{pid}/cgroup")
            cmdline = Path(f"/proc/{pid}/cmdline")
            records.append(
                {
                    "pid": pid,
                    "cgroup": cgroup.read_text(errors="replace")
                    if cgroup.exists()
                    else None,
                    "cmdline": (
                        cmdline.read_bytes()
                        .replace(b"\0", b" ")
                        .decode(errors="replace")
                        if cmdline.exists()
                        else None
                    ),
                }
            )
        atomic_json(directory / "pid-cgroup-context.json", records)

    def _assert_stable(self, first: Snapshot, second: Snapshot) -> None:
        drift = {
            device: abs(first.hbm[device] - second.hbm[device])
            for device in self.devices
        }
        if any(value > self.args.max_hbm_drift_mb for value in drift.values()):
            raise GateFailure(f"idle HBM is not stable: drift={drift}")
        first.summary["stable"] = True
        second.summary["stable"] = True

    def _runtime_sample(self, number: int) -> None:
        npu = self.root.run(["npu-smi", "info"])
        raw_path = self.repeat_dir / "runtime" / f"npu-smi-{number:06d}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(npu.stdout, encoding="utf-8")
        hbm = parse_hbm_usage(npu.stdout)
        compute = parse_compute_pids(npu.stdout)
        missing = [device for device in self.devices if device not in hbm]
        if missing:
            raise GateFailure(f"runtime npu-smi omitted requested devices: {missing}")

        if not self.ownership:
            ownership: list[dict[str, Any]] = []
            for device in self.devices:
                candidates = []
                for pid in compute.get(device, []):
                    cgroup_path = Path(f"/proc/{pid}/cgroup")
                    if not cgroup_path.is_file():
                        continue
                    cgroup = cgroup_path.read_text(encoding="utf-8", errors="replace")
                    if self.container_id in cgroup:
                        candidates.append((pid, cgroup))
                if len(candidates) != 1:
                    return
                pid, cgroup = candidates[0]
                ownership.append(
                    {
                        "host_pid": pid,
                        "physical_npu_id": device,
                        "container_id": self.container_id,
                        "cgroup": cgroup,
                    }
                )
            self.ownership = ownership
            self.host_pids = [item["host_pid"] for item in ownership]
            self._capture_pid_context(self.repeat_dir / "runtime", self.host_pids)

        if not self.ownership:
            return
        for item in self.ownership:
            device = item["physical_npu_id"]
            pid = item["host_pid"]
            current = compute.get(device, [])
            if pid not in current or any(value != pid for value in current):
                raise GateFailure(
                    f"runtime PID scope changed on physical NPU {device}: {current}"
                )
        per_device = {str(device): hbm[device][0] for device in self.devices}
        sample = {
            "captured_at": utc_now(),
            "host_pids": self.host_pids,
            "physical_npu_hbm_mb": per_device,
            "total_hbm_mb": sum(per_device.values()),
            "raw_npu_smi": evidence_record(self.repeat_dir, raw_path),
        }
        with self.hbm_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.hbm_sample_count += 1
        for device in self.devices:
            self.per_device_peaks[device] = max(
                self.per_device_peaks[device], hbm[device][0]
            )
        atomic_json(
            self.host_peak_path,
            {
                "schema_version": "strict-host-peak-hbm/v1",
                "generated_at": utc_now(),
                "source": "host sudo npu-smi info",
                "devices": self.devices,
                "sample_count": self.hbm_sample_count,
                "sample_failure_count": 0,
                "peak_hbm_mb": sum(self.per_device_peaks.values()),
                "per_device_peak_hbm_mb": {
                    str(device): self.per_device_peaks[device]
                    for device in self.devices
                },
            },
        )

    def _run_command(self) -> int:
        stdout_path = self.repeat_dir / "runner.log"
        stderr_path = self.repeat_dir / "runner.stderr.log"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [*self.root.prefix, "docker", "start", "--attach", self.container_id],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            atomic_json(
                self.repeat_dir / "orchestrated-command.json",
                {
                    "argv": self.args.command,
                    "docker_start_container_id": self.container_id,
                    "host_wrapper_pid": process.pid,
                    "started_at": utc_now(),
                },
            )
            sample_number = 0
            try:
                while process.poll() is None:
                    sample_number += 1
                    self._runtime_sample(sample_number)
                    time.sleep(self.args.sample_interval_seconds)
            except BaseException:
                self._cleanup_owned_container(allow_stop=True)
                process.wait(timeout=60)
                raise
            atomic_json(
                self.repeat_dir / "orchestrated-command.json",
                {
                    "argv": self.args.command,
                    "host_wrapper_pid": process.pid,
                    "docker_start_container_id": self.container_id,
                    "finished_at": utc_now(),
                    "exit_code": int(process.returncode),
                },
            )
            return int(process.returncode)

    def _container_is_stopped_or_removed(self) -> bool:
        result = self.root.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self.container_id],
            check=False,
        )
        return result.returncode != 0 or result.stdout.strip() == "false"

    def _cleanup_owned_container(self, *, allow_stop: bool) -> None:
        if not self.container_id:
            return
        result = self.root.run(["docker", "inspect", self.container_id], check=False)
        if result.returncode != 0:
            return
        try:
            record = json.loads(result.stdout)[0]
            labels = (record.get("Config") or {}).get("Labels") or {}
            running = bool((record.get("State") or {}).get("Running"))
        except (IndexError, TypeError, json.JSONDecodeError) as error:
            raise GateFailure("owned container inspect became malformed") from error
        if labels.get("vllm-hust.strict-startup-id") != self.startup_id:
            raise GateFailure("container ownership label mismatch; refusing cleanup")
        if running:
            if not allow_stop:
                raise GateFailure("owned container is unexpectedly running")
            self.root.run(["docker", "stop", "--time", "30", self.container_id])
        remaining = self.root.run(["docker", "inspect", self.container_id], check=False)
        if remaining.returncode == 0:
            self.root.run(["docker", "rm", self.container_id])

    def _cleanup_facts(self, exit_code: int) -> dict[str, Any]:
        npu = self.root.run(["npu-smi", "info"])
        ss = self.root.run(["ss", "-H", "-lntup"])
        inspect = self.root.run(["docker", "inspect", self.container_id], check=False)
        cleanup_dir = self.repeat_dir / "cleanup"
        cleanup_dir.mkdir(parents=True, exist_ok=True)
        npu_path = cleanup_dir / "npu-smi.txt"
        ss_path = cleanup_dir / "ss-lntup.txt"
        inspect_path = cleanup_dir / "docker-inspect.json"
        npu_path.write_text(npu.stdout, encoding="utf-8")
        ss_path.write_text(ss.stdout, encoding="utf-8")
        inspect_path.write_text(
            inspect.stdout or inspect.stderr or "container absent\n", encoding="utf-8"
        )
        compute = parse_compute_pids(npu.stdout)
        pids_absent = all(not Path(f"/proc/{pid}").exists() for pid in self.host_pids)
        npu_absent = all(
            pid not in compute.get(device, [])
            for device in self.devices
            for pid in self.host_pids
        )
        return {
            "schema_version": CLEANUP_SCHEMA,
            "hostname": self.hostname,
            "startup_instance_id": self.startup_id,
            "container_id": self.container_id,
            "exit_code": exit_code,
            "host_pids": self.host_pids,
            "physical_npu_ids": self.devices,
            "service_port": self.args.service_port,
            "container_stopped_or_removed": self._container_is_stopped_or_removed(),
            "pids_absent": pids_absent,
            "port_released": not port_listener_pids(ss.stdout, self.args.service_port),
            "npu_processes_absent": npu_absent,
            "lease_released": False,
            "finished_at": utc_now(),
            "raw_evidence": {
                "npu_smi": evidence_record(self.repeat_dir, npu_path),
                "listening_sockets": evidence_record(self.repeat_dir, ss_path),
                "container_inspect": evidence_record(self.repeat_dir, inspect_path),
            },
        }

    def run(self) -> int:
        self.repeat_dir.mkdir(parents=True, exist_ok=False)
        self.lease.acquire()
        snapshots: list[Snapshot] = []
        cleanup: dict[str, Any] | None = None
        command_exit = 125
        try:
            self._create_owned_container()
            snapshots.append(self._host_snapshot(1))
            deadline = time.monotonic() + self.args.snapshot_interval_seconds
            time.sleep(max(0.0, deadline - time.monotonic()))
            snapshots.append(self._host_snapshot(2))
            self._assert_stable(*snapshots)
            atomic_json(
                self.repeat_dir / "pre-start-snapshot-summary.json",
                [snapshot.summary for snapshot in snapshots],
            )
            if self.args.dry_run:
                atomic_json(
                    self.repeat_dir / "dry-run-plan.json",
                    {
                        "schema_version": DRY_RUN_SCHEMA,
                        "startup_instance_id": self.startup_id,
                        "command": self.args.command,
                        "strict_execution_evidence_written": False,
                    },
                )
                self._cleanup_owned_container(allow_stop=False)
                return 0
            command_exit = self._run_command()
            cleanup = self._cleanup_facts(command_exit)
            if command_exit != 0:
                raise GateFailure(f"owned command exited with {command_exit}")
            if len(self.ownership) != len(self.devices) or not self.hbm_path.is_file():
                raise GateFailure(
                    "host PID/NPU ownership or HBM samples are incomplete"
                )
            immutable_path = self.repeat_dir / "immutable-input-attestation.json"
            if not immutable_path.is_file():
                raise GateFailure(
                    "owned command did not write immutable input attestation"
                )
            submission_path = self.repeat_dir / "submission" / "run_leaderboard.json"
            try:
                submission = json.loads(submission_path.read_text(encoding="utf-8"))
                metric_peak = (submission.get("metrics") or {}).get("peak_mem_mb")
            except (OSError, AttributeError, json.JSONDecodeError) as error:
                raise GateFailure(
                    "submission metrics are missing or malformed"
                ) from error
            peaks = [
                json.loads(line)["total_hbm_mb"]
                for line in self.hbm_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            peak = max(peaks)
            if metric_peak != peak:
                raise GateFailure(
                    f"submission peak_mem_mb does not match host HBM peak: {metric_peak} != {peak}"
                )
            required_cleanup = (
                "container_stopped_or_removed",
                "pids_absent",
                "port_released",
                "npu_processes_absent",
            )
            if not all(cleanup[field] is True for field in required_cleanup):
                raise GateFailure(f"owned cleanup could not be proven: {cleanup}")

            cleanup["lease_released"] = True
            released_at = self.lease.mark_released()
            cleanup_path = self.repeat_dir / "cleanup-chain-attestation.json"
            atomic_json(cleanup_path, cleanup)
            evidence = build_strict_evidence(
                hostname=self.hostname,
                startup_id=self.startup_id,
                target_id=self.args.target_id,
                side=self.args.side,
                container_id=self.container_id,
                service_port=self.args.service_port,
                runtime_image_digest=self.args.runtime_image_digest,
                immutable_inputs=evidence_record(self.repeat_dir, immutable_path),
                devices=self.devices,
                acquired_at=self.lease.acquired_at,
                released_at=released_at,
                snapshots=[snapshot.summary for snapshot in snapshots],
                ownership=self.ownership,
                hbm_samples=evidence_record(self.repeat_dir, self.hbm_path),
                peak_hbm_mb=peak,
                cleanup=evidence_record(self.repeat_dir, cleanup_path),
            )
            atomic_json(self.repeat_dir / "strict_execution_evidence.json", evidence)
            return 0
        except BaseException as error:
            try:
                self._cleanup_owned_container(allow_stop=True)
            except Exception as container_cleanup_error:
                error = GateFailure(
                    f"{error}; owned container cleanup failed: {container_cleanup_error}"
                )
            if cleanup is None and not self.args.dry_run:
                try:
                    cleanup = self._cleanup_facts(command_exit)
                except Exception as cleanup_error:
                    cleanup = {"error": str(cleanup_error)}
            atomic_json(
                self.repeat_dir / "strict_execution_failure.json",
                {
                    "schema_version": FAILURE_SCHEMA,
                    "startup_instance_id": self.startup_id,
                    "failed_at": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                    "cleanup_observation": cleanup,
                    "strict_execution_evidence_written": False,
                },
            )
            print(f"strict repeat rejected: {error}", file=sys.stderr)
            return 2
        finally:
            if self.lease.handles:
                self.lease.mark_released()
            self._make_outputs_readable()

    def _make_outputs_readable(self) -> None:
        if not self.repeat_dir.exists():
            return
        self.root.run(
            [
                "chown",
                "-R",
                f"{self.args.output_uid}:{self.args.output_gid}",
                str(self.repeat_dir),
            ]
        )
        self.root.run(["chmod", "-R", "u+rwX,go+rX", str(self.repeat_dir)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly one strict benchmark command under host evidence gates"
    )
    parser.add_argument("--repeat-dir", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--side", choices=("upstream", "vllm-hust"), required=True)
    parser.add_argument("--physical-npu", type=int, action="append", required=True)
    parser.add_argument("--service-port", type=int, required=True)
    parser.add_argument("--runtime-image-digest", required=True)
    parser.add_argument("--container-name-prefix", default="vllm-hust-strict-repeat")
    parser.add_argument(
        "--repo-host-path",
        type=Path,
        default=Path("/data/home/vllm-hust-codex-21rc/home/vllm-hust-benchmark"),
    )
    parser.add_argument(
        "--shared-data-host-path", type=Path, default=Path("/data/shared_datasets")
    )
    parser.add_argument(
        "--lease-dir", type=Path, default=Path("/var/lock/vllm-hust-benchmark")
    )
    parser.add_argument("--snapshot-interval-seconds", type=float, default=15.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-idle-hbm-mb", type=int, default=4096)
    parser.add_argument("--max-hbm-drift-mb", type=int, default=256)
    parser.add_argument(
        "--output-uid",
        type=int,
        default=int(
            os.environ.get(
                "VLLM_HUST_CALLER_UID", os.environ.get("SUDO_UID", os.getuid())
            )
        ),
    )
    parser.add_argument(
        "--output-gid",
        type=int,
        default=int(
            os.environ.get(
                "VLLM_HUST_CALLER_GID", os.environ.get("SUDO_GID", os.getgid())
            )
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("an owned command is required after --")
    if len(set(args.physical_npu)) != len(args.physical_npu) or any(
        device < 0 for device in args.physical_npu
    ):
        parser.error("physical NPU IDs must be unique non-negative integers")
    if not 0 < args.service_port < 65536:
        parser.error("service port must be between 1 and 65535")
    if not IMAGE_DIGEST_RE.fullmatch(args.runtime_image_digest):
        parser.error("runtime image digest must be sha256:<64 lowercase hex>")
    if args.snapshot_interval_seconds < 15:
        parser.error("snapshot interval must be at least 15 seconds")
    if args.sample_interval_seconds <= 0:
        parser.error("sample interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    if os.geteuid() != 0:
        caller_uid = os.getuid()
        caller_gid = os.getgid()
        command_argv = list(argv) if argv is not None else sys.argv[1:]
        if "--help" in command_argv or "-h" in command_argv:
            parse_args(command_argv)
            return 0
        probe = subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, text=True, check=False
        )
        if probe.returncode:
            print(
                f"strict repeat rejected: sudo -n is unavailable: {probe.stderr.strip()}",
                file=sys.stderr,
            )
            return 2
        environment = {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "VLLM_HUST_CALLER_UID": str(caller_uid),
            "VLLM_HUST_CALLER_GID": str(caller_gid),
            "VLLM_HUST_STRICT_ROOT_REEXEC": "1",
        }
        os.execvpe(
            "sudo",
            [
                "sudo",
                "-n",
                "env",
                *[f"{key}={value}" for key, value in environment.items()],
                sys.executable,
                __file__,
                *command_argv,
            ],
            {"PATH": environment["PATH"]},
        )
    os.umask(0o022)

    def reject_signal(signum: int, _frame: object) -> None:
        raise GateFailure(f"received signal {signum}")

    signal.signal(signal.SIGINT, reject_signal)
    signal.signal(signal.SIGTERM, reject_signal)
    try:
        args = parse_args(argv)
        return StrictRepeatOrchestrator(args).run()
    except GateFailure as error:
        print(f"strict repeat rejected: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
