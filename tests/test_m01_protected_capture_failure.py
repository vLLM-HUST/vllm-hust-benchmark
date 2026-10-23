import hashlib
import json
from pathlib import Path
import tarfile


BASE = Path(__file__).resolve().parents[1] / (
    "reports/vllm-hust-midterm-20261019/m01-recovery/protected-capture-20261023"
)


def test_original_failure_and_checkpoint_counts():
    path = BASE / "m01-capture-b1-r7-protected-failed-20261023.tar.gz"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == (
            "98bc3eef8d96d20ab1f2e55dfc1ee324ae04e256712f210494b8dd3a4005034a"  # pragma: allowlist secret
        )
    )
    with tarfile.open(path) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    prefix = "m01-capture-b1-r7-protected/"
    for line in files[prefix + "SHA256SUMS"].decode().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256(files[prefix + name]).hexdigest() == expected
    receipt = json.loads(files[prefix + "receipt.json"])
    assert receipt["status"] == "failed"
    assert receipt["sessions_drained"] is True
    assert receipt["all_devices_released"] is True
    assert receipt["effective_flops"] is None
    assert receipt["m01_attainment"] is None
    checkpoint = json.loads(files[prefix + "request-checkpoint.json"])
    assert len(checkpoint["requests"]) == 200
    assert sum(row["output"]["success"] for row in checkpoint["requests"]) == 48
    log = files[prefix + "service.log"].decode()
    assert "torch.OutOfMemoryError" in log
    assert "aclrtMallocHostWithCfg" in log
    assert "error code is 207001" in log
    assert not any(name.startswith(prefix + "worker-") for name in files)
