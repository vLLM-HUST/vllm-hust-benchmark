import importlib.util
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

PATH = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/integrate_hbm_samples.py"
)
spec = importlib.util.spec_from_file_location("hbm_samples", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partial_intervals_bound_estimate():
    result = module.integrate(
        [
            (0, 0, 99, 0, "read"),
            (0, 10, 10, 0, "read"),
            (0, 20, 20, 0, "read"),
            (0, 30, 30, 0, "read"),
        ],
        5,
        25,
    )
    assert result["fully_contained_bytes"] == 20 * 32
    assert result["intersecting_interval_bytes"] == 60 * 32
    assert result["prorated_bytes"] == (5 + 20 + 15) * 32
    assert result["minimum_channel_coverage"] == 1


def test_no_extrapolation_before_first_sample():
    result = module.integrate(
        [(0, 10, 100, 0, "write"), (0, 20, 10, 0, "write")], 0, 20
    )
    assert result["minimum_channel_coverage"] == 0.5
    assert result["prorated_bytes"] == 320


def test_reject_duplicate_timestamps():
    with pytest.raises(ValueError):
        module.integrate([(0, 10, 1, 0, "read"), (0, 10, 2, 0, "read")], 0, 20)


def test_decoder_export_crosscheck():
    samples = [(0, 0, 99, 0, "read"), (0, 1e9, 32768, 0, "read")]
    assert module.check_export(samples, [(0, 1e9, 1.0, 0, "read")]) == {
        "samples_crosschecked": 1,
        "passed": True,
    }
    with pytest.raises(ValueError, match="disagree"):
        module.check_export(samples, [(0, 1e9, 2.0, 0, "read")])


def test_decoder_rejects_empty_input():
    with pytest.raises(ValueError, match="No bandwidth"):
        module.check_export([], [])


def test_eight_read_only_channels_do_not_qualify():
    measured = dict(
        channels=[dict(channel=i, direction="read") for i in range(8)],
        minimum_channel_coverage=1,
    )
    assert not module.has_complete_hbm_coverage(measured)


def test_published_probe_archive_and_interval_bounds():
    directory = PATH.parent / "m01-recovery"
    manifest = json.loads((directory / "archive-members.json").read_text())
    with tarfile.open(directory / "counter-qualification.tar.gz") as archive:
        assert set(archive.getnames()) == set(manifest)
        for member in archive.getmembers():
            assert member.isfile()
            assert (
                hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                == manifest[member.name]
            )
    result = json.loads((directory / "hbm-sampling-integration.json").read_text())
    assert result["decoder_crosscheck"]["passed"]
    for phase in result["phases"]:
        assert phase["minimum_channel_coverage"] == 1
        assert len(phase["channels"]) == 8
        assert (
            phase["fully_contained_bytes"]
            <= phase["prorated_bytes"]
            <= phase["intersecting_interval_bytes"]
        )
        assert phase["phase"]["numerics_passed"]
