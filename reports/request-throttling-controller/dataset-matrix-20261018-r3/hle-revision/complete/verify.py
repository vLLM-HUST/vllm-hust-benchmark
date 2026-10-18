"""Verify the public Revision pair against its independently verified checkpoints."""

import hashlib
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).parent
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert Path(name).name == name
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "DATASET_PAIR_COMPLETE"
    assert report["quality_pair"]["pair_input_check"][
        "all_contract_fields_equal_except_solver_base_url"
    ]
    for arm in ["native", "controller"]:
        proof = json.loads(
            (root.parent / f"{arm}-quality-checkpoint/checkpoint.json").read_text()
        )
        for key, value in report["quality_pair"]["arms"][arm].items():
            assert proof[key] == value
        track = proof["summary"]["tracks"]["revision"]
        assert track["total"] == 1143
        assert track["graded"] + track["unresolved"] == 1143
        assert track["accuracy"] == track["successes"] / 1143
        assert all(stage["status"] == "SUCCEEDED" for stage in proof["stages"].values())
        assert proof["tasks_sha256"] == report["quality_pair"]["tasks_sha256"]
        assert (
            proof["client_code_sha256"] == report["quality_pair"]["client_code_sha256"]
        )
    native = report["quality_pair"]["arms"]["native"]["summary"]["tracks"]["revision"]
    controller = report["quality_pair"]["arms"]["controller"]["summary"]["tracks"][
        "revision"
    ]
    retained = controller["accuracy"] / native["accuracy"]
    budget = report["quality_budget"]
    assert budget["observed"]["revision"] == retained
    assert budget["passed"] == (retained >= budget["native_score_retained_minimum"])
    performance = json.loads((root.parent / "performance/checkpoint.json").read_text())
    assert report["performance"] == performance
    for modality in ["text", "image"]:
        for arm in ["native", "controller"]:
            rows = [
                row
                for row in performance["records"]
                if row["modality"] == modality
                and row["arm"] == arm
                and row["repetition"] != "warmup"
            ]
            assert len(rows) == 3
            for key, value in performance["aggregate"][modality][arm].items():
                assert statistics.median(row[key] for row in rows) == value
    print(
        json.dumps(
            {
                "complete_revision_pair": "PASS",
                "quality_budget_passed": budget["passed"],
                "private_contract_comparison": "independent private archive audit receipt only",
            }
        )
    )


if __name__ == "__main__":
    main()
