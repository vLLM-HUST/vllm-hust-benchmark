from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backfill_historical_pr_benchmarks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("historical_pr_backfill", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_spec(path: Path, *, scenario: str, chip_model: str = "910B2", chip_count: int = 1) -> None:
    path.write_text(
        json.dumps(
            {
                "scenario": scenario,
                "model": "Qwen/Qwen2.5-14B-Instruct",
                "model_precision": "FP16",
                "hardware_chip_model": chip_model,
                "chip_count": chip_count,
                "node_count": 1,
            }
        ),
        encoding="utf-8",
    )


def test_collect_official_specs_filters_to_910b2_single_chip(tmp_path: Path) -> None:
    module = load_module()
    write_spec(
        tmp_path / "official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json",
        scenario="sharegpt-online",
    )
    write_spec(
        tmp_path / "official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-4chip-910b2.json",
        scenario="sonnet-throughput",
        chip_count=4,
    )
    write_spec(
        tmp_path / "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b3.json",
        scenario="random-online",
        chip_model="910B3",
    )

    specs = module.collect_official_specs(tmp_path)

    assert [spec.workload for spec in specs] == ["sharegpt-online"]
    assert specs[0].chip_model == "910B2"
    assert specs[0].precision == "FP16"


def test_load_plan_defaults_plugin_ref_and_label(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"targets": [{"core_ref": "abcdef1234567890", "pr_number": 42}]}),
        encoding="utf-8",
    )

    targets = module.load_plan(plan)

    assert len(targets) == 1
    assert targets[0].label == "abcdef123456"
    assert targets[0].core_ref == "abcdef1234567890"
    assert targets[0].plugin_ref == "main"
    assert targets[0].pr_number == 42


def test_load_plan_uses_resolved_default_plugin_ref(tmp_path: Path) -> None:
    module = load_module()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"targets": [{"core_ref": "abcdef1234567890"}]}),
        encoding="utf-8",
    )

    targets = module.load_plan(plan, default_plugin_ref="51e577b17b46")

    assert targets[0].plugin_ref == "51e577b17b46"


def test_parse_npu_smi_chip_models_ignores_chip_index_rows() -> None:
    module = load_module()
    output = """
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2               | OK            | 95.2        38                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3423 / 65536         |
+===========================+===============+====================================================+
| 1     910B2               | OK            | 89.8        35                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3417 / 65536         |
"""

    assert module.parse_npu_smi_chip_models(output, "0") == ["910B2"]
    assert module.parse_npu_smi_chip_models(output, "0,1") == ["910B2", "910B2"]


def test_managed_dev_hub_defaults_use_isolated_backfill_service(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(sys, "argv", ["backfill_historical_pr_benchmarks.py"])

    args = module.parse_args()

    assert args.managed_container == "vllm-hust-backfill"
    assert args.managed_systemd_unit == "vllm-hust-backfill.service"


def test_ensure_plugin_build_info_materializes_source_worktree_file(tmp_path: Path) -> None:
    module = load_module()
    plugin_worktree = tmp_path / "plugin"
    package_dir = plugin_worktree / "vllm_ascend"
    package_dir.mkdir(parents=True)

    module.ensure_plugin_build_info(plugin_worktree, chip_model="910B2")

    assert (package_dir / "_build_info.py").read_text(encoding="utf-8") == (
        "# Auto-generated file for benchmark source worktree\n"
        "__device_type__ = 'A2'\n"
    )


def test_dev_hub_secret_env_maps_vllm_key_to_openai_key(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    dev_hub = tmp_path / "dev-hub"
    dev_hub.mkdir()
    (dev_hub / ".env").write_text(
        "VLLM_HUST_API_KEY='local-secret'\n"
        "OPENAI_API_KEY='openai-secret'\n"
        "UNRELATED=value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("VLLM_HUST_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    env = module.dev_hub_secret_env(dev_hub)

    assert env == {
        "VLLM_HUST_API_KEY": "local-secret",
        "OPENAI_API_KEY": "openai-secret",
    }


def test_dev_hub_secret_env_does_not_override_process_env(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    dev_hub = tmp_path / "dev-hub"
    dev_hub.mkdir()
    (dev_hub / ".env").write_text(
        "VLLM_HUST_API_KEY=local-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "process-openai-secret")
    monkeypatch.delenv("VLLM_HUST_API_KEY", raising=False)

    env = module.dev_hub_secret_env(dev_hub)

    assert env == {"VLLM_HUST_API_KEY": "local-secret"}
