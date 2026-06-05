import json
from pathlib import Path

from vllm_hust_benchmark.same_spec import build_same_spec_payload
from vllm_hust_benchmark.same_spec import runtime_model_path_has_required_artifacts
from vllm_hust_benchmark.same_spec import write_same_spec_payload


def _spec() -> dict:
    return {
        "id": "official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3",
        "label": "Official Ascend Jan 2026 baseline for vllm-hust goal tracking",
        "scenario": "random-online",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "model_parameters": "14B",
        "model_precision": "FP16",
        "hardware_vendor": "Huawei",
        "hardware_chip_model": "910B3",
        "chip_count": 1,
        "node_count": 1,
        "server_parameters": {
            "tensor_parallel_size": 1,
            "enforce_eager": "",
            "trust_remote_code": "",
            "disable_log_stats": "",
            "host": "0.0.0.0",
            "port": 8000,
        },
        "client_parameters": {
            "backend": "vllm",
            "endpoint": "/v1/completions",
            "dataset_name": "random",
            "num_prompts": 200,
            "input_len": 1024,
            "output_len": 256,
            "request_rate": 1,
            "host": "127.0.0.1",
            "port": 8000,
        },
    }


def _prefix_repetition_spec() -> dict:
    spec = _spec()
    spec["id"] = "official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3"
    spec["label"] = "Official Ascend Jan 2026 prefix repetition online baseline for vllm-hust goal tracking"
    spec["scenario"] = "prefix-repetition-online"
    spec["client_parameters"] = {
        "backend": "vllm",
        "endpoint": "/v1/completions",
        "dataset_name": "prefix_repetition",
        "num_prompts": 200,
        "input_len": 4096,
        "output_len": 256,
        "request_rate": 1,
        "host": "127.0.0.1",
        "port": 8000,
    }
    return spec


def test_build_same_spec_payload_injects_dtype() -> None:
    payload = build_same_spec_payload(_spec())

    assert payload["resolved_server_parameters"]["dtype"] == "float16"
    assert payload["resolved_client_parameters"]["random_input_len"] == 1024
    assert payload["resolved_client_parameters"]["random_output_len"] == 256
    assert "input_len" not in payload["resolved_client_parameters"]
    assert "output_len" not in payload["resolved_client_parameters"]


def test_build_same_spec_payload_maps_prefix_repetition_legacy_lengths() -> None:
    payload = build_same_spec_payload(_prefix_repetition_spec())

    assert payload["resolved_client_parameters"]["prefix_repetition_prefix_len"] == 3840
    assert payload["resolved_client_parameters"]["prefix_repetition_suffix_len"] == 256
    assert payload["resolved_client_parameters"]["prefix_repetition_num_prefixes"] == 10
    assert payload["resolved_client_parameters"]["prefix_repetition_output_len"] == 256
    assert "input_len" not in payload["resolved_client_parameters"]
    assert "output_len" not in payload["resolved_client_parameters"]


def test_same_spec_hash_ignores_runtime_model_path() -> None:
    left = build_same_spec_payload(_spec(), runtime_model="Qwen/Qwen2.5-14B-Instruct")
    right = build_same_spec_payload(
        _spec(),
        runtime_model="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/abc",
    )

    assert left["resolved_server_parameters"]["model"] != right["resolved_server_parameters"]["model"]
    assert left["resolved_spec_hash"] == right["resolved_spec_hash"]


def test_same_spec_hash_ignores_host_and_port_overrides() -> None:
    baseline = build_same_spec_payload(_spec())
    overridden = build_same_spec_payload(
        _spec(),
        server_port=8001,
        client_port=8001,
        client_host="127.0.0.1",
    )

    assert overridden["resolved_server_parameters"]["port"] == 8001
    assert overridden["resolved_client_parameters"]["port"] == 8001
    assert overridden["resolved_client_parameters"]["host"] == "127.0.0.1"
    assert baseline["resolved_spec_hash"] == overridden["resolved_spec_hash"]


def test_write_same_spec_payload(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(_spec()), encoding="utf-8")
    output_file = tmp_path / "resolved.json"

    write_same_spec_payload(
        spec_file=spec_file,
        output_file=output_file,
        runtime_model="Qwen/Qwen2.5-14B-Instruct",
    )

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["spec_id"] == _spec()["id"]


def test_runtime_model_path_has_required_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    # _path_has_complete_indexed_weights requires index file + shard files referenced in weight_map
    index_file = model_dir / "model.safetensors.index.json"
    shard_file = model_dir / "model-00001-of-00002.safetensors"
    shard_file.touch()
    index_file.write_text(
        json.dumps({"weight_map": {"model embedder": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )

    assert runtime_model_path_has_required_artifacts(model_dir)


def test_runtime_model_path_requires_tokenizer_assets(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    assert not runtime_model_path_has_required_artifacts(model_dir)


def test_runtime_model_path_requires_weight_markers(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert not runtime_model_path_has_required_artifacts(model_dir)