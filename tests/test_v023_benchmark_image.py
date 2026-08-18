from pathlib import Path


def test_v023_benchmark_image_only_adds_frozen_benchmark_clients() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker" / "Dockerfile.vllm-ascend-v023-bench").read_text(
        encoding="utf-8"
    )

    assert "quay.io/ascend/vllm-ascend:v0.23.0-openeuler" in dockerfile
    assert '"datasets==3.3.0"' in dockerfile
    assert '"xxhash==3.6.0"' in dockerfile
    assert "ARG VLLM_CORE_COMMIT" in dockerfile
    assert "ARG VLLM_ASCEND_COMMIT" in dockerfile
    assert "org.opencontainers.image.vllm-core-commit" in dockerfile
    assert "org.opencontainers.image.vllm-ascend-commit" in dockerfile
    assert 'test -n "${VLLM_CORE_COMMIT}"' in dockerfile
    assert 'test -n "${VLLM_ASCEND_COMMIT}"' in dockerfile
    assert "pip install --no-cache-dir" in dockerfile
    assert "vllm-ascend" not in dockerfile.split("RUN", maxsplit=1)[1]
    assert "torch-npu" not in dockerfile.split("RUN", maxsplit=1)[1]


def test_v018_runtime_is_rebuilt_on_the_official_v023_container() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker" / "Dockerfile.vllm-v018-on-ascend-v023").read_text(
        encoding="utf-8"
    )

    assert "FROM quay.io/ascend/vllm-ascend:v0.23.0-openeuler" in dockerfile
    assert "COPY --from=v018-python /usr/local/python3.11.14" in dockerfile
    assert "pip install --no-build-isolation --no-deps /opt/vllm-v018" in dockerfile
    assert (
        "pip install --no-build-isolation --no-deps /opt/vllm-ascend-v018" in dockerfile
    )
    assert "RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE" in dockerfile
    assert "LD_PRELOAD=" in dockerfile
    assert "libcust_opapi.so" in dockerfile
