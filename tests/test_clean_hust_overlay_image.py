from pathlib import Path


def test_clean_overlay_removes_both_source_trees_before_copy() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker" / "Dockerfile.vllm-hust-clean-overlay").read_text(
        encoding="utf-8"
    )

    remove = dockerfile.index(
        "rm -rf /vllm-workspace/vllm/vllm /vllm-workspace/vllm-ascend/vllm_ascend"
    )
    core_copy = dockerfile.index("COPY core/vllm /vllm-workspace/vllm/vllm")
    plugin_copy = dockerfile.index(
        "COPY plugin/vllm_ascend /vllm-workspace/vllm-ascend/vllm_ascend"
    )

    assert remove < core_copy < plugin_copy
    assert "LABEL org.opencontainers.image.vllm-core-commit=" in dockerfile
    assert "LABEL org.opencontainers.image.vllm-ascend-commit=" in dockerfile
    assert 'test -n "${VLLM_CORE_COMMIT}"' in dockerfile
    assert 'test -n "${VLLM_ASCEND_COMMIT}"' in dockerfile
