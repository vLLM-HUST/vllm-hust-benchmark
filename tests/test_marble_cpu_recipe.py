"""Recipe boundaries only, not image-build or task execution evidence."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_recipe_defaults_to_nonroot_offline_help_not_task_execution():
    recipe = (ROOT / "configs/workload-native/marble-compat-cpu.Dockerfile").read_text(
        "utf-8"
    )
    assert "ARG MARBLE_BASE_IMAGE" in recipe and "FROM ${MARBLE_BASE_IMAGE}" in recipe
    assert "USER 65534:65534" in recipe
    assert 'CMD ["--help"]' in recipe
    assert "HF_HUB_OFFLINE=1" in recipe and "TRANSFORMERS_OFFLINE=1" in recipe
    copies = [line for line in recipe.splitlines() if line.startswith("COPY ")]
    assert copies == ["COPY client/ /opt/marble-client/", "COPY author/ /author/"]
    assert "--config_path" not in recipe
