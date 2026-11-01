"""Environment policy fixtures; no original task, renderer, Docker or model."""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "typst_environment", ROOT / "scripts/verify_structeval_typst_environment.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_pdf_read_enablement_does_not_relax_ps_or_resource_rules(tmp_path):
    policy = tmp_path / "policy.xml"
    policy.write_text(
        '<policymap><policy domain="coder" pattern="PDF" rights="none"/>'
        '<policy domain="coder" pattern="PS" rights="none"/>'
        '<policy domain="resource" name="memory" value="256MiB"/></policymap>'
    )
    module().allow_pdf_read(policy)
    actual = ET.parse(policy)
    assert actual.find(".//policy[@pattern='PDF']").get("rights") == "read"
    assert actual.find(".//policy[@pattern='PS']").get("rights") == "none"
    assert actual.find(".//policy[@domain='resource']").get("value") == "256MiB"


@pytest.mark.parametrize(
    "content",
    [
        "<policymap/>",
        '<policymap><policy domain="coder" pattern="PDF"/><policy domain="coder" pattern="PDF"/></policymap>',
    ],
)
def test_unknown_policy_is_retained_not_silently_replaced(tmp_path, content):
    policy = tmp_path / "policy.xml"
    policy.write_text(content)
    original = policy.read_bytes()
    with pytest.raises(ValueError, match="inspect environment"):
        module().allow_pdf_read(policy)
    assert policy.read_bytes() == original


def test_recipe_declares_observed_environment_not_author_lock_or_task_execution():
    text = (
        ROOT / "configs/workload-native/structeval-typst-cpu.Dockerfile"
    ).read_text()
    assert "ARG RENDER_BASE" in text
    assert "COPY typst /usr/local/bin/typst" in text
    assert "not an original author lock" in text
    assert text.rfind("USER 65534:65534") > text.rfind("RUN ")
    assert "run_structeval_original_json" not in text


def test_distribution_nested_comments_preserved_byte_for_byte(tmp_path):
    original = (
        '<policymap>\r\n<!-- <policy domain="cache" name="shared-secret"/>\r\n'
        "<!-- distribution nested comment ends here -->\r\n"
        '<policy domain="coder" rights="none" pattern="PDF"/>\r\n</policymap>'
    ).encode()
    policy = tmp_path / "policy.xml"
    policy.write_bytes(original)
    module().allow_pdf_read(policy)
    assert policy.read_bytes() == original.replace(b'rights="none"', b'rights="read"')
