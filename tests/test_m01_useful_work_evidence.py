import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/vllm-hust-midterm-20261019/m01-recovery/useful-work-20261023"


def test_m01_candidates_remain_unadmitted_and_sources_preserved():
    archive = EVIDENCE / "derivation-sources.tar.gz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == (
            "955196081af1a0ad14914523712cc774fea5065c44d52cb9211d25423956be11"  # pragma: allowlist secret
        )
    )
    with tarfile.open(archive) as bundle:
        source_hashes = {}
        for member in bundle.getmembers():
            assert member.isfile() and Path(member.name).name == member.name
            source_hashes[member.name] = hashlib.sha256(
                bundle.extractfile(member).read()
            ).hexdigest()
    checked = set()
    for path in EVIDENCE.glob("*.json"):
        record = json.loads(path.read_text())
        assert record["effective_flops"] is None
        assert record["efficiency_percent"] is None
        for source, digest in record["source_sha256"].items():
            name = Path(source).name
            if name in source_hashes:
                assert digest == source_hashes[name]
                checked.add(name)
    assert len(checked) == 6


def test_m01_partial_target_sum_and_clipping_bounds():
    def read(name):
        return json.loads((EVIDENCE / name).read_text())

    gdn = read("target-gdn-candidate-r1.json")
    parent = read("target-vocab-conv-candidate-r1.json")
    assert gdn["target_family_candidate_subtotal"] == (
        parent["combined_target_partial_subtotal"]
        + gdn["prefill_basic_flops"]
        + gdn["recurrent_basic_flops"]
        + gdn["preprocess_basic_flops"]
    )
    assert gdn["necessary_decode_rows"] + gdn["excluded_active_verify_rows"] == 57303
    bounds = read("terminal-draft-acceptance-bounds-r1.json")
    assert bounds["useful_accepted_draft_tokens_lower"] == 32021
    assert bounds["useful_accepted_draft_tokens_upper"] == 32056
    assert "mislabeled" in bounds["supersedes_interpretation"]
