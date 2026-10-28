"""Score private native FLEURS/Whisper outputs without another model call.

This explicitly declared CER integration profile is not the FLEURS Conformer
paper baseline. Do not mix incomplete locale coverage into a full-suite score.
"""

import argparse
from importlib import metadata
import json
from pathlib import Path

from prepare_xsum_original_assets import sha


def collect_cases(root, receipt):
    if (
        receipt["source_id"] != "FLEURS"
        or receipt["status"] != "NATIVE_EXECUTION_COMPLETE_SCORE_PENDING"
        or receipt["failures"]
    ):
        raise ValueError(
            "complete native execution receipt required; failures stay visible"
        )
    cases = {}
    seen = set()
    for case in receipt["cases"]:
        key = (case["locale"], case["position"])
        if key in seen or Path(case["file"]).name != case["file"]:
            raise ValueError("duplicate case or uncontained private output")
        seen.add(key)
        path = root / case["file"]
        if sha(path) != case["sha256"]:
            raise ValueError("native private result identity differs")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload["locale"], payload["position"]) != key:
            raise ValueError("original task identity differs")
        reference, prediction = (
            payload["original"]["transcription"],
            payload["result"]["text"],
        )
        if not isinstance(reference, str) or not isinstance(prediction, str):
            raise ValueError("original reference and native text must remain strings")
        cases.setdefault(case["locale"], []).append((reference, prediction))
    if set(cases) != set(receipt["selected_locales"]):
        raise ValueError("selected locale missing; do not silently drop languages")
    expected = receipt["expected_cases_per_locale"]
    if set(expected) != set(cases):
        raise ValueError("expected locale coverage differs")
    for locale, count in expected.items():
        if count <= 0 or (not receipt["full"] and count != 3):
            raise ValueError("fixed three original cases per selected locale required")
        if {position for name, position in seen if name == locale} != set(range(count)):
            raise ValueError("original contiguous sample coverage incomplete")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh score output required")
    if metadata.version("jiwer") != "4.0.0":
        raise ValueError("pinned jiwer==4.0.0 required")
    import jiwer
    from whisper.normalizers import BasicTextNormalizer
    from run_fleurs_original import PROFILE, verify_framework

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    verify_framework(profile)
    receipt_path = args.run / "receipt-private.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["profile_sha256"] != sha(PROFILE):
        raise ValueError("scoring profile differs from native execution")
    cases = collect_cases(args.run, receipt)
    normalizer = BasicTextNormalizer()
    scores = {}
    for locale, rows in sorted(cases.items()):
        refs, hyps = map(list, zip(*rows))
        scores[locale] = {
            "cases": len(rows),
            "raw_cer": jiwer.cer(refs, hyps),
            "whisper_basic_normalized_cer": jiwer.cer(
                [normalizer(s) for s in refs], [normalizer(s) for s in hyps]
            ),
        }
    result = {
        "source_id": "FLEURS",
        "status": "DECLARED_NATIVE_CER_PROFILE_SCORED",
        "native_receipt_sha256": sha(receipt_path),
        "profile_sha256": sha(PROFILE),
        "locales": scores,
        "completed_locales": len(scores),
        "all_102_locales": len(scores) == 102,
        "all_original_test_rows_claim": bool(receipt["full"] and len(scores) == 102),
        "macro_raw_cer": sum(s["raw_cer"] for s in scores.values()) / len(scores),
        "macro_whisper_basic_normalized_cer": sum(
            s["whisper_basic_normalized_cer"] for s in scores.values()
        )
        / len(scores),
        "metric_boundary": profile["metric_boundary"],
        "source_checkbox_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_locales": len(scores),
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
