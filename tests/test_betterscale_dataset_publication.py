"""Keep published outcome projections consistent with measured aggregates."""

from pathlib import Path
import runpy


def test_betterscale_dataset_publication():
    verifier = (
        Path(__file__).resolve().parents[1]
        / "reports/betterscale-dataset-validation-20261015/verify_public.py"
    )
    runpy.run_path(str(verifier))["main"]()
