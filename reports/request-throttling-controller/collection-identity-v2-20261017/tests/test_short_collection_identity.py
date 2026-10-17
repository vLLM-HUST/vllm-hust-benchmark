import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('short_collector',Path(__file__).parents[1]/'scripts/collect_short_throttle_trial.py')
collector=importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

def test_reject_stale_collection_root():
    with pytest.raises(ValueError,match='Collection root'):
        collector.validate_collection_identity({'run_id':'r17','remote_root':'/workspace/r16'},'r17','r17--window')

def test_explicit_collection_correction_preserves_run_identity():
    assert collector.validate_collection_identity({'run_id':'r17','remote_root':'/workspace/r16'},'r17','r17--window','/workspace/r17')=='/workspace/r17'

def test_override_cannot_mask_wrong_job():
    with pytest.raises(ValueError,match='Run/job'):
        collector.validate_collection_identity({'run_id':'r17','remote_root':'/workspace/r17'},'r17','r16--window','/workspace/r17')
