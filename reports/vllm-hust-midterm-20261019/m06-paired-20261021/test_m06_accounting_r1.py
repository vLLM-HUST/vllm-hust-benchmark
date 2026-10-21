"""Reject incomplete measurements and retain non-request lifecycle costs."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('m06_audit', Path(__file__).with_name('audit_m06_pair_r1.py'))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    receipt = {'status': 'measured', 'all_devices_released': True, 'bench_exit_code': 0,
               'allocated_cards': 2, 'reserved_cards': 4, 'simultaneous_other_services': 0,
               'started': 1000., 'ended': 1100., 'started_monotonic': 0., 'ended_monotonic': 100.,
               'bench_started_monotonic': 10., 'bench_ended_monotonic': 40.}
    bench = {'completed': 200, 'failed': 0, 'output_lens': [256]*200, 'input_lens': [100]*200,
             'ttfts': [.1]*200, 'latencies': [1.]*200, 'start_times': [11.]*200,
             'errors': ['']*200, 'total_output_tokens': 51200, 'total_input_tokens': 20000,
             'duration': 20., 'output_throughput': 2560., 'mean_tpot_ms': .9/255*1000}
    return receipt, bench


def test_full_lifecycle_not_request_duration():
    result = MODULE.metrics(*fixture())
    assert result['lifecycle_reservation_card_s_per_token'] == 400/51200
    assert result['request_reservation_card_s_per_token'] == 80/51200
    assert result['post_client_release_s'] == 60
    assert result['p99_nearest_rank'] == 198


@pytest.mark.parametrize('field,value', [('all_devices_released', False), ('ended', 1101.), ('reserved_cards', 2)])
def test_invalid_lifecycle_rejected(field, value):
    receipt, bench = fixture()
    receipt[field] = value
    with pytest.raises(AssertionError):
        MODULE.metrics(receipt, bench)


@pytest.mark.parametrize('field,value', [('failed', 1), ('total_output_tokens', 1), ('duration', 101.)])
def test_invalid_output_accounting_rejected(field, value):
    receipt, bench = fixture()
    bench = copy.deepcopy(bench)
    bench[field] = value
    with pytest.raises(AssertionError):
        MODULE.metrics(receipt, bench)
