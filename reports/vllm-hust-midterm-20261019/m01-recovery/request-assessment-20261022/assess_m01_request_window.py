"""Derive request-window empirical attainment without rewriting earlier evidence."""
import json
from pathlib import Path
import sqlite3
from statistics import mean
import sys

sys.path.insert(0, '/root/gennova-evidence/20261019-midterm-aligned')
from integrate_hbm_samples import integrate, has_complete_hbm_coverage
from audit_m01_request_task_boundaries import digest


def assess(data, hbm):
    boundary, identities, composition, accounting, window_review, reference = data
    allowed = {'PROFILING_ENABLE', 'PROFILING_DISABLE', 'EVENT_RECORD'}
    outside = [task for rank in identities['outside_tasks'] for task in rank]
    assert len(outside) == 7
    assert all(task['task_type_name'] in allowed for task in outside)
    assert all(c['boundary_or_tail_numeric_values_all_zero'] for c in window_review['checks'])
    for rank in boundary['ranks']:
        assert rank['tables']['TASK']['nominal']['crossing'] == 0
        comm = rank['tables']['COMMUNICATION_OP']['contracted_2ms']
        assert comm['total'] == comm['inside'] == 83096
    direction_terms = {'lower': [], 'upper': []}
    for repetition in range(3):
        for bound, key in [('lower', 'fully_contained_bytes'),
                           ('upper', 'intersecting_interval_bytes')]:
            terms = []
            for rank, measured in enumerate(hbm):
                ref = reference['ranks'][rank]['phases'][repetition]
                duration = ref['phase']['end_wall'] - ref['phase']['start_wall']
                for direction in ('read', 'write'):
                    volume = sum(c[key] for c in measured['channels'] if c['direction'] == direction)
                    rate = sum(c['fully_contained_bytes'] for c in ref['physical']['channels']
                               if c['direction'] == direction) / duration
                    terms.append(volume / rate)
            direction_terms[bound].append(max(terms))
    compute = composition['resource_mean_seconds']['compute']
    comm = boundary['communication_reference_terms']
    resources = {bound: dict(compute=compute,
                            communication=mean(t[f'{bound}_seconds'] for t in comm),
                            hbm_directional=mean(direction_terms[bound]))
                 for bound in ('lower', 'upper')}
    duration = boundary['request_phase']['duration_seconds']
    percentages = {bound: 100 * max(terms.values()) / duration
                   for bound, terms in resources.items()}
    return dict(status='instrumented_request_window_empirical_assessment',
                request_seconds=duration, basic_operations=accounting['basic_operations'],
                basic_operations_per_second=accounting['basic_operations']/duration,
                resource_mean_seconds=resources, empirical_attainment_percent=percentages,
                hbm_directional_repetitions_seconds=direction_terms,
                threshold_percent=50, threshold_met=percentages['lower'] >= 50,
                verdict=('below_internal_threshold' if percentages['upper'] < 50
                         else 'above_internal_threshold' if percentages['lower'] >= 50
                         else 'boundary_indeterminate'),
                uninstrumented_attainment=None, formal_b1_admission=False,
                limitations=accounting['limitations'][2:] + [
                    'Trace instrumentation remains present; request-window accounting excludes client startup before the first request, but is not an uninstrumented score.',
                    'Request interval replaces the device envelope only in this new derivation.',
                    'HBM lower bounds contract and upper bounds expand the request window by 2ms on each side; communication uses its separately retained sampling bounds.',
                    'Bounds describe sampling boundaries, not confidence intervals or reference repeat uncertainty.',
                    'Observed TASK capture coverage does not prove profiler implementation completeness.'])


def main():
    paths = [Path('/workspace') / name for name in (
        'm01-request-task-boundaries.json', 'm01-request-outside-task-identities.json',
        'm01-worker-composition-review.json', 'm01-worker-final-accounting.json',
        'm01-worker-window-review.json', 'm01-hbm-reference-worker-r1/audit.json')]
    data = [json.loads(p.read_text()) for p in paths]
    hardware_path = Path('/workspace/m01-worker-r1-envelope-hardware.json')
    hardware = json.loads(hardware_path.read_text())
    window = next(w for w in hardware['windows'] if w['name'] == 'first_request_to_last_response')
    hbm = []
    sources = {str(p): digest(p) for p in paths + [hardware_path, Path(__file__)]}
    for device in window['hbm_devices']:
        path = Path(device['db'])
        assert digest(path) == device['sha256']
        sources[str(path)] = device['sha256']
        with sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True) as db:
            samples = db.execute('SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0').fetchall()
        # SQLite stores these integer-valued nanoseconds as REAL. Integer
        # arithmetic avoids losing 256ns when subtracting mixed float/int bounds.
        assert all(timestamp == int(timestamp) for _, timestamp, _, _, _ in samples)
        samples = [(device, int(timestamp), count, channel, direction)
                   for device, timestamp, count, channel, direction in samples]
        measured = integrate(samples, data[0]['request_start_ns']-2000000,
                             data[0]['request_end_ns']+2000000)
        assert has_complete_hbm_coverage(measured)
        contracted = integrate(samples, data[0]['request_start_ns']+2000000,
                               data[0]['request_end_ns']-2000000)
        assert has_complete_hbm_coverage(contracted)
        for outer, inner in zip(measured['channels'], contracted['channels']):
            assert (outer['channel'], outer['direction']) == (inner['channel'], inner['direction'])
            outer['fully_contained_bytes'] = inner['fully_contained_bytes']
        measured['fully_contained_bytes'] = contracted['fully_contained_bytes']
        measured['bounds_scope'] = 'contracted-contained lower; expanded-intersecting upper'
        hbm.append(measured)
    result = assess(data, hbm)
    result.update(hbm_expanded_window=hbm, source_sha256=sources)
    with Path('/workspace/m01-request-window-assessment.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('source_sha256', 'hbm_expanded_window', 'limitations')}, indent=2))


if __name__ == '__main__':
    main()
