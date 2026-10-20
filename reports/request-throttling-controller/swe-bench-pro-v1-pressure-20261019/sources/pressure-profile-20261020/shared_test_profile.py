"""Materialize and check the shared graph engine contract; no device allocation."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'config/unified-graph-profile-v1.json'
SHARED_PROFILES = {'baseline': PROFILE,
                   'pressure-256-v1': ROOT / 'config/unified-graph-profile-pressure-256-v1.json'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def profile_path(shared_profile="baseline"):
    if shared_profile not in SHARED_PROFILES:
        raise ValueError("Unknown central shared profile: " + str(shared_profile))
    return SHARED_PROFILES[shared_profile]


def load(shared_profile="baseline"):
    return read(profile_path(shared_profile))


def validate_engine(engine, packages, shared_profile="baseline"):
    expected = load(shared_profile)
    errors = []
    for key, value in expected['engine'].items():
        if key not in engine or engine[key] != value or type(engine[key]) is not type(value):
            errors.append('engine.' + key + ' differs from shared contract')
    extras = set(engine) - set(expected['engine']) - {'model', 'worker_cls'}
    if extras:
        errors.append('undeclared engine arguments: ' + ', '.join(sorted(extras)))
    for key, value in expected['packages'].items():
        if packages.get(key) != value:
            errors.append('package differs: ' + key)
    if errors:
        raise ValueError('; '.join(errors))
    return {'shared_profile_id': expected['id'],
            'shared_profile_sha256': hashlib.sha256(profile_path(shared_profile).read_bytes()).hexdigest(),
            'scope': 'configuration only; loaded bytes, graph execution and quality remain to verify'}


def validate_runtime(receipt, shared_profile="baseline"):
    # The caller must collect these from effective configuration and real execution,
    # never manufacture the receipt from launch arguments.
    validate_engine(receipt['effective_engine'], receipt['packages'], shared_profile)
    if receipt.get('graph_fallback_observed') is not False:
        raise ValueError('graph fallback absent/unknown proof')
    ranks = receipt.get('workers', [])
    if len(ranks) != load(shared_profile)['engine']['tensor_parallel_size'] or {r.get('rank') for r in ranks} != {0, 1}:
        raise ValueError('missing/duplicate TP workers')
    for worker in ranks:
        for key in ('worker_epoch', 'loaded_sources_sha256', 'loaded_libraries_sha256',
                    'graph_execution_evidence', 'request_window'):
            if not worker.get(key):
                raise ValueError('missing runtime evidence: ' + key)
        if worker.get('enforce_eager') is not False or worker.get('cudagraph_mode') != 'FULL_DECODE_ONLY':
            raise ValueError('worker is eager or graph mode differs')
        if worker.get('decode_graph_replay_count', 0) <= 0:
            raise ValueError('no actual decode graph replay witnessed')
    return {'configuration_and_receipt_fields': 'pass',
            'qualification': 'evidence contents still require review; not formal cost/quality acceptance'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-profile", choices=SHARED_PROFILES, default="baseline")
    sub = parser.add_subparsers(dest='command', required=True)
    materialize = sub.add_parser('engine')
    materialize.add_argument('--model', required=True)
    check = sub.add_parser('check')
    check.add_argument('--profile', required=True, type=Path)
    runtime = sub.add_parser('runtime-check')
    runtime.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'engine':
        result = dict(load(args.shared_profile)['engine'], model=args.model)
    elif args.command == 'check':
        profile = read(args.profile)
        result = validate_engine(profile['engine'], profile.get('packages', profile.get('SDK', {})), args.shared_profile)
    else:
        result = validate_runtime(read(args.receipt), args.shared_profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
