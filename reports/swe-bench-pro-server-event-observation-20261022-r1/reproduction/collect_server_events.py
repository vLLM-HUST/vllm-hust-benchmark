"""Read-only, bounded event collection. Never submit inference or retry a job.

Every received batch is flushed/fsynced before validation, retaining failures.
A valid prefix is evidence only through its last sequence; process death can
lose uncollected tail events. Collector RTT bounds epoch offset, not monotonic
clock equivalence. Buffer loss, service restart and source drift disqualify.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time

import httpx


def validate_batch(batch, *, service_id, source_sha256, instance_id, after_sequence, previous_clock=None):
    summary = batch['summary']
    if summary['service_id'] != service_id or summary['observer_source_sha256'] != source_sha256:
        raise ValueError('Observer identity/source mismatch')
    if not isinstance(summary.get('instance_id'), str) or not summary['instance_id']:
        raise ValueError('Missing observer process identity')
    if instance_id is not None and summary['instance_id'] != instance_id:
        raise ValueError('Observer restarted')
    if summary['dropped'] != 0 or summary['snapshot_errors'] != 0:
        raise ValueError('Observer lost events or state')
    seen = summary['total_seen']
    if type(seen) is not int or seen < after_sequence:
        raise ValueError('Event sequence regressed')
    rows = batch['events']
    if len(rows) != seen - after_sequence:
        raise ValueError('Event sequence gap or duplicate')
    previous_time = previous_clock
    for expected, row in enumerate(rows, after_sequence + 1):
        if row['sequence'] != expected or row['service_id'] != service_id:
            raise ValueError('Event sequence/identity mismatch')
        stamp = row['monotonic_ns']
        if type(stamp) is not int or (previous_time is not None and stamp < previous_time):
            raise ValueError('Event clock regressed')
        previous_time = stamp
    clock = batch['clock']
    if not math.isfinite(clock['epoch']) or type(clock['monotonic_ns']) is not int:
        raise ValueError('Invalid observer clock')
    if previous_time is not None and clock['monotonic_ns'] < previous_time:
        raise ValueError('Observer clock predates events')
    return summary['instance_id'], seen


async def collect(client, endpoint, output, *, service_id, source_sha256,
                  duration_seconds, interval_seconds=1, max_polls=None):
    if (not math.isfinite(duration_seconds) or duration_seconds <= 0
            or not math.isfinite(interval_seconds) or interval_seconds < 0
            or (max_polls is not None and (type(max_polls) is not int or max_polls <= 0))):
        raise ValueError('Require a finite collection duration and valid polling bounds')
    cursor, instance_id, polls = 0, None, 0
    previous_clock = None
    start = time.monotonic()
    def persist(stream, value):
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    with Path(output).open('x') as stream:
        try:
            while True:
                sent_epoch, sent_monotonic = time.time(), time.monotonic()
                response = await client.get(endpoint.rstrip('/') + '/experiment-server-events',
                                            params={'after_sequence': cursor})
                received_epoch, received_monotonic = time.time(), time.monotonic()
                response.raise_for_status()
                batch = response.json()
                persist(stream, {'poll': polls + 1, 'requested_after_sequence': cursor,
                    'sent_epoch': sent_epoch, 'received_epoch': received_epoch,
                    'request_rtt_seconds': received_monotonic - sent_monotonic,
                    'batch': batch})
                new_instance, new_cursor = validate_batch(batch, service_id=service_id,
                    source_sha256=source_sha256, instance_id=instance_id, after_sequence=cursor,
                    previous_clock=previous_clock)
                if received_epoch < sent_epoch:
                    raise ValueError('Collector wall clock regressed')
                offset = [batch['clock']['epoch'] - received_epoch,
                          batch['clock']['epoch'] - sent_epoch]
                persist(stream, {'validated_poll': polls + 1,
                    'observer_epoch_minus_collector_epoch_bounds_seconds': offset,
                    'instance_id': new_instance, 'validated_through_sequence': new_cursor})
                instance_id, cursor, polls = new_instance, new_cursor, polls + 1
                previous_clock = batch['clock']['monotonic_ns']
                if ((max_polls is not None and polls >= max_polls)
                        or time.monotonic() - start >= duration_seconds):
                    break
                await asyncio.sleep(min(interval_seconds, max(0, duration_seconds - (time.monotonic() - start))))
        except BaseException as error:
            try:
                persist(stream, {'collector_error_type': type(error).__name__,
                                 'validated_through_sequence': cursor, 'completed_polls': polls})
            except (OSError, ValueError):
                pass  # Preserve the original failure if its report cannot be written.
            raise
        result = {'service_id': service_id, 'observer_source_sha256': source_sha256,
                  'instance_id': instance_id, 'validated_through_sequence': cursor,
                  'completed_polls': polls, 'coverage': 'Observed prefix only, not uncollected server tail'}
        persist(stream, {'collector_summary': result})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--service-id', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-seconds', type=float, required=True)
    parser.add_argument('--interval-seconds', type=float, default=1)
    args = parser.parse_args()
    async def run():
        async with httpx.AsyncClient(timeout=10, follow_redirects=False, trust_env=False) as client:
            return await collect(client, args.endpoint, args.output, service_id=args.service_id,
                source_sha256=args.source_sha256, duration_seconds=args.duration_seconds,
                interval_seconds=args.interval_seconds)
    result = asyncio.run(run())
    result['file_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
