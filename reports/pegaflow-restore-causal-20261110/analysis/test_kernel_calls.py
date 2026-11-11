"""Synthetic parser checks; these fixtures are not hardware observations."""
import tempfile
import unittest
from pathlib import Path

from analyze_kernel_calls import FUNCTIONS, loss_metadata, pair_calls, summarize

GROUP = 'pgrestore_fixture'


def event(us, index, kind, tid=8, cpu=1, pointer='0x1000'):
    return (f'kworker/u2:1-{tid} [{cpu:03d}] .... {us / 1e6:.6f}: '
            f'{GROUP}_{index}_{kind}: (fixture) ' +
            (f'arg0={pointer}' if kind == 'enter' else 'retval=0'))


def fixture(directory, trace):
    (directory / 'environment.txt').write_text('group=' + GROUP + '\n')
    (directory / 'kernel-calls.txt').write_text(trace)
    (directory / 'probe-profile.txt').write_text('\n'.join(
        f'{GROUP}/{GROUP}_{i}_{k} {trace.count(f"{GROUP}_{i}_{k}:")} 0'
        for i in range(4) for k in ['enter', 'leave']))
    for name in ['per-cpu-before', 'per-cpu-after']:
        d = directory / name
        d.mkdir()
        (d / 'cpu1.txt').write_text(
            'entries: 0\noverrun: 0\ncommit overrun: 0\ndropped events: 0\n')


class KernelCalls(unittest.TestCase):
    def test_nested_elapsed_and_migration(self):
        text = '\n'.join([event(0, 0, 'enter'), event(10, 1, 'enter'),
                          event(15, 2, 'enter'), event(20, 2, 'leave'),
                          event(30, 1, 'leave'), event(40, 0, 'leave', cpu=2)])
        calls, quality, count = pair_calls(text, GROUP)
        self.assertEqual(count, 6)
        self.assertTrue(quality['pairing_complete'])
        outer = next(r for r in calls if r['function'] == FUNCTIONS[0])
        self.assertEqual(outer['elapsed_ns'], 40000)
        self.assertEqual(outer['outside_traced_children_ns'], 20000)
        self.assertTrue(outer['migrated'])

    def test_threads_are_paired_separately(self):
        text = '\n'.join([event(0, 0, 'enter', tid=8), event(1, 0, 'enter', tid=9),
                          event(5, 0, 'leave', tid=8), event(10, 0, 'leave', tid=9)])
        calls, quality, _ = pair_calls(text, GROUP)
        self.assertTrue(quality['pairing_complete'])
        self.assertEqual(sorted(r['elapsed_ns'] for r in calls), [5000, 9000])

    def test_broken_nesting_is_discarded(self):
        text = '\n'.join([event(0, 0, 'enter'), event(1, 1, 'enter'),
                          event(10, 0, 'leave')])
        calls, quality, _ = pair_calls(text, GROUP)
        self.assertEqual(calls, [])
        self.assertEqual(quality['nesting_errors'], 1)
        self.assertFalse(quality['pairing_complete'])

    def test_capture_boundaries_are_incomplete(self):
        calls, quality, _ = pair_calls('\n'.join([
            event(1, 0, 'leave'), event(2, 0, 'enter')]), GROUP)
        self.assertEqual(calls, [])
        self.assertEqual(quality['orphan_leaves'], 1)
        self.assertEqual(quality['open_calls_at_end'], 1)
        self.assertFalse(quality['pairing_complete'])

    def test_loss_inside_pair_is_not_timed(self):
        calls, quality, _ = pair_calls('\n'.join([
            event(0, 0, 'enter'), 'CPU:1 [LOST 2 EVENTS]', event(10, 0, 'leave')]), GROUP)
        self.assertEqual(calls, [])
        self.assertEqual(quality['discarded_pairs_spanning_damage'], 1)

    def test_private_identity_and_gap(self):
        with tempfile.TemporaryDirectory() as name:
            d = Path(name)
            text = '\n'.join([event(0, 0, 'enter'), event(10, 0, 'leave'),
                              event(100, 0, 'enter'), event(105, 0, 'leave')])
            fixture(d, text)
            result = summarize(d)
            self.assertTrue(result['complete_loss_free_pairing'])
            self.assertNotIn('0x1000', str(result))
            self.assertNotIn('kworker/u2:1', str(result))
            self.assertEqual(result['per_thread_context_gaps'][0][
                'gap_between_same_context_calls']['max_ns'], 90000)

    def test_missed_returns_weaken_quality(self):
        with tempfile.TemporaryDirectory() as name:
            d = Path(name)
            fixture(d, '')
            p = d / 'probe-profile.txt'
            p.write_text(p.read_text().replace('_0_leave 0 0', '_0_leave 0 1'))
            self.assertFalse(loss_metadata(d, GROUP)['loss_free'])

    def test_profile_hits_detect_truncated_trace(self):
        with tempfile.TemporaryDirectory() as name:
            d = Path(name)
            fixture(d, '\n'.join([event(0, 0, 'enter'), event(10, 0, 'leave')]))
            p = d / 'probe-profile.txt'
            p.write_text(p.read_text().replace('_0_enter 1 0', '_0_enter 2 0'))
            result = summarize(d)
            self.assertFalse(result['complete_loss_free_pairing'])
            self.assertEqual(len(result['loss_checks']['event_hit_count_mismatches']), 1)

    def test_other_channel_does_not_determine_target_timing(self):
        with tempfile.TemporaryDirectory() as name:
            d = Path(name)
            rows = []
            for start, duration, channel in [(0, 10, 44), (100, 2000, 77)]:
                rows += [event(start, 0, 'enter'),
                         event(start + 1, 1, 'enter') + f' chan_id={channel}',
                         event(start + duration - 1, 1, 'leave'),
                         event(start + duration, 0, 'leave')]
            fixture(d, '\n'.join(rows))
            result = summarize(d, target_channel=44)
            self.assertEqual(result['paired_call_count'], 2)
            self.assertEqual(result['all_channels_paired_call_count'], 4)
            self.assertEqual(result['functions'][FUNCTIONS[0]]['elapsed']['max_ns'], 10000)
            self.assertNotIn(77, [c for r in result['largest_elapsed_calls']
                                  for c in r['channels']])

    def test_shared_outer_call_is_not_attributed_to_one_channel(self):
        with tempfile.TemporaryDirectory() as name:
            d = Path(name)
            text = '\n'.join([
                event(0, 0, 'enter'), event(1, 1, 'enter') + ' chan_id=44',
                event(5, 1, 'leave'), event(10, 1, 'enter') + ' chan_id=77',
                event(1000, 1, 'leave'), event(1100, 0, 'leave')])
            fixture(d, text)
            result = summarize(d, target_channel=44)
            self.assertEqual(result['paired_call_count'], 1)
            self.assertEqual(result['ambiguous_multichannel_calls_excluded'], 1)
            self.assertEqual(result['functions'][FUNCTIONS[1]]['elapsed']['max_ns'], 4000)

    def test_overrun_or_unread_tail_weaken_quality(self):
        with tempfile.TemporaryDirectory() as name:
            d = Path(name)
            fixture(d, '')
            p = d / 'per-cpu-after/cpu1.txt'
            p.write_text(p.read_text().replace('overrun: 0', 'overrun: 1', 1))
            self.assertFalse(loss_metadata(d, GROUP)['loss_free'])
            p.write_text('entries: 1\noverrun: 0\ncommit overrun: 0\ndropped events: 0\n')
            self.assertFalse(loss_metadata(d, GROUP)['loss_free'])


if __name__ == '__main__':
    unittest.main()
