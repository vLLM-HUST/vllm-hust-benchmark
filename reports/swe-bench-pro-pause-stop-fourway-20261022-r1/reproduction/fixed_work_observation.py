"""Symmetric read-only observations for finite-work replay comparisons.

Uses the same engine/scheduler/program and exporter queries as pair-run.py.
Missing observations remain errors, never zero utilization or capacity.
"""
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode
from urllib.request import urlopen


def get(url):
    with urlopen(url, timeout=5) as response:
        raw = response.read().decode()
        return json.loads(raw) if 'json' in response.headers.get('Content-Type', '') else raw


class FixedWorkObservation:
    def __init__(self, folder, backend, endpoint, *, device=None, interval=1.0):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=False)
        self.backend, self.endpoint = backend, endpoint
        self.device, self.interval = device, interval
        self.stop = threading.Event()
        self.threads = []
        self.summary = {'scope': 'Read-only sampled observations, not atomic decisions',
                        'samples': {}, 'errors': {}}

    def loop(self, name, sources, interval):
        count = errors = 0
        try:
            with (self.folder / (name + '-private.jsonl')).open('x') as output:
                while not self.stop.is_set():
                    row = {'epoch': time.time(), 'monotonic': time.monotonic()}
                    for key, url in sources:
                        try:row[key] = get(url)
                        except Exception as error:
                            row[key + '_error'] = repr(error)
                            errors += 1
                    output.write(json.dumps(row) + '\n');output.flush()
                    count += 1
                    self.stop.wait(interval)
        except Exception as error:
            self.summary[name + '_error'] = repr(error)
            errors += 1
        finally:
            self.summary['samples'][name] = count
            self.summary['errors'][name] = errors

    def __enter__(self):
        sources = [('engine', self.backend + '/metrics'),
                   ('scheduler', self.endpoint + '/scheduler-diagnostics'),
                   ('programs', self.endpoint + '/programs')]
        groups = [('observations', sources, self.interval)]
        if self.device:
            selector = self.device['selector']
            names = ['npu_chip_info_utilization', 'npu_chip_info_hbm_bandwidth_utilization',
                     'npu_chip_info_cube_utilization', 'npu_chip_info_vector_utilization',
                     'npu_chip_info_hbm_used_memory']
            queries = [(name, name + selector) for name in names]
            queries.append(('source_timestamps', 'timestamp(npu_chip_info_utilization' + selector + ')'))
            sources = [(name, self.device['prometheus'] + '/api/v1/query?' + urlencode({'query': query}))
                       for name, query in queries]
            groups.append(('device-observations', sources, 5.0))
        for name, sources, interval in groups:
            thread = threading.Thread(target=self.loop, args=(name, sources, interval), daemon=True)
            thread.start();self.threads.append(thread)
        return self

    def __exit__(self, *exception):
        self.stop.set()
        for thread in self.threads:thread.join()
        self.summary['status'] = ('COMPLETE' if all(self.summary['samples'].values())
                                  and not any(self.summary['errors'].values()) else 'INCOMPLETE')
        (self.folder / 'observation-receipt.json').write_text(json.dumps(self.summary, indent=2) + '\n')
