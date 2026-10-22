"""Resolve every out-of-request task without changing the captured trace."""
import json
from pathlib import Path
import sqlite3

from audit_m01_request_task_boundaries import digest


def main():
    path = Path('/workspace/m01-request-task-boundaries.json')
    boundary = json.loads(path.read_text())
    sources = {str(path): digest(path), __file__: digest(__file__)}
    ranks = []
    for name, expected in boundary['source_sha256'].items():
        if not name.endswith('.db'):
            continue
        assert digest(name) == expected
        sources[name] = expected
        with sqlite3.connect(Path(name).as_uri()+'?mode=ro&immutable=1', uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute('''SELECT t.*, s.value AS task_type_name
                FROM TASK t LEFT JOIN STRING_IDS s ON t.taskType=s.id
                WHERE t.startNs < ? OR t.endNs > ? ORDER BY t.startNs''',
                (boundary['request_start_ns'], boundary['request_end_ns'])).fetchall()
            ranks.append([dict(row) for row in rows])
    result = dict(outside_tasks=ranks, source_sha256=sources,
                  formal_m01_attainment=None)
    with Path('/workspace/m01-request-outside-task-identities.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(ranks, indent=2))


if __name__ == '__main__':
    main()
