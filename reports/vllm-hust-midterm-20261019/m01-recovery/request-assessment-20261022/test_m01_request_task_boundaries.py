import sqlite3

from audit_m01_request_task_boundaries import partition


def test_disjoint_boundary_partition():
    with sqlite3.connect(':memory:') as db:
        db.execute('create table TASK(startNs integer, endNs integer)')
        db.executemany('insert into TASK values (?, ?)',
                       [(0, 9), (21, 22), (10, 20), (10, 10),
                        (20, 20), (9, 11), (19, 21), (0, 30)])
        got = partition(db, 'TASK', 10, 20)
    assert got == dict(total=8, before=1, after=1, inside=3, crossing=3,
                       first_ns=0, last_ns=30)


def test_expansion_does_not_hide_outside_tasks():
    with sqlite3.connect(':memory:') as db:
        db.execute('create table TASK(startNs integer, endNs integer)')
        db.executemany('insert into TASK values (?, ?)', [(8, 9), (11, 19), (21, 23)])
        nominal = partition(db, 'TASK', 10, 20)
        expanded = partition(db, 'TASK', 8, 22)
    assert nominal['inside'] == 1
    assert expanded['inside'] == 2
    assert expanded['crossing'] == 1
