"""A dead leader must not leave late-writing children in a sealed run."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from run_native import cleanup_group


class CleanupTests(unittest.TestCase):
    def test_recorded_leader_can_exec_with_a_clean_environment(self):
        from run_native import process_state

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            script = 'import os,sys; os.execve(sys.executable,[sys.executable,"-c","import time; time.sleep(10)"],{})'
            process = subprocess.Popen(
                [sys.executable, "-c", script],
                start_new_session=True,
                env=dict(os.environ, VLLM_HUST_TEST_OUTPUT=folder),
            )
            identity = process_state(process.pid)
            deadline = time.monotonic() + 3
            while (
                b"VLLM_HUST_TEST_OUTPUT="
                in Path(f"/proc/{process.pid}/environ").read_bytes()
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            try:
                actions = cleanup_group(identity, out, grace_s=1)
                self.assertTrue(actions)
                process.wait(timeout=3)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_orphan_child_stops_before_sealing(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            script = """import json, os, time
from pathlib import Path
p = Path(os.environ['VLLM_HUST_TEST_OUTPUT'])
fields = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
(p/'identity.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':fields[19]}))
if os.fork():
    os._exit(0)
time.sleep(2)
(p/'late-write').write_text('must not appear')
"""
            process = subprocess.Popen(
                [sys.executable, "-c", script],
                start_new_session=True,
                env=dict(os.environ, VLLM_HUST_TEST_OUTPUT=folder),
            )
            process.wait(timeout=5)
            actions = cleanup_group(
                json.loads((out / "identity.json").read_text()), out, grace_s=1
            )
            self.assertTrue(actions)
            time.sleep(2.1)
            self.assertFalse((out / "late-write").exists())


if __name__ == "__main__":
    unittest.main()
