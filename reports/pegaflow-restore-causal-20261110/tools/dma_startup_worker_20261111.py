"""Keep model startup's unused pinned host buffers out of the serving budget."""

import json
import os
import time
from pathlib import Path

import torch_npu
from vllm_ascend.worker.worker import NPUWorker


class DMAStartupWorker(NPUWorker):
    def _trim_unused_host_cache(self, phase):
        torch_npu.npu.synchronize()
        before = torch_npu._C._npu_hostMemoryStats()
        before_memory = Path("/sys/fs/cgroup/memory.current").read_text().strip()
        torch_npu._C._npu_hostEmptyCache()
        after = torch_npu._C._npu_hostMemoryStats()
        row = dict(
            event="startup_host_cache_trim",
            pid=os.getpid(),
            epoch=time.time(),
            phase=phase,
            before=before,
            after=after,
            cgroup_before_bytes=before_memory,
            cgroup_after_bytes=Path("/sys/fs/cgroup/memory.current")
            .read_text()
            .strip(),
        )
        directory = Path(os.environ["PEGAFLOW_TELEMETRY_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "startup-host-cache.jsonl").open("a") as out:
            out.write(json.dumps(row, default=str) + "\n")

    def load_model(self):
        super().load_model()
        self._trim_unused_host_cache("after_load_model")

    def compile_or_warm_up_model(self):
        self._trim_unused_host_cache("before_graph")
        result = super().compile_or_warm_up_model()
        self._trim_unused_host_cache("after_graph")
        return result
