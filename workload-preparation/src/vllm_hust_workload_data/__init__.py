"""Source-backed dataset preparation only; research observation stays outside."""

from .adapters import iter_workload
from .conversations import oasst_history_prefix
from .records import WORKLOAD_SCHEMA_VERSION, WorkloadRecord
from .registry import adapter_spec, source_spec

__version__ = "0.1.0"
PROVIDER_CONTRACT_VERSION = "benchmark-workload-provider/v1"

__all__ = [
    "WORKLOAD_SCHEMA_VERSION",
    "WorkloadRecord",
    "adapter_spec",
    "iter_workload",
    "oasst_history_prefix",
    "source_spec",
]
