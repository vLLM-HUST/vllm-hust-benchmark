"""Task-only Python bootstrap, also inherited by original Agent subprocesses.

Mount this directory read-only and use it as PYTHONPATH in the dedicated
container. Base site-packages keep precedence over supplemental packages.
"""

import os
from pathlib import Path
import sys


for variable in [
    "MLAGENTBENCH_COMPATIBILITY_DIR",
    "MLAGENTBENCH_SUPPLEMENTAL_DIR",
]:
    value = os.environ.get(variable)
    if value:
        path = Path(value).resolve(strict=True)
        if not path.is_dir():
            raise ValueError(f"{variable} must point to a directory")
        if str(path) not in sys.path:
            sys.path.append(str(path))
