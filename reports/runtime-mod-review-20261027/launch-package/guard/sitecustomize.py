try:
    """Experimental bootstrap, identical physical-placement guard for every arm."""
    import os

    if os.getenv("BENCH_PHYSICAL_CARDS"):
        import json
        import subprocess
        from pathlib import Path
        import torch

        original = torch.npu.set_device
        verified = set()

        def guarded_set_device(device):
            original(device)
            index = torch.npu.current_device()
            if index in verified:
                return
            # One-element allocation establishes visibility, never a benchmark workload.
            _sentinel = torch.empty(1, device=f"npu:{index}")
            physical = int(os.environ["BENCH_PHYSICAL_CARDS"].split(",")[index])
            info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
            matches = []
            for line in info.splitlines():
                cols = line.split("|")
                if len(cols) < 4:
                    continue
                numbers = cols[1].split()
                if len(numbers) == 2 and all(x.isdigit() for x in numbers):
                    pid = cols[2].strip()
                    container_pid = cols[-2].strip()
                    if str(os.getpid()) in (pid, container_pid):
                        matches.append(int(numbers[0]))
            if physical not in matches or any(c != physical for c in matches):
                raise RuntimeError(
                    f"Physical placement not verified: pid={os.getpid()} expected={physical} observed={matches}"
                )
            evidence = Path(__file__).resolve().parents[1] / "evidence"
            (
                evidence
                / f"{os.environ['BENCH_RUN_LABEL']}-placement-{os.getpid()}-{index}.json"
            ).write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "logical": index,
                        "physical": physical,
                        "smi": info,
                    },
                    indent=2,
                )
            )
            verified.add(index)

        torch.npu.set_device = guarded_set_device
        if (
            os.getenv("HUST_VSPEC_ENABLED") == "1"
            or os.getenv("BENCH_SHARED_PADDING") == "1"
        ):
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from vspec_compat import install

            install()

except BaseException:
    import os
    import traceback

    traceback.print_exc()
    os._exit(78)  # Python otherwise swallows sitecustomize failures.
