"""Lease-bound native TP2 capture; no driver-side device initialization."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--backend", choices=["FFTS+", "HOST", "AIV"], required=True)
    a = p.parse_args()
    if (
        not os.environ.get("NPU_SHARE_LEASE_ID")
        or os.environ.get("NPU_SHARE_DEVICE_COUNT") != "2"
    ):
        raise RuntimeError("An atomic two-device npu-share lease is required")
    if os.environ.get("HCCL_OP_EXPANSION_MODE") != a.backend:
        raise RuntimeError("Backend environment differs from requested treatment")
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def record(kind, **fields):
        with (root / "receipt.jsonl").open("a") as f:
            f.write(
                json.dumps(dict(kind=kind, monotonic_ns=time.monotonic_ns(), **fields))
                + "\n"
            )
        print(kind, flush=True)

    record(
        "configuration",
        backend=a.backend,
        scope="Fresh native-tier pilot, not historical reproduction or a service speedup trial",
        lease={k: v for k, v in os.environ.items() if k.startswith("NPU_SHARE_")},
        script_sha256=digest(__file__),
        runtime_manifest_sha256=digest(a.runtime / "manifest.json"),
        model=str(a.model),
        model_config_sha256=digest(a.model / "config.json"),
        prompt_token_ids=[[1000] * 8, [1001] * 8],
        output_tokens=32,
    )
    import vllm
    import vllm_ascend
    from vllm import LLM, SamplingParams

    for module, source in [(vllm, "vllm-stock"), (vllm_ascend, "vllm-ascend")]:
        if (
            not Path(module.__file__)
            .resolve()
            .is_relative_to((a.runtime / source).resolve())
        ):
            raise RuntimeError(f"Unexpected module: {module.__file__}")
    record("sources", vllm=vllm.__file__, ascend=vllm_ascend.__file__)
    llm = None
    try:
        llm = LLM(
            model=str(a.model),
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            dtype="float16",
            max_model_len=128,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            gpu_memory_utilization=0.15,
            seed=17,
            async_scheduling=False,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            disable_custom_all_reduce=True,
            compilation_config={
                "cudagraph_mode": "PIECEWISE",
                "cudagraph_capture_sizes": [1],
            },
            profiler_config={
                "profiler": "torch",
                "torch_profiler_dir": str(root / "profile"),
                "torch_profiler_with_stack": False,
                "torch_profiler_with_memory": False,
            },
        )
        config = llm.llm_engine.vllm_config
        record(
            "effective_configuration",
            graph_mode=str(config.compilation_config.cudagraph_mode),
            capture_sizes=config.compilation_config.cudagraph_capture_sizes,
            splitting_ops=config.compilation_config.splitting_ops,
            parallel_config=str(config.parallel_config),
        )
        params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)

        def generate(token):
            output = llm.generate(
                [{"prompt_token_ids": [token] * 8}], params, use_tqdm=False
            )
            tokens = list(output[0].outputs[0].token_ids)
            if len(tokens) != 32:
                raise RuntimeError("Incomplete request")
            return tokens

        expected = [generate(token) for token in (1000, 1001)]
        record("warmup_complete", outputs=expected)
        llm.start_profile()
        try:
            actual = [generate(token) for token in (1000, 1001)]
        finally:
            llm.stop_profile()
        record("profile_outputs", outputs=actual, warmup_parity=actual == expected)
        if actual != expected:
            raise RuntimeError("Warmup/profile token mismatch")
        record("complete")
    except BaseException as exc:
        record("failure", error=repr(exc))
        raise
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
