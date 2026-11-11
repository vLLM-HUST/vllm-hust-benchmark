"""Real TP2 graph-mode diagnostic; never a replacement for the frozen B0/B1 tests."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
import time
import traceback
import uuid
from measurements import request_metrics


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def main():
    out = Path(os.environ["VLLM_HUST_TEST_OUTPUT"])
    write(
        out / "owned-process.json",
        {
            "pid": os.getpid(),
            "start_ticks": Path("/proc/self/stat").read_text().split()[21],
        },
    )
    config = json.loads(Path(os.environ["VLLM_HUST_TEST_CONFIG"]).read_text())
    workload = json.loads(Path(config["workload_file"]).read_text())
    records = []
    llm = None
    started = time.monotonic()
    write(
        out / "started.json",
        {"pid": os.getpid(), "time_epoch": time.time(), "config": config},
    )
    try:
        import torch
        import torch_npu  # noqa: F401 -- registers the NPU backend

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        from vllm import LLM, SamplingParams
        import vllm
        import vllm_ascend

        write(
            out / "runtime.json",
            {
                "vllm": importlib.metadata.version("vllm"),
                "vllm_ascend": importlib.metadata.version("vllm-ascend"),
                "torch": torch.__version__,
                "torch_npu": importlib.metadata.version("torch-npu"),
                "python": sys.version,
                "triton_ascend": importlib.metadata.version("triton-ascend"),
                "core_file": vllm.__file__,
                "plugin_file": vllm_ascend.__file__,
                "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                "role": config.get(
                    "role",
                    "installed 0.23 diagnostic; neither required B0 0.18 nor qualified B1",
                ),
            },
        )
        if config.get("arm") == "B0" and any(
            importlib.metadata.version(name) != "0.18.0"
            for name in ["vllm", "vllm-ascend"]
        ):
            raise ValueError("B0 must actually load vLLM/vLLM-Ascend 0.18.0")
        if config.get("arm") == "B1":
            expected = config["version_receipt"]["arms"]["B1"]["observed"]
            for name in ["vllm", "vllm-ascend", "torch", "torch-npu"]:
                if importlib.metadata.version(name) != expected["packages"][name]:
                    raise ValueError(
                        "Runtime package changed after version admission: " + name
                    )
            for name, module in [("vllm", vllm), ("vllm_ascend", vllm_ascend)]:
                if (
                    Path(module.__file__).resolve()
                    != Path(expected["origins"][name]).resolve()
                ):
                    raise ValueError(
                        "Import origin changed after version admission: " + name
                    )
        llm = LLM(**config["kwargs"])
        write(
            out / "effective-config.json",
            {
                "compilation": str(llm.llm_engine.vllm_config.compilation_config),
                "cache": str(llm.llm_engine.vllm_config.cache_config),
                "parallel": str(llm.llm_engine.vllm_config.parallel_config),
                "load_elapsed_s": time.monotonic() - started,
            },
        )
        # Task text is not used for warmup. No answers or KV are carried between requests.
        warm = llm.generate(
            [{"prompt_token_ids": workload["warmup_tokens"]}],
            SamplingParams(temperature=0, max_tokens=8, seed=0),
            use_tqdm=False,
        )
        write(
            out / "warmup.json",
            {"outputs": len(warm), "output_tokens": len(warm[0].outputs[0].token_ids)},
        )
        with (out / "requests.jsonl").open("w") as sink:
            for task in workload["requests"]:
                row = {
                    k: task[k] for k in ("id", "group", "input_tokens", "max_tokens")
                }
                before = time.monotonic()
                row["sent_s"] = before
                try:
                    response = llm.generate(
                        [{"prompt_token_ids": task["prompt_token_ids"]}],
                        SamplingParams(
                            temperature=0,
                            seed=0,
                            max_tokens=task["max_tokens"],
                            ignore_eos=task["ignore_eos"],
                        ),
                        use_tqdm=False,
                    )[0]
                    end = time.monotonic()
                    output = response.outputs[0]
                    metrics = vars(response.metrics) if response.metrics else {}
                    timings = request_metrics(
                        metrics, before, end, len(output.token_ids)
                    )
                    row.update(
                        status="success",
                        end_s=end,
                        elapsed_s=end - before,
                        output_tokens=len(output.token_ids),
                        output_token_ids=list(output.token_ids),
                        output_text=output.text,
                        finish_reason=output.finish_reason,
                        stop_reason=output.stop_reason,
                        engine_metrics=metrics,
                        **timings,
                        expected_answer=task.get("expected_answer"),
                        prompt_sha256=hashlib.sha256(
                            json.dumps(task["prompt_token_ids"]).encode()
                        ).hexdigest(),
                    )
                except Exception:
                    row.update(
                        status="failed",
                        end_s=time.monotonic(),
                        error=traceback.format_exc(),
                    )
                records.append(row)
                sink.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                sink.flush()
                print(
                    "REAL_REQUEST",
                    row["id"],
                    row["status"],
                    row.get("output_tokens"),
                    flush=True,
                )
                if row["status"] != "success":
                    raise RuntimeError(
                        "Real request failed; later requests remain not_run"
                    )
        write(
            out / "normalized.json",
            {
                "run_id": str(uuid.uuid4()),
                "cards": config["cards"],
                "evidence": {
                    "effective_config": "effective-config.json",
                    "requests": "requests.jsonl",
                    "runtime": "runtime.json",
                    "warmup": "warmup.json",
                    "service_log": "stage-01.log",
                },
                "scope": config.get(
                    "role", "supplementary diagnostic, no formal B0/B1 comparison"
                ),
                "requests_completed": len(records),
            },
        )
    except BaseException:
        write(
            out / "failure.json",
            {
                "error": traceback.format_exc(),
                "requests_attempted": len(records),
                "requests_planned": len(workload["requests"]),
            },
        )
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except Exception:
                write(out / "shutdown-error.json", {"error": traceback.format_exc()})
        write(
            out / "terminal.json",
            {"elapsed_s": time.monotonic() - started, "requests": len(records)},
        )


if __name__ == "__main__":
    main()
