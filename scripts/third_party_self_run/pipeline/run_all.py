"""Execute the kit, run real native inference and render all four target dispositions."""

import argparse
import csv
import fcntl
import hashlib
from html import escape
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from measurements import group_metrics

ROOT = Path(__file__).resolve().parent
REPORT_PYTHON = os.environ.get(
    "VLLM_HUST_REPORT_PYTHON", str(ROOT.parent / ".report-venv/bin/python")
)


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n")


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def acquire():
    handles = []
    try:
        for device in [5, 6]:
            for name in [
                f"/root/npu-device-{device}.lock",
                f"/run/lock/betterscale-npu-{device}.lock",
                f"/tmp/gennova-device-leases/device-{device}.lock",
                f"/tmp/gennova-physical-{device}.lock",
                f"/tmp/npu-physical-{device}.lock",
                f"/tmp/zllm2-npu-{device}.lock",
            ]:
                p = Path(name)
                p.parent.mkdir(parents=True, exist_ok=True)
                handle = p.open("a+")
                handles.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle = Path("/tmp/gennova-device-leases/container-model-loading.lock").open(
            "a+"
        )
        handles.append(handle)
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        text = subprocess.check_output(["npu-smi", "info"], text=True)
        for device in [5, 6]:
            if f"No running processes found in NPU {device}" not in text:
                raise RuntimeError(f"Physical NPU {device} is busy")
        return handles
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def execute(out):
    config = json.loads((ROOT / "diagnostic-config.json").read_text())
    queued = time.monotonic()
    while True:
        try:
            handles = acquire()
            break
        except (BlockingIOError, RuntimeError) as error:
            waited = time.monotonic() - queued
            save(
                out / "resource-wait.json",
                {
                    "waited_s": waited,
                    "reason": str(error),
                    "policy": "wait without preempting other model owners",
                },
            )
            if waited >= 1800:
                raise TimeoutError("Resource queue exceeded 1800 seconds") from error
            time.sleep(5)
    mapping = subprocess.check_output(["npu-smi", "info", "-m"], text=True)
    (out / "device-mapping.txt").write_text(mapping)
    logic = {}
    for line in mapping.splitlines():
        words = line.split()
        if (
            len(words) >= 3
            and words[0] in ["5", "6"]
            and words[1] == "0"
            and words[2].isdigit()
        ):
            logic[int(words[0])] = int(words[2])
    if set(logic) != {5, 6}:
        raise RuntimeError("Cannot verify physical-to-logical mapping")
    env = dict(
        os.environ,
        ASCEND_RT_VISIBLE_DEVICES=",".join(str(logic[i]) for i in [5, 6]),
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_NO_USAGE_STATS="1",
        TRITON_CACHE_DIR=str(out / "cache/triton"),
        VLLM_CACHE_ROOT=str(out / "cache/vllm"),
        TORCHINDUCTOR_CACHE_DIR=str(out / "cache/inductor"),
        VLLM_HUST_TEST_OUTPUT=str(out),
        VLLM_HUST_TEST_CASE="supplementary-real-inference",
        VLLM_HUST_TEST_ARM=config.get("arm", "LOCAL023"),
        VLLM_HUST_TEST_REPEAT="1",
        VLLM_HUST_TEST_CONFIG=str(ROOT / "diagnostic-config.json"),
        VLLM_HUST_TEST_CONTRACT=str(ROOT / "diagnostic-contract.json"),
    )
    env["PYTHONPATH"] = ":".join(
        config.get("compat_pythonpath", []) + config.get("extra_pythonpath", [])
    )
    if config.get("general_plugins"):
        env["VLLM_PLUGINS"] = ",".join(config["general_plugins"])
    save(
        out / "invocation.json",
        {
            "argv": ["python3", "run_native.py", "native-workflow.json"],
            "mapping": logic,
            "visible_devices": env["ASCEND_RT_VISIBLE_DEVICES"],
            "started_epoch": time.time(),
            "pid": os.getpid(),
        },
    )
    child = None
    try:
        with (
            (out / "adapter.log").open("w") as log,
            (out / "telemetry.jsonl").open("w") as telemetry,
        ):
            child = subprocess.Popen(
                ["python3", "run_native.py", "native-workflow.json"],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 1540
            while child.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("Native workflow deadline exceeded")
                observation = subprocess.run(
                    ["npu-smi", "info"], capture_output=True, text=True, timeout=15
                )
                telemetry.write(
                    json.dumps(
                        {
                            "monotonic_s": time.monotonic(),
                            "epoch": time.time(),
                            "raw": observation.stdout,
                            "returncode": observation.returncode,
                        }
                    )
                    + "\n"
                )
                telemetry.flush()
                stage_log = out / "stage-01.log"
                if stage_log.exists():
                    body = stage_log.read_text(errors="replace")
                    if any(
                        message in body
                        for message in [
                            "EngineCore failed to start",
                            "Worker failed with error",
                            "Engine core initialization failed",
                        ]
                    ):
                        save(
                            out / "controlled-abort.json",
                            {
                                "stage": "engine_initialization",
                                "reason": "Native worker reported a fatal initialization error; terminate owned stages without waiting for outer timeout",
                            },
                        )
                        raise RuntimeError(
                            "Native engine initialization failed; see stage-01.log"
                        )
                time.sleep(5)
            save(
                out / "execution.json",
                {"returncode": child.returncode, "finished_epoch": time.time()},
            )
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        from run_native import cleanup_group

        try:
            actions = []
            for record in sorted(out.glob("stage-*-process.json")):
                actions.extend(cleanup_group(json.loads(record.read_text()), out))
            save(
                out / "cleanup.json",
                {"quiescent": True, "actions": actions, "finished_epoch": time.time()},
            )
        finally:
            for handle in handles:
                handle.close()


def render(out):
    # Run rendering in an existing isolated plotting environment; inference uses
    # the actual installed NPU stack. No global packages are replaced.
    from inference_test_report import SECTIONS, write_report, plotter
    from run_vllm_hust_tests import THRESHOLDS

    plan = json.loads((ROOT / "execution-plan.json").read_text())
    rows = []
    # Missing thresholds are never plotted as zero or used as acceptance criteria.
    plot_cases = json.loads(json.dumps(plan["cases"]))
    for case in plot_cases:
        if case["kind"] == "context":
            case["rules"]["max_tail_ratio"] = {}
        row = {
            "case": case["id"],
            "kind": case["kind"],
            "repeat": 1,
            "arms": {},
            "status": "unassessable",
            "reason": case["blocker"],
        }
        rows.append(row)
        (out / case["id"] / "r001").mkdir(parents=True, exist_ok=True)
    # The stock renderer only dereferences thresholds for present comparison rows.
    # Plotter fix below hides unknown thresholds, never substitutes a number.
    import inference_test_report as renderer

    original = renderer.render_panels

    def render_known_thresholds(plt, path, title, panels):
        cleaned = [
            (
                metric,
                [
                    (
                        label,
                        values,
                        target if target and target[0] is not None else None,
                    )
                    for label, values, target in entries
                ],
            )
            for metric, entries in panels
        ]
        return original(plt, path, title, cleaned)

    renderer.render_panels = render_known_thresholds
    summary = {
        "plan_sha256": digest(ROOT / "execution-plan.json"),
        "cases": [r["case"] for r in rows],
        "complete_plan": False,
        "thresholds": THRESHOLDS,
        "targets": {s: "unassessable" for s, *_ in SECTIONS},
        "overall_status": "unassessable",
        "results": rows,
        "report_origin": "readiness-and-real-diagnostic wrapper; not a sealed formal run",
        "preflight_returncode": json.loads((out / "preflight.json").read_text())[
            "returncode"
        ],
    }
    save(out / "summary.json", summary)
    plt = plotter()
    write_report(out, plot_cases, summary, plt)
    probe = out / "native"
    records = []
    if (probe / "requests.jsonl").exists():
        records = [
            json.loads(line)
            for line in (probe / "requests.jsonl").read_text().splitlines()
            if line
        ]
    successful = [r for r in records if r["status"] == "success"]
    failure = (
        json.loads((probe / "failure.json").read_text())
        if (probe / "failure.json").exists()
        else {}
    )
    diagnostic = {
        "planned": len(
            json.loads((ROOT / "diagnostic-workload.json").read_text())["requests"]
        ),
        "attempted": len(records),
        "completed": len(successful),
        "not_run": len(
            json.loads((ROOT / "diagnostic-workload.json").read_text())["requests"]
        )
        - len(records),
        "failure": failure,
        "receipt_exported": (probe / "receipt.json").exists(),
        "groups": {},
    }
    with (out / "real-inference.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "group",
                "status",
                "input_tokens",
                "output_tokens",
                "elapsed_s",
                "ttft_ms",
                "tpot_ms",
                "finish_reason",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.get(k)
                    for k in [
                        "id",
                        "group",
                        "status",
                        "input_tokens",
                        "output_tokens",
                        "elapsed_s",
                        "ttft_ms",
                        "tpot_ms",
                        "finish_reason",
                    ]
                ]
            )
    for group in ["mmlu-diagnostic", "context-4096", "context-16384"]:
        subset = [r for r in successful if r["group"] == group]
        diagnostic["groups"][group] = group_metrics(subset)
    save(out / "diagnostic-summary.json", diagnostic)
    if successful:
        fig, axes = plt.subplots(4, figsize=(11, 12), layout="constrained")
        fig.suptitle("Real installed-stack TP2 diagnostic (not B0/B1 acceptance)")
        names = [r["id"] for r in successful]
        axes[0].bar(names, [r["elapsed_s"] for r in successful], color="#18728e")
        axes[0].set_ylabel("generate wall seconds")
        axes[1].bar(names, [r["output_tokens"] for r in successful], color="#697c99")
        axes[1].set_ylabel("actual output tokens")
        axes[2].bar(
            names,
            [r.get("ttft_ms") or float("nan") for r in successful],
            color="#18728e",
        )
        axes[2].set_ylabel("native TTFT (ms)")
        axes[3].bar(
            names,
            [r.get("tpot_ms") or float("nan") for r in successful],
            color="#697c99",
        )
        axes[3].set_ylabel("TPOT (ms/token)")
        for ax in axes:
            ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        fig.savefig(out / "charts/real-inference.png", dpi=165)
        plt.close(fig)
    intro = (
        '<section style="border:2px solid #a65c28;padding:16px"><h2>本次执行范围与卡点</h2>'
        "<p>原版正式入口已实际执行；预检未通过。下面四项均为“无法判定”，不是数值未达标，也没有冒用历史成绩。"
        "缺失值保留为空，图上的 No valid measurement 不表示0分。正式重复次数、业务阈值、费用和部署身份未确认时不自行填造。</p>"
        f"<p>本机真实推理：计划10请求，实际尝试{len(records)}，成功{len(successful)}，未执行{10 - len(records)}。"
        "实际版本与源码见runtime.json和冻结配置；BF16、双卡物理5/6，图模式。轻负载诊断不替代正式验收。</p>"
        '<p><a href="preflight.log">原版预检报错</a> · <a href="diagnostic-summary.json">实机汇总</a> · '
        '<a href="real-inference.csv">逐请求指标</a> · <a href="native/stage-01.log">原生日志</a> · '
        '<a href="../execution-plan.json">执行清单</a> · <a href="../local-assets.json">本地数据路径</a></p>'
        + (f"<pre>{escape(failure.get('error', ''))}</pre>" if failure else "")
        + "</section>"
    )
    p = out / "report.html"
    text = p.read_text().replace("<body>", "<body>" + intro)
    if successful:
        text = text.replace(
            "</body>",
            '<section><h2>真实推理补充记录</h2><img src="charts/real-inference.png"></section></body>',
        )
    p.write_text(text)
    save(
        out / "checksums.json",
        {
            p.relative_to(out).as_posix(): digest(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "checksums.json" and "cache" not in p.parts
        },
    )
    print("REPORT", out / "report.html", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="run-001")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    if args.render_only:
        render(out)
        return
    if not (ROOT / "execution-plan.json").exists():
        subprocess.run(["python3", str(ROOT / "prepare.py")], check=True, cwd=ROOT)
    plan = json.loads((ROOT / "execution-plan.json").read_text())
    for name, expected in plan["files"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(
                f"Frozen file changed: {name}; create a new protocol version"
            )
    out.mkdir(exist_ok=False)
    with (out / "preflight.log").open("w") as log:
        pre = subprocess.run(
            [
                REPORT_PYTHON,
                str(ROOT / "run_vllm_hust_tests.py"),
                "run",
                "--plan",
                str(ROOT / "execution-plan.json"),
                "--sha256",
                digest(ROOT / "execution-plan.json"),
                "--out",
                str(out / "formal"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    save(out / "preflight.json", {"returncode": pre.returncode})
    probe = out / "native"
    probe.mkdir()
    try:
        execute(probe)
    except BaseException as e:
        save(
            probe / "orchestrator-error.json",
            {"error": str(e), "type": type(e).__name__},
        )
    # A failed cleanup must not produce an integrity seal over live log writers.
    if (
        list(probe.glob("stage-*-process.json"))
        and not (probe / "cleanup.json").exists()
    ):
        raise RuntimeError(
            "Native processes are not confirmed quiescent; refusing to seal outputs"
        )
    subprocess.run(
        [REPORT_PYTHON, str(ROOT / "run_all.py"), "--out", str(out), "--render-only"],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
