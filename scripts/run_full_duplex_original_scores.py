"""Invoke original duplex scorers on native captures, ASR and timing artifacts.

Default is plan-only. Neural phases use external NPU judge or original NPU VAD.
Local judge results are explicitly non-official and never relabeled as GPT.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

PHASES = (
    "v10-nonjudge",
    "v10-backchannel-npu",
    "v10-interruption-localjudge",
    "v15-timing-npu",
    "v15-localjudge",
    "v15-summary",
)
NONJUDGE_SUBSETS = (
    "synthetic_pause_handling",
    "candor_pause_handling",
    "candor_turn_taking",
)


def native_arguments(args):
    """Original v1.5 argument values; no crop, limit, threshold or retry rewrite."""
    return argparse.Namespace(
        phase={
            "v15-timing-npu": "timing",
            "v15-localjudge": "custom-judge",
            "v15-summary": "summarize",
        }[args.phase],
        tree=[args.engine + "=" + str(args.tree)],
        manifest=[],
        manifest_projection=None,
        only=[],
        out=args.scores,
        reference_source=args.reference_source,
        limit=None,
        retry_failed=False,
        audio_loader="soundfile",
        bootstrap=2000,
        seed=20260925,
        source_scores=args.scores,
        judge_config=args.judge_config,
        concurrency=1,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        timeout_s=120.0,
        retry_sleep_s=5.0,
    )


def backchannel(args, paths):
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend
    import soundfile
    from benchmarks.duplex.reference_core import load_module
    from benchmarks.duplex.reference_v10 import parse_result_block
    from full_duplex_npu_vad import load_npu_silero

    torch.npu.set_device(0)
    original = load_module(paths["backchannel"], "original_backchannel_npu_binding")

    def load_pcm_wave(path):
        if soundfile.info(path).subtype != "PCM_16":
            raise ValueError(
                "validated backchannel decoder requires original PCM16 WAV"
            )
        wave, rate = soundfile.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(wave.T.copy()), rate

    previous_load = original.torchaudio.load
    original.load_silero_vad = load_npu_silero
    original.torchaudio.load = load_pcm_wave
    stdout = io.StringIO()
    try:
        with (
            contextlib.chdir(paths["backchannel"].parent),
            contextlib.redirect_stdout(stdout),
        ):
            original.eval_backchannel(str(args.tree / "icc_backchannel"))
    finally:
        original.torchaudio.load = previous_load
    (args.output / "original-stdout.txt").write_text(
        stdout.getvalue(), encoding="utf-8"
    )
    return parse_result_block(stdout.getvalue())


def score(args):
    from benchmarks.duplex.reference_source import verify_reference

    if args.phase.startswith("v10-"):
        from benchmarks.duplex.reference_core import V10_REFERENCE_FILES
        from benchmarks.duplex.reference_v10 import evaluate, JudgeLedger

        paths = verify_reference(args.reference_source, V10_REFERENCE_FILES)
        if args.phase == "v10-backchannel-npu":
            return backchannel(args, paths)
        judge = None
        subsets = list(NONJUDGE_SUBSETS)
        if args.phase == "v10-interruption-localjudge":
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ[args.api_key_env],
                base_url=args.base_url,
                timeout=120,
            )
            judge = JudgeLedger(
                client,
                args.output / "judge-exchanges.jsonl",
                str(client.base_url),
                args.served_model,
            )
            subsets = ["synthetic_user_interruption"]
        result = evaluate(args.tree, paths, subsets, judge)
        return {name: result["subsets"][name]["result"] for name in subsets}

    from benchmarks.eval.benchmark_duplex_reference import open_engines

    native = native_arguments(args)
    paths = verify_reference(args.reference_source)
    engines = open_engines(native)
    if args.phase == "v15-summary":
        from benchmarks.duplex.reference_summary import run_summarize

        return run_summarize(native, engines, paths)
    if args.phase == "v15-localjudge":
        from benchmarks.duplex.reference_custom_judge import run_custom

        if args.local_judge_output.exists():
            raise FileExistsError(
                "local judge output must be new; reuse completed judgments"
            )
        native.out = args.local_judge_output
        counts = run_custom(native, engines, paths)
    else:
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend
        from benchmarks.duplex.reference_core import HashCache
        from benchmarks.duplex.reference_timing import run_timing
        from full_duplex_npu_vad import load_npu_silero

        torch.npu.set_device(0)
        hashes = HashCache(native.out / "hash-cache.json")
        counts = run_timing(
            native, engines, paths, hashes, silero_loader=load_npu_silero
        )
        hashes.save()
    if not counts or counts.get("failed", 0) or counts.get("invalid", 0):
        raise RuntimeError(
            "original phase returned empty/failed/invalid results: " + str(counts)
        )
    return dict(counts)


def validate(args):
    if not args.engine or "=" in args.engine:
        raise ValueError("engine must be a nonempty label without '='")
    if args.phase.startswith("v15-") and args.scores is None:
        raise ValueError("original v1.5 scores directory required")
    if "localjudge" in args.phase:
        url = urlsplit(args.base_url or "")
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("explicit credential-free local judge base URL required")
        if args.phase == "v10-interruption-localjudge" and not args.served_model:
            raise ValueError("explicit actual served local judge model required")
        if args.phase == "v15-localjudge" and (
            args.judge_config is None or args.local_judge_output is None
        ):
            raise ValueError(
                "original custom-judge config and separate new output required"
            )
        if (
            args.local_judge_output is not None
            and args.local_judge_output == args.scores
        ):
            raise ValueError(
                "local judge output must remain separate from original scores"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    for flag in ["omni-root", "reference-source", "tree", "output"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    for flag in ["scores", "judge-config", "local-judge-output"]:
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--engine", default="minicpmo-npu-derived")
    parser.add_argument("--base-url")
    parser.add_argument("--served-model")
    parser.add_argument("--api-key-env", default="FULL_DUPLEX_LOCAL_JUDGE_API_KEY")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for key in [
        "omni_root",
        "reference_source",
        "tree",
        "output",
        "scores",
        "judge_config",
        "local_judge_output",
    ]:
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    validate(args)
    report = {
        "source_id": "FULL_DUPLEX_BENCH",
        "phase": args.phase,
        "status": "PLAN_ONLY_NO_MODEL_CALLS",
        "source_checkbox_eligible": False,
        "official_gpt_judge_executed": False,
        "local_judge_results_are_non_official": "localjudge" in args.phase,
    }
    if not args.execute:
        print(json.dumps(report))
        return 0
    repository = Path(__file__).resolve().parents[1]
    for path in (args.output, args.tree, args.scores, args.local_judge_output):
        if path is not None and path.is_relative_to(repository):
            raise ValueError("native trees, scores and new output must remain external")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.omni_root))
    try:
        with (args.output / "original-score.log").open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                report["result"] = score(args)
        report["status"] = "ORIGINAL_PHASE_COMPLETED_NOT_WHOLE_BENCHMARK"
    except (Exception, SystemExit) as error:
        report.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        (args.output / "qualification-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
