"""Pinned original render components in qualified offline containers.

Input must be unchanged parsed_code from the author's original workflow. This
worker does not replace that parser, native generation, syntax scoring or VQA.
Plan by default; never execute candidate code on the host.
"""

import argparse
import asyncio
from functools import partial
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace

RENDERER_SHA256 = "7c5a37b25df94bf370cc37dda86d269bcc071680a5ec3b07484a777ffee0b09f"  # pragma: allowlist secret - fixed original author module
RENDERERS = {
    "matplotlib": (RENDERER_SHA256, "render_matplotlib_and_screenshot"),
    "latex": (
        "b53cd11950d6eb77cf42e71d55e020950ddd538f54252328c134440a8a40832d",  # pragma: allowlist secret - fixed original author module
        "render_latex_to_png",
    ),
    "typst": (
        "b6e7bf8e8ebeaca7836cc4d853f7a87772e46e386e38ada07386e0015dd2a39c",  # pragma: allowlist secret - fixed original author module
        "render_typst_and_screenshot",
    ),
    "svg": (
        "58539c3236da7cf26e3af3c1fc4e13036320822422bcb79dbdd797c96d438c06",  # pragma: allowlist secret - fixed original author module
        "render_svg_and_screenshot",
    ),
    "html": (
        "9903eecf78ff9201d54b8cc2aebad1908ba5e68d5e2fdc8426cdb72a07c40f99",  # pragma: allowlist secret - fixed original author module
        "render_html_and_screenshot",
    ),
    "canvas": (
        "06bc0e0790cac32f54ae724147ee7d096e5d34a3fc74d837aab0558f24632806",  # pragma: allowlist secret - fixed original author module
        "render_canvas_and_screenshot",
    ),
    "markdown": (
        "4daf5cd98a27786a23a2aa372cfcd087e2a12520add3b32bc07cde39dc44a1f0",  # pragma: allowlist secret - fixed original author module
        "render_markdown_and_screenshot",
    ),
}
EXTRA_MODULES = {
    "svg": {
        "render_utils.py": "947172ba0d07979caf4180db0f9204d9c8db2aad81dae7f2608e5ece33932de4",  # pragma: allowlist secret - original relative import
    }
}
EXTRA_MODULES["html"] = dict(EXTRA_MODULES["svg"])
for _kind in ("canvas", "markdown"):
    EXTRA_MODULES[_kind] = {
        **EXTRA_MODULES["svg"],
        "render_html.py": RENDERERS["html"][0],
    }


async def browser_without_external_assets(original_start, requested, *args, **kwargs):
    """Same author factory/arguments; block unfetched network assets, not model0."""
    browser, context, page, playwright = await original_start(*args, **kwargs)

    async def guard(route):
        url = route.request.url
        if url.startswith(("http://", "https://", "ws://", "wss://")):
            requested.append(url)
            await route.abort()
        else:
            await route.continue_()

    await context.route("**/*", guard)
    return browser, context, page, playwright


def guard_browser_factories(module, namespace, requested):
    """Guard nested pinned renderer factories, never unrelated imported packages."""
    targets = [module] + [
        value
        for name, value in list(sys.modules.items())
        if name.startswith(namespace + ".")
    ]
    bindings = [
        (target, target.start_browser)
        for target in targets
        if hasattr(target, "start_browser")
    ]
    for target, original_start in bindings:
        target.start_browser = partial(
            browser_without_external_assets, original_start, requested
        )


def run_retaining_errors(original_run, *args, **kwargs):
    """Keep author's exact command/return/exception; retain captured diagnostics."""
    try:
        return original_run(*args, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            raw = (
                error.stderr.encode("utf-8", "replace")
                if isinstance(error.stderr, str)
                else error.stderr
            )
            os.write(
                2, b"\nOriginal captured compiler/converter stderr:\n" + raw + b"\n"
            )
        raise


def invoke_original_renderer(module, kind, task_id, code, output):
    """Preserve the author's distinct LaTeX positional contract and defaults."""
    render = getattr(module, RENDERERS[kind][1])
    if kind == "latex":
        return render(code, str(output), task_id)
    return render(task_id, code, str(output))


def require_isolation():
    if os.name != "posix" or os.geteuid() == 0 or os.getpid() != 1:
        raise ValueError("nonroot Linux container PID1 required")
    if not Path("/.dockerenv").is_file() or not os.statvfs("/").f_flag & os.ST_RDONLY:
        raise ValueError("read-only Docker container required")
    if {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise ValueError("network-none container required")
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    if (
        any(
            int(status[key].strip(), 16)
            for key in ("CapEff", "CapPrm", "CapInh", "CapBnd")
        )
        or status["NoNewPrivs"].strip() != "1"
    ):
        raise ValueError("drop all capabilities and disallow privilege escalation")
    if any(
        Path(path).exists() for path in ("/var/run/docker.sock", "/run/docker.sock")
    ) or list(Path("/dev").glob("davinci*")):
        raise ValueError("no Docker API or NPU device may be exposed")
    if any(
        value
        and re.search(
            r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIALS?)(?:$|_)",
            key,
        )
        for key, value in os.environ.items()
    ):
        raise ValueError(
            "renderer must not inherit common credential environment variables"
        )


def verify_inputs(renderer, code_file, expected_sha256, renderer_kind="matplotlib"):
    if renderer_kind not in RENDERERS:
        raise ValueError("explicit supported original renderer required")
    renderer = Path(renderer).resolve(strict=True)
    if hashlib.sha256(renderer.read_bytes()).hexdigest() != RENDERERS[renderer_kind][0]:
        raise ValueError("unchanged pinned original renderer required")
    for name, expected in EXTRA_MODULES.get(renderer_kind, {}).items():
        path = renderer.parent / name
        if (
            path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("unchanged original relative module required")
    code_file = Path(code_file).resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("explicit parsed-code SHA256 required")
    if code_file.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(
            "parsed-code file exceeds declared worker input resource bound"
        )
    raw = code_file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(
            "parsed-code snapshot changed; never reconstruct replacement code"
        )
    return renderer, code_file, raw.decode("utf-8")


def execute(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.task_id):
        raise ValueError("original safe task identifier required")
    result = {
        "source_id": "STRUCTEVAL",
        "task_id": args.task_id,
        "status": "PLAN_ONLY",
        "model_calls": 0,
        "vqa_executed": False,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        return result
    require_isolation()
    os.umask(0o077)
    kind = getattr(args, "renderer_kind", "matplotlib")
    renderer, code_file, code = verify_inputs(
        args.renderer, args.code_file, args.code_sha256, kind
    )
    readonly_inputs = [renderer, code_file] + [
        renderer.parent / name for name in EXTRA_MODULES.get(kind, {})
    ]
    if not all(os.statvfs(path).f_flag & os.ST_RDONLY for path in readonly_inputs):
        raise ValueError("original module and parsed code must be mounted read-only")
    output = args.output.resolve()
    output_root = next(
        (
            base
            for base in (Path("/tmp"), Path("/output"))
            if output.is_relative_to(base)
        ),
        None,
    )
    if output_root is None or output.exists():
        raise ValueError("fresh bounded tmpfs output required")
    mounts = [
        line.split(" - ", 1)
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
    ]
    if not any(
        left.split()[4] == str(output_root) and right.split()[0] == "tmpfs"
        for left, right in mounts
    ):
        raise ValueError("output root must be an explicit bounded tmpfs mount")
    space = os.statvfs(output_root)
    if space.f_blocks * space.f_frsize > 256 * 1024**2:
        raise ValueError("output tmpfs exceeds declared256MiB bound")
    output.mkdir(mode=0o700, parents=True)
    # This file-size bound complements the Docker CPU/RAM/PID/time/tmpfs limits.
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    saved_fds = [os.dup(1), os.dup(2)]
    try:
        with (output / "renderer-private.log").open("xb") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            package = ModuleType("_pinned_structeval_renderer")
            package.__path__ = [str(renderer.parent)]
            sys.modules[package.__name__] = package
            spec = importlib.util.spec_from_file_location(
                package.__name__ + ".component", renderer
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            requested = []
            if kind in EXTRA_MODULES:
                # Canvas/Markdown call a relative HTML renderer. Guard every
                # pinned module's existing factory without rewriting its body.
                guard_browser_factories(module, package.__name__, requested)
            if kind == "typst":
                author_run = module.subprocess.run
                module.subprocess = SimpleNamespace(
                    run=lambda *a, **kw: run_retaining_errors(author_run, *a, **kw)
                )
            score = invoke_original_renderer(module, kind, args.task_id, code, output)
            if inspect.isawaitable(score):
                score = asyncio.run(score)
            if requested:
                (output / "unavailable-assets-private.json").write_text(
                    json.dumps(requested), encoding="utf-8"
                )
                raise ValueError(
                    "unfetched original browser assets; resource block, not model score"
                )
    finally:
        for target, descriptor in zip((1, 2), saved_fds):
            os.dup2(descriptor, target)
            os.close(descriptor)
    if type(score) is not int or score not in (0, 1):
        raise ValueError("unexpected original renderer result")
    image = output / (args.task_id + ".png")
    if image.exists() and (not image.is_file() or image.is_symlink()):
        raise ValueError("nonregular named output retained as resource/protocol block")
    png_valid = False
    if image.is_file() and not image.is_symlink():
        from PIL import Image

        try:
            with Image.open(image) as pixels:
                png_valid = pixels.format == "PNG"
                pixels.verify()
        except (OSError, ValueError):
            png_valid = False
    result.update(
        status="ORIGINAL_RENDERER_RETURNED_VQA_PENDING",
        render_score=score,
        png_valid=png_valid,
        parsed_code_sha256=args.code_sha256,
        original_renderer_sha256=RENDERERS[kind][0],
        renderer_kind=kind,
        image_present=image.is_file() and not image.is_symlink(),
        original_parser_relation_verified_by_worker=False,
        native_pipeline_complete=False,
    )
    with (output / "component-result-private.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(result, stream, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--renderer-kind", choices=sorted(RENDERERS), default="matplotlib"
    )
    args = parser.parse_args()
    try:
        result = execute(args)
    except (ValueError, OSError) as error:
        parser.exit(
            1,
            f"StructEval worker rejected ({type(error).__name__}); retain private container logs\n",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
