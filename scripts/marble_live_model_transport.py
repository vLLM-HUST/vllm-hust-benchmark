"""Task-scoped loopback forwarding; no orphan replay or model implementation.

The context exists only inside the original MARBLE child. Every invocation gets
a fresh queue. Its worker stops dispatching when the original child returns;
an in-flight response is retained before shutdown. Nothing restarts a model.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit

from appworld_model_transport import model_proxy, process_one, save_new


def canonical_request(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def saved_responses(root, expected_model):
    """Validate explicit private custody; never turn an unanswered claim into a hit.

    The operator must separately verify the current backend binding. A model
    alias or a self-authored manifest is not evidence of weight/device parity.
    Reuse is for recovering this validation run, never performance measurement.
    """
    root = Path(root).resolve(strict=True)
    manifest_path = root / "provenance-private.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("bounded original saved-response manifest required")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if (
        not isinstance(expected_model, str)
        or not expected_model
        or manifest.get("source_id") != "MULTIAGENTBENCH"
        or manifest.get("model") != expected_model
        or manifest.get("historical_backend_same_api_birth_verified") is not True
        or manifest.get("original_nonthinking_binding") is not True
    ):
        raise ValueError("explicit original saved-response model binding required")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not 0 < len(rows) <= 1000:
        raise ValueError("bounded original saved-response inventory required")
    suffixes = {
        ".request.json",
        ".forwarded-request.json",
        ".response.raw",
        ".response.json",
    }
    cache = {}
    total_bytes = 0
    for row in rows:
        identity = row["identity"]
        if not re.fullmatch("[a-f0-9]{32}", identity) or set(row["sha256"]) != suffixes:
            raise ValueError("complete original returned-response identity required")
        raw = {}
        for suffix, digest in row["sha256"].items():
            path = root / (identity + suffix)
            if path.is_symlink() or path.stat().st_size > 1024 * 1024:
                raise ValueError("bounded non-symlink original response asset required")
            total_bytes += path.stat().st_size
            if total_bytes > 16 * 1024 * 1024:
                raise ValueError("saved-response inventory exceeds16MiB limit")
            raw[suffix] = path.read_bytes()
            if hashlib.sha256(raw[suffix]).hexdigest() != digest:
                raise ValueError("original saved-response hash mismatch")
        request = json.loads(raw[".request.json"])
        options = request.get("chat_template_kwargs", {})
        if options.get("enable_thinking", False) is not False:
            raise ValueError("conflicting original thinking setting")
        expected = {
            **request,
            "chat_template_kwargs": {**options, "enable_thinking": False},
        }
        response = json.loads(raw[".response.json"])
        if (
            request.get("model") != expected_model
            or json.loads(raw[".forwarded-request.json"]) != expected
            or response.get("http_status") != 200
            or json.loads(raw[".response.raw"]) != response.get("response")
        ):
            raise ValueError("original request/forward/returned response mismatch")
        key = canonical_request(request)
        if key in cache:
            raise ValueError("ambiguous saved responses for one original request")
        cache[key] = {"original_identity": identity, "raw": raw}
    return cache


def uncertain_requests(root):
    """Carry known unanswered claims forward as explicit no-resubmission guards."""
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "provenance-private.json").read_text("utf-8"))
    count = manifest.get("excluded_claim_without_saved_response", 0)
    if type(count) is not int or not 0 <= count <= 1000:
        raise ValueError("bounded unresolved-claim count required")
    if not count:
        return set()
    path = root / "uncertain-requests-private.json"
    if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("bounded unresolved-claim identities required")
    rows = json.loads(path.read_text("utf-8"))
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("all known unanswered claims must be guarded")
    hashes = {row["canonical_request_sha256"] for row in rows}
    if len(hashes) != count or any(
        not re.fullmatch("[a-f0-9]{64}", value) for value in hashes
    ):
        raise ValueError("complete unanswered-claim request hashes required")
    return hashes


def reuse_one(queue, cache, reuse_state, blocked=()):
    """Reuse only the next exact original request; otherwise leave it to the backend."""
    for path in sorted(
        queue.glob("*.request.json"), key=lambda p: p.stat().st_mtime_ns
    ):
        identity = path.name.removesuffix(".request.json")
        if path.is_symlink() or not re.fullmatch("[a-f0-9]{32}", identity):
            raise ValueError("original private queue identity required")
        if (queue / (identity + ".response.json")).exists() or (
            queue / (identity + ".claim")
        ).exists():
            continue
        key = canonical_request(json.loads(path.read_bytes()))
        if hashlib.sha256(key.encode()).hexdigest() in blocked:
            raise ValueError("known unanswered original claim; inspect, never resubmit")
        cached = cache.get(key)
        if cached is None:
            return False
        marker = {
            "original_identity": cached["original_identity"],
            "new_model_calls": 0,
            "reused_saved_response": True,
        }
        # One original returned sample can satisfy at most one new logical call,
        # including across the three original child processes of this run.
        try:
            with (reuse_state / (cached["original_identity"] + ".json")).open(
                "x"
            ) as stream:
                json.dump({**marker, "new_request_identity": identity}, stream)
        except FileExistsError:
            if not (
                reuse_state / (cached["original_identity"] + ".complete.json")
            ).exists():
                raise ValueError(
                    "uncertain saved-response consumption; inspect, never resubmit"
                )
            del cache[key]
            return False
        try:
            with (queue / (identity + ".claim")).open("x") as stream:
                json.dump(marker, stream)
        except FileExistsError:
            del cache[key]
            continue
        del cache[key]
        for suffix in [".forwarded-request.json", ".response.raw"]:
            save_new(queue / (identity + suffix), cached["raw"][suffix])
        save_new(
            queue / (identity + ".reuse-private.json"), json.dumps(marker).encode()
        )
        save_new(
            queue / (identity + ".response.json"),
            cached["raw"][".response.json"],
            owner_uid=path.stat().st_uid,
        )
        save_new(
            reuse_state / (cached["original_identity"] + ".complete.json"),
            json.dumps(marker).encode(),
        )
        return True
    return False


def loopback_endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError("explicit owned loopback /v1 endpoint required")
    return value.rstrip("/")


@contextmanager
def live_model_transport(
    queue,
    api_base,
    *,
    port=14005,
    timeout=600,
    reuse_root=None,
    expected_model=None,
    reuse_state=None,
):
    """Preserve original requests; only the disclosed nonthinking binding is added.

    This is an HTTP transport, not proof of a NPU-backed service. The operator
    must qualify the bound backend and resource window before native execution.
    Existing or uncertain queues are rejected, never resumed implicitly.
    """
    api_base = loopback_endpoint(api_base)
    if port and urlsplit(api_base).port == port:
        raise ValueError("backend must not point back to the model proxy")
    if timeout <= 0:
        raise ValueError("positive model transport timeout required")
    cache = (
        saved_responses(reuse_root, expected_model) if reuse_root is not None else {}
    )
    blocked = uncertain_requests(reuse_root) if reuse_root is not None else set()
    if reuse_root is not None:
        reuse_state = (
            Path(reuse_state).resolve(strict=True) if reuse_state is not None else None
        )
        if (
            reuse_state is None
            or not reuse_state.is_dir()
            or reuse_state.is_relative_to(Path(reuse_root).resolve())
        ):
            raise ValueError(
                "separate current-run response consumption directory required"
            )
    queue = Path(queue)
    if queue.exists() or queue.is_symlink():
        raise ValueError("fresh task-scoped model queue required; no orphan replay")
    queue.mkdir(mode=0o700)
    stop = threading.Event()
    errors = []

    def serve():
        try:
            while not stop.is_set():
                if (cache or blocked) and reuse_one(queue, cache, reuse_state, blocked):
                    continue
                if not process_one(
                    queue,
                    api_base,
                    disable_thinking=True,
                    timeout=timeout,
                ):
                    stop.wait(0.05)
        except Exception as error:
            errors.append(error)
            with (queue / "worker-error-private.json").open("x") as stream:
                json.dump({"type": type(error).__name__, "message": str(error)}, stream)
            stop.set()

    with model_proxy(queue, port=port, timeout=timeout) as actual_port:
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            yield actual_port
        finally:
            stop.set()
            worker.join(timeout + 1)
            if worker.is_alive():
                raise RuntimeError(
                    "model transport still in flight; inspect claim, never resubmit"
                )
        if errors:
            raise RuntimeError(
                "model transport failed; retained private error"
            ) from errors[0]
