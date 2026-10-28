"""Private file transport between offline Agent and trusted model worker.

Never execute task code here. Mount only a dedicated queue subdirectory, not
Docker sockets or their parents. Raw requests/responses stay server-private.
"""

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


def save_new(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
    temporary.replace(path)


@contextmanager
def model_proxy(queue, *, port=14001, timeout=600):
    queue = Path(queue).resolve(strict=True)
    if not queue.is_dir() or timeout <= 0:
        raise ValueError(
            "existing dedicated private queue and positive timeout required"
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 1024 * 1024:
                self.send_error(413)
                return
            raw = self.rfile.read(size)
            json.loads(raw)
            identity = uuid.uuid4().hex
            save_new(queue / (identity + ".request.json"), raw)
            response = queue / (identity + ".response.json")
            deadline = time.monotonic() + timeout
            while not response.exists():
                if time.monotonic() >= deadline:
                    self.send_error(
                        504, "Retained model job timed out; no resubmission"
                    )
                    return
                time.sleep(0.1)
            payload = json.loads(response.read_bytes())
            wire = json.dumps(payload["response"]).encode()
            self.send_response(payload["http_status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(wire)))
            self.end_headers()
            self.wfile.write(wire)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def process_one(queue, api_base, *, api_key=None, disable_thinking=False, timeout=600):
    endpoint = urlsplit(api_base)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
        or endpoint.username
        or endpoint.password
    ):
        raise ValueError(
            "owned loopback service or explicit SSH-forwarded endpoint required"
        )
    queue = Path(queue).resolve(strict=True)
    for path in sorted(
        queue.glob("*.request.json"), key=lambda p: p.stat().st_mtime_ns
    ):
        identity = path.name.removesuffix(".request.json")
        if (
            len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)
            or path.is_symlink()
        ):
            raise ValueError("invalid private queue identity")
        result = queue / (identity + ".response.json")
        if result.exists():
            continue
        try:
            claim = (queue / (identity + ".claim")).open("xb")
        except FileExistsError:
            continue
        with claim:
            raw = path.read_bytes()
            body = json.loads(raw)
            if disable_thinking:
                options = body.setdefault("chat_template_kwargs", {})
                if (
                    "enable_thinking" in options
                    and options["enable_thinking"] is not False
                ):
                    raise ValueError(
                        "do not override an original conflicting thinking setting"
                    )
                options["enable_thinking"] = False
            forwarded = json.dumps(body).encode()
            save_new(queue / (identity + ".forwarded-request.json"), forwarded)
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = "Bearer " + api_key
            request = Request(
                api_base.rstrip("/") + "/chat/completions",
                data=forwarded,
                headers=headers,
            )
            try:
                response = urlopen(request, timeout=timeout)
            except HTTPError as error:
                response = error
            with response:
                status, wire = response.code, response.read()
            save_new(queue / (identity + ".response.raw"), wire)
            payload = {"http_status": status, "response": json.loads(wire)}
            save_new(result, json.dumps(payload).encode())
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument("--disable-thinking", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute-one",
        action="store_true",
        help="send at most one pending original request; otherwise plan only",
    )
    mode.add_argument(
        "--watch",
        action="store_true",
        help="serve bounded original jobs until stop marker or idle timeout",
    )
    parser.add_argument("--max-jobs", type=int, default=300)
    parser.add_argument("--idle-timeout", type=float, default=600)
    args = parser.parse_args()
    if not args.execute_one and not args.watch:
        print(json.dumps({"status": "PLAN_ONLY", "model_calls": 0}))
        return
    os.umask(0o077)
    if args.max_jobs < 1 or args.idle_timeout <= 0:
        raise ValueError("positive explicit worker bounds required")
    processed = 0
    idle_since = time.monotonic()
    limit = args.max_jobs if args.watch else 1
    while processed < limit:
        if (args.queue / "STOP").exists():
            break
        found = process_one(
            args.queue,
            args.api_base,
            api_key=os.environ[args.api_key_env] if args.api_key_env else None,
            disable_thinking=args.disable_thinking,
        )
        if found:
            processed += 1
            idle_since = time.monotonic()
        elif not args.watch or time.monotonic() - idle_since >= args.idle_timeout:
            break
        else:
            time.sleep(0.2)
    print(
        json.dumps(
            {
                "status": "WORKER_RETURNED_NOT_BENCHMARK_COMPLETION",
                "jobs_returned": processed,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
