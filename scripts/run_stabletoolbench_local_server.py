"""Run pinned author cache/simulator logic with loopback-only networking.

The real-API tier is explicitly unavailable (local HTTP 503). Original cache
and simulator code remain unchanged; this is not live ToolBench API execution.
"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlsplit


PINS = {
    "main.py": "3631ec36eb9c2540f6e9f65392006002986f7321748b08ac571b80fc2d7eb65e",  # pragma: allowlist secret - author SHA256
    "utils.py": "25a85e8e72be920046968453a64d70b5b1653bab6d4add573e159a3169b8204c",  # pragma: allowlist secret - author SHA256
}


def local_network_only(event, arguments):
    if event == "socket.connect":
        address = arguments[1]
        if (
            not isinstance(address, tuple)
            or not ipaddress.ip_address(address[0]).is_loopback
        ):
            raise PermissionError("non-loopback connection denied")
    if event == "socket.getaddrinfo":
        if arguments[0] not in ("127.0.0.1", "::1", "localhost"):
            raise PermissionError("external DNS denied")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-server", "cache", "tools", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--simulator-endpoint", required=True)
    parser.add_argument("--simulator-model", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    endpoint = urlsplit(args.simulator_endpoint)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname != "127.0.0.1"
        or not endpoint.port
        or endpoint.path != "/v1"
        or endpoint.query
        or endpoint.fragment
        or endpoint.username
        or not 0 < args.port < 65536
    ):
        parser.error("explicit loopback /v1 simulator and valid local port required")
    root = args.author_server.resolve(strict=True)
    for name, digest in PINS.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            parser.error(f"unrecognized author file: {name}")
    cache = args.cache.resolve(strict=True)
    tools = args.tools.resolve(strict=True)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "real_api_tier": "LOCAL_503",
                    "host": "127.0.0.1",
                }
            )
        )
        return

    class Unavailable(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(503)
            self.end_headers()

        def log_message(self, *_):
            pass

    fallback = ThreadingHTTPServer(("127.0.0.1", 0), Unavailable)
    threading.Thread(target=fallback.serve_forever, daemon=True).start()
    args.output.mkdir(exist_ok=False)
    config = dict(
        api_key="local-loopback",  # pragma: allowlist secret - noncredential for loopback-only service
        api_base=args.simulator_endpoint,
        model=args.simulator_model,
        temperature=0,
        toolbench_url=f"http://127.0.0.1:{fallback.server_port}/unavailable",
        tools_folder=str(tools),
        cache_folder=str(cache),
        is_save=False,
        port=args.port,
        log_file=str(args.output.resolve() / "server.private.log"),
    )
    (args.output / "config.yml").write_text(json.dumps(config), encoding="utf-8")

    sys.addaudithook(local_network_only)
    os.chdir(args.output)
    sys.path.insert(0, str(root))
    import uvicorn

    spec = importlib.util.spec_from_file_location(
        "stabletoolbench_author_server", root / "main.py"
    )
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    uvicorn.run(author.app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
