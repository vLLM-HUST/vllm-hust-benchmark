"""Expose the original NOMAD validity check without exposing private labels."""

import argparse
from email import policy
from email.parser import BytesParser
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
from tempfile import TemporaryDirectory


COMPETITION = "nomad2018-predict-transparent-conductors"
MAX_BYTES = 1024 * 1024


class ValidatorHTTP(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": "not found"})
            return
        self.route("GET")

    def do_POST(self):
        if self.path != "/validate":
            self.reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        content_type = self.headers.get("Content-Type", "")
        if (
            not 0 < length <= MAX_BYTES
            or not content_type.startswith("multipart/form-data;")
            or not content_type.isascii()
            or self.headers.get("Transfer-Encoding")
        ):
            self.reply(400, {"error": "bounded multipart file required"})
            return
        self.connection.settimeout(15)
        body = self.rfile.read(length)
        if len(body) != length:
            self.reply(400, {"error": "incomplete submission"})
            return
        self.route("POST", content_type, body)


def extract_submission(content_type, body):
    encoded = b"Content-Type: " + content_type.encode("ascii") + b"\r\n\r\n" + body
    request = BytesParser(policy=policy.default).parsebytes(encoded)
    files = [
        part
        for part in request.iter_parts()
        if part.get_content_disposition() == "form-data"
        and part.get_param("name", header="content-disposition") == "file"
    ]
    if len(files) != 1 or not files[0].get_filename():
        raise ValueError("expected one submission file")
    payload = files[0].get_payload(decode=True)
    if not payload:
        raise ValueError("empty submission file")
    return payload


def validator(data_dir):
    from mlebench.data import is_dataset_prepared
    from mlebench.grade import validate_submission
    from mlebench.registry import Registry

    competition = Registry(data_dir).get_competition(COMPETITION)
    if not is_dataset_prepared(competition, grading_only=True):
        raise ValueError("prepared NOMAD grading data required")

    class Worker(ValidatorHTTP):
        def route(self, method, content_type=None, body=None):
            if method == "GET":
                self.reply(200, {"status": "running"})
                return
            try:
                payload = extract_submission(content_type, body)
            except ValueError:
                self.reply(400, {"error": "invalid multipart submission"})
                return
            with TemporaryDirectory() as tmp:
                path = Path(tmp) / "submission.csv"
                path.write_bytes(payload)
                try:
                    valid, _ = validate_submission(path, competition)
                except Exception:
                    self.reply(500, {"error": "submission validation failed"})
                    return
            # Do not return the score, private labels or exception diagnostics.
            self.reply(
                200,
                {"result": "Submission is valid." if valid else "Submission invalid!"},
            )

    return Worker


class UnixHTTPConnection(HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=20)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self.path))


def frontend(socket_path):
    class Agent(ValidatorHTTP):
        def route(self, method, content_type=None, body=None):
            headers = {"Content-Type": content_type} if content_type else {}
            conn = UnixHTTPConnection(socket_path)
            try:
                conn.request(method, self.path, body=body, headers=headers)
                response = conn.getresponse()
                payload = json.loads(response.read(1024))
                self.reply(response.status, payload)
            except (OSError, ValueError):
                self.reply(502, {"error": "isolated submission validator unavailable"})
            finally:
                conn.close()

    return Agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("worker", "frontend"))
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.role == "worker":
        if args.data_dir is None or args.socket.exists() or args.socket.is_symlink():
            parser.error("worker needs private --data-dir and a new socket")
        from socketserver import UnixStreamServer

        os.umask(0o077)
        with UnixStreamServer(str(args.socket), validator(args.data_dir)) as server:
            server.serve_forever()
    else:
        if args.data_dir is not None:
            parser.error("frontend must not receive a private data directory")
        with ThreadingHTTPServer(("127.0.0.1", 5000), frontend(args.socket)) as server:
            server.serve_forever()


if __name__ == "__main__":
    main()
