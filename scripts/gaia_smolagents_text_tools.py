"""Declared text/web tool profile; not the original open_deep_research runner.

Reuses stable smolagents search/parser and page formatting, restricts network
destinations to public addresses, and sends Python only to the isolated tool.
Binary attachment and audiovisual readers remain unqualified in this profile.
"""

import http.client
import ipaddress
import os
import socket
import ssl
from urllib.parse import quote, urljoin, urlsplit
import uuid


def public_target(url):
    value = urlsplit(url)
    if (
        value.scheme not in {"http", "https"}
        or not value.hostname
        or value.username
        or value.password
        or len(url) > 8192
    ):
        raise ValueError("bounded public HTTP(S) URL without credentials required")
    port = value.port or (443 if value.scheme == "https" else 80)
    if port != (443 if value.scheme == "https" else 80):
        raise ValueError("only public standard HTTP(S) ports supported")
    addresses = socket.getaddrinfo(value.hostname, port, type=socket.SOCK_STREAM)
    ips = {row[4][0] for row in addresses}
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError(
            "private, loopback, metadata and mixed DNS destinations denied"
        )
    return value, port, sorted(ips)[0]


def public_page(url):
    """Pin each validated DNS address and revalidate every redirect."""
    for _ in range(6):
        value, port, address = public_target(url)
        connection = http.client.HTTPConnection(value.hostname, port, timeout=20)
        raw = socket.create_connection((address, port), timeout=20)
        try:
            if value.scheme == "https":
                raw = ssl.create_default_context().wrap_socket(
                    raw, server_hostname=value.hostname
                )
            connection.sock = raw
            path = (value.path or "/") + ("?" + value.query if value.query else "")
            connection.request(
                "GET", path, headers={"User-Agent": "HUST-GAIA-validation/1.0"}
            )
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise ValueError("redirect has no destination")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise RuntimeError(f"public web HTTP status {response.status}")
            mime = response.getheader("Content-Type", "").split(";")[0]
            if not (
                mime.startswith("text/")
                or mime in {"application/json", "application/xhtml+xml"}
            ):
                raise ValueError("nontext media needs a separately qualified reader")
            body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("public page exceeds declared2MiB tool budget")
            return body.decode("utf-8", errors="replace")
        finally:
            connection.close()
            raw.close()
    raise ValueError("public page redirect limit exceeded")


def make_tools(task_input, task_dir):
    from smolagents import Tool, VisitWebpageTool, WebSearchTool
    from gaia_isolated_python_tool import run_code

    class Search(WebSearchTool):
        def search_duckduckgo(self, query):
            parser = self._create_duckduckgo_parser()
            parser.feed(
                public_page("https://lite.duckduckgo.com/lite/?q=" + quote(query))
            )
            return parser.results

    class Visit(VisitWebpageTool):
        def forward(self, url: str) -> str:
            from markdownify import markdownify
            import re

            text = re.sub(r"\n{3,}", "\n\n", markdownify(public_page(url)).strip())
            return self._truncate_content(text, self.max_output_length)

    class Python(Tool):
        name = "python_interpreter"
        description = "Run Python using the original smolagents interpreter in a resource-limited offline container. No host files or network."
        inputs = {"code": {"type": "string", "description": "Python code to execute."}}
        output_type = "string"

        def forward(self, code: str) -> str:
            return run_code(
                code,
                docker_host=os.environ["GAIA_DOCKER_HOST"],
                image=os.environ["GAIA_PYTHON_IMAGE"],
                framework_venv=os.environ["GAIA_FRAMEWORK_VENV"],
                output=task_dir / ("python-" + uuid.uuid4().hex),
            )

    return [Search(engine="duckduckgo"), Visit(), Python()]
