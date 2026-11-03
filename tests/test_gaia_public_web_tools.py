"""No external requests: public-address and redirect trust-boundary controls."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "gaia_text_tools",
    Path(__file__).parents[1] / "scripts/gaia_smolagents_text_tools.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://user:pass@example.org/",  # pragma: allowlist secret - invented rejected-URL fixture, no live credential
        "http://example.org:2375/",
    ],
)
def test_disallowed_url(url):
    with pytest.raises(ValueError):
        module.public_target(url)


@pytest.mark.parametrize(
    "ips", [["127.0.0.1"], ["169.254.169.254"], ["10.1.1.1"], ["8.8.8.8", "::1"]]
)
def test_all_dns_addresses_must_be_public(monkeypatch, ips):
    monkeypatch.setattr(
        module.socket,
        "getaddrinfo",
        lambda *a, **k: [(0, 0, 0, "", (ip, 443)) for ip in ips],
    )
    with pytest.raises(ValueError, match="destinations denied"):
        module.public_target("https://example.org/")


def test_public_address_is_returned_for_pinned_connection(monkeypatch):
    monkeypatch.setattr(
        module.socket, "getaddrinfo", lambda *a, **k: [(0, 0, 0, "", ("8.8.8.8", 443))]
    )
    value, port, ip = module.public_target("https://example.org/path")
    assert value.hostname == "example.org" and port == 443 and ip == "8.8.8.8"
