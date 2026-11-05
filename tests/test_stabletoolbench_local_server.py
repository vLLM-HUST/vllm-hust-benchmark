from pathlib import Path
import runpy

import pytest


@pytest.mark.parametrize("address", [("127.0.0.1", 19467), ("::1", 19467, 0, 0)])
def test_local_model_connections_allowed(address):
    guard = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts/run_stabletoolbench_local_server.py"
        )
    )["local_network_only"]
    guard("socket.connect", (None, address))


@pytest.mark.parametrize(
    "event,args",
    [
        ("socket.connect", (None, ("8.8.8.8", 443))),
        ("socket.connect", (None, "/run/docker.sock")),
        ("socket.getaddrinfo", ("api.openai.com", 443)),
    ],
)
def test_nonlocal_network_is_rejected(event, args):
    guard = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts/run_stabletoolbench_local_server.py"
        )
    )["local_network_only"]
    with pytest.raises(PermissionError):
        guard(event, args)
