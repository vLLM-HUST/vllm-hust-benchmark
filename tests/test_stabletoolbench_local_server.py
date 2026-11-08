from pathlib import Path
import runpy
import subprocess
import sys

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


def test_installed_audit_hook_blocks_real_socket_calls():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_stabletoolbench_local_server.py"
    )
    code = """
import runpy, socket, sys
guard = runpy.run_path(sys.argv[1])['local_network_only']
sys.addaudithook(guard)
blocked = 0
for call in (lambda: socket.getaddrinfo('example.com', 443),
             lambda: socket.socket().connect(('192.0.2.1', 80))):
    try:
        call()
    except PermissionError:
        blocked += 1
assert blocked == 2
with socket.socket() as listener:
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    with socket.create_connection(listener.getsockname(), timeout=2):
        connection, _ = listener.accept()
        connection.close()
print('LOOPBACK_ALLOWED_EXTERNAL_DENIED')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LOOPBACK_ALLOWED_EXTERNAL_DENIED"
