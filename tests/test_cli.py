"""Tests for CLI/gateway-server wiring."""

from tequila_mule.cli import GatewayServer
from tequila_mule.config import Config


def test_advertise_host_defaults_to_real_hostname_on_bind_all(monkeypatch):
    """`gateway.host: 0.0.0.0` (bind all interfaces) is not a routable address
    a compute node can dial for /internal/register — it must resolve to the
    machine's real hostname instead, or every backend registration silently
    times out after 5 minutes."""
    monkeypatch.setattr("socket.gethostname", lambda: "login-node.cluster.example")
    config = Config(**{"gateway": {"host": "0.0.0.0", "port": 8000}})
    assert GatewayServer._advertise_host(config) == "login-node.cluster.example"


def test_advertise_host_resolves_ipv6_bind_all(monkeypatch):
    """The IPv6 bind-all address is the same trap as 0.0.0.0."""
    monkeypatch.setattr("socket.gethostname", lambda: "login-node.cluster.example")
    config = Config(**{"gateway": {"host": "::", "port": 8000}})
    assert GatewayServer._advertise_host(config) == "login-node.cluster.example"


def test_advertise_host_explicit_override_wins(monkeypatch):
    """An explicit `advertise_host` always wins, even over a bind-all host."""
    monkeypatch.setattr("socket.gethostname", lambda: "login-node.cluster.example")
    config = Config(
        **{"gateway": {"host": "0.0.0.0", "port": 8000, "advertise_host": "gw.internal"}}
    )
    assert GatewayServer._advertise_host(config) == "gw.internal"


def test_advertise_host_reuses_specific_bind_host():
    """A non-bind-all host is already routable, so it's reused as-is."""
    config = Config(**{"gateway": {"host": "10.0.0.5", "port": 8000}})
    assert GatewayServer._advertise_host(config) == "10.0.0.5"
