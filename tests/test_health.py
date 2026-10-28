"""Tests for the backend health monitor."""

from unittest.mock import MagicMock

import pytest
import respx
from httpx import Response as HttpxResponse

from birdbrain.health import HealthMonitor


def _monitor(registry: dict[str, list[str]]) -> HealthMonitor:
    gateway = MagicMock()
    gateway.get_all_backend_urls = lambda name: registry.get(name, [])
    return HealthMonitor("laguna-xs", gateway, extra_backend_names=["nomic-embed"])


@pytest.mark.asyncio
async def test_dead_sidecar_counts_as_down_even_with_healthy_primary():
    """The secondary shares the job, so its outage must surface for a restart."""
    monitor = _monitor({"laguna-xs": ["http://g043:50001"], "nomic-embed": ["http://g043:50002"]})
    with respx.mock:
        respx.get("http://g043:50001/health").mock(return_value=HttpxResponse(200))
        respx.get("http://g043:50002/health").mock(return_value=HttpxResponse(500))
        assert await monitor._check_once() == ["nomic-embed"]


@pytest.mark.asyncio
async def test_one_dead_replica_is_degraded_not_down():
    monitor = _monitor({"laguna-xs": ["http://g043:50001", "http://g043:50002"]})
    with respx.mock:
        respx.get("http://g043:50001/health").mock(return_value=HttpxResponse(200))
        respx.get("http://g043:50002/health").mock(return_value=HttpxResponse(503))
        assert await monitor._check_once() == []


@pytest.mark.asyncio
async def test_nothing_registered_reports_unknown():
    assert await _monitor({})._check_once() is None
