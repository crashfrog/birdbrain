"""Backend health monitoring."""

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitor one pool's instances and trigger a restart when any served
    backend (the primary or a co-located secondary) has no healthy instance.

    All of a pool's instances share one Slurm job, so restarting the job is
    the only remedy for a dead sidecar too. A replica-only outage (some but
    not all instances of a backend down) is logged as degraded and does not
    restart: the surviving replicas keep serving.
    """

    def __init__(
        self,
        backend_name: str,
        gateway,
        check_interval: int = 30,
        extra_backend_names: Optional[list[str]] = None,
    ) -> None:
        self.backend_name = backend_name
        self.gateway = gateway
        self.check_interval = check_interval
        # Every gateway backend_name this pool's job serves.
        self.backend_names = [backend_name, *(extra_backend_names or [])]
        self.is_healthy = True
        self.running = False
        self.task: Optional[asyncio.Task] = None
        self.on_unhealthy_callback: Optional[callable] = None
        self.on_healthy_callback: Optional[callable] = None

    def set_unhealthy_callback(self, callback: callable) -> None:
        """Set callback to trigger on backend failure."""
        self.on_unhealthy_callback = callback

    def set_healthy_callback(self, callback: callable) -> None:
        """Set (sync) callback run after every check where all backends answer."""
        self.on_healthy_callback = callback

    async def start(self) -> None:
        """Start health monitoring loop."""
        self.running = True
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitor stopped")

    async def _check_once(self) -> Optional[list[str]]:
        """Probe every registered instance. Returns the backend names with no
        healthy instance, or None when nothing is registered yet."""
        groups = {
            name: urls
            for name in self.backend_names
            if (urls := self.gateway.get_all_backend_urls(name))
        }
        if not groups:
            return None

        down = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for name, urls in groups.items():
                healthy = 0
                for url in urls:
                    try:
                        resp = await client.get(f"{url}/health")
                        if resp.status_code == 200:
                            healthy += 1
                        else:
                            logger.warning(f"[{name}] {url} unhealthy: HTTP {resp.status_code}")
                    except httpx.RequestError as e:
                        logger.warning(f"[{name}] {url} unreachable: {e}")
                if healthy == 0:
                    down.append(name)
                elif healthy < len(urls):
                    logger.warning(f"[{name}] Degraded: {healthy}/{len(urls)} instances healthy")
        return down

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        consecutive_failures = 0

        while self.running:
            down = await self._check_once()
            if down is None:
                # Nothing registered (cold start / restart in progress): reset,
                # so a failing replacement job can trigger its own restart.
                self.is_healthy = True
                consecutive_failures = 0
            elif not down:
                if not self.is_healthy:
                    logger.info(f"[{self.backend_name}] Backend recovered")
                self.is_healthy = True
                consecutive_failures = 0
                if self.on_healthy_callback:
                    self.on_healthy_callback()
            else:
                consecutive_failures += 1
                logger.warning(
                    f"[{self.backend_name}] No healthy instance for {down} "
                    f"({consecutive_failures} consecutive)"
                )
                # Mark unhealthy after 3 consecutive failures
                if consecutive_failures >= 3 and self.is_healthy:
                    self.is_healthy = False
                    logger.error(f"[{self.backend_name}] Backend marked unhealthy")
                    if self.on_unhealthy_callback:
                        await self.on_unhealthy_callback()

            await asyncio.sleep(self.check_interval)