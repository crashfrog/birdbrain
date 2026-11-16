"""Slurm job lifecycle management."""
        # Slurm's wall-time clock starts now, not at submission: a job can sit
        # queued for hours. Counting from submission would trigger the next
        # rotation that much too early, overlapping jobs and holding GPUs.
        job.expires_at = (
            start_time + _parse_wall_time(self.backend_config.slurm.wall_time)
        ).isoformat()
                previous = self.current_job
                if previous is not None and previous.job_id != job.job_id:
                    # The gateway already routes to the new job; the old one
                    # would otherwise keep its GPUs and ports untracked until
                    # its wall time.
                    self._cancel_job(previous, f"replaced by job {job.job_id}")

import asyncio
import json
import logging
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from .config import BackendPoolConfig, PathsConfig

logger = logging.getLogger(__name__)

# Backoff between replacement jobs after consecutive failures (a job that
# died, never registered, or was marked unhealthy). Without it, a fault that
# kills every replacement — e.g. the login node losing its route to the
# compute nodes, so health checks fail while registration callbacks still
# land — resubmits a GPU job every few minutes indefinitely. Delays double
# from the base up to the cap; one passing health check resets them.
RESTART_BACKOFF_BASE_SECONDS = 60
RESTART_BACKOFF_MAX_SECONDS = 1800


@dataclass
class VLLMInstance:
    """One vLLM server process inside a pool's Slurm job."""

    port: int
    model_name: str
    backend_name: str  # gateway pool this instance registers under
    # Indices into the job's GPU allocation; None = inherit the whole allocation.
    gpu_indices: Optional[list[int]]
    vllm_extra_args: str
    env: dict[str, str]
    secondary: bool  # secondaries start only after every primary replica is healthy


@dataclass
class JobState:
    """State of a Slurm job and the vLLM instances it serves."""

    job_id: str
    node: Optional[str]
    status: str  # PENDING, RUNNING, COMPLETED, FAILED
    submitted_at: str
    expires_at: str
    expected_ports: list[int] = field(default_factory=list)
    # backend_name -> URLs registered by this job (primary replicas + secondaries)
    registered_backends: dict[str, list[str]] = field(default_factory=dict)


def _parse_wall_time(wall_time: str) -> timedelta:
    """Parse wall_time (HH:MM:SS or D-HH:MM:SS) into a timedelta."""
    if "-" in wall_time:
        days_str, time_str = wall_time.split("-")
        days = int(days_str)
        h, m, s = map(int, time_str.split(":"))
        return timedelta(days=days, hours=h, minutes=m, seconds=s)

    h, m, s = map(int, wall_time.split(":"))
    return timedelta(hours=h, minutes=m, seconds=s)


class LifecycleManager:
    """Manage Slurm job submission and rotation for a single backend pool."""

    def __init__(
        self,
        backend_config: BackendPoolConfig,
        paths: PathsConfig,
        gateway_host: str,
        gateway_port: int,
        gateway: "Gateway",  # type: ignore
    ) -> None:
        self.name = backend_config.name
        self.backend_config = backend_config
        self.paths = paths
        self.gateway_host = gateway_host
        self.gateway_port = gateway_port
        self.gateway = gateway
        self.current_job: Optional[JobState] = None
        self.pending_job: Optional[JobState] = None
        self.running = False
        self.task: Optional[asyncio.Task] = None
        # Guard against concurrent cold-start triggers from the rotation loop
        # and the health monitor's unhealthy callback racing each other.
        self._restart_in_progress: bool = False
        # Number of consecutive squeue misses before declaring a job dead.
        # squeue can transiently return empty (e.g. brief scheduler hiccup);
        # requiring 3 misses (~90s) avoids false-positive restarts.
        self._squeue_fail_count: int = 0
        # Replacement jobs that failed in a row since the last passing health
        # check; drives the resubmission backoff (RESTART_BACKOFF_*).
        self._consecutive_failures: int = 0

        # Setup Jinja2 for job template rendering
        template_dir = Path(paths.job_template).parent
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        self.template_name = Path(paths.job_template).name

        # Per-backend state file, e.g. ~/.birdbrain/state-default.json
        self.state_file = Path(paths.state_file).expanduser().with_name(
            f"state-{self.name}.json"
        )
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Legacy single-backend state file, used as a one-time migration
        # fallback so an in-place upgrade of an existing "default" backend
        # doesn't lose track of an already-running job.
        self._legacy_state_file = Path(paths.state_file).expanduser()

    def _build_instance_plan(self) -> list[VLLMInstance]:
        """Instances this pool's job launches, in port order: primary replicas
        (each owning an equal, contiguous slice of the job's GPUs), then
        secondary models pinned to their configured GPU."""
        slurm = self.backend_config.slurm
        model = self.backend_config.model
        ports = self.backend_config.instance_ports()
        gpus_per_replica = slurm.gpus_per_job // slurm.replicas

        instances = [
            VLLMInstance(
                port=ports[i],
                model_name=model.name,
                backend_name=self.name,
                gpu_indices=(
                    None
                    if slurm.replicas == 1
                    else list(range(i * gpus_per_replica, (i + 1) * gpus_per_replica))
                ),
                vllm_extra_args=model.vllm_extra_args,
                env=model.env,
                secondary=False,
            )
            for i in range(slurm.replicas)
        ]
        for j, sec in enumerate(self.backend_config.secondary_models):
            instances.append(
                VLLMInstance(
                    port=ports[slurm.replicas + j],
                    model_name=sec.model_name,
                    backend_name=sec.backend_name,
                    gpu_indices=[sec.gpu_device],
                    vllm_extra_args=sec.vllm_extra_args,
                    env=sec.env,
                    secondary=True,
                )
            )
        return instances

    def _backend_model_names(self) -> dict[str, str]:
        """backend_name -> served model id, for every pool this job serves."""
        names = {self.name: self.backend_config.model.name}
        for sec in self.backend_config.secondary_models:
            names[sec.backend_name] = sec.model_name
        return names

    def _job_from_dict(self, data: dict) -> JobState:
        """Build a JobState, migrating the pre-multi-instance shape
        (single `port` + `backend_url`) to expected_ports/registered_backends."""
        data = dict(data)
        port = data.pop("port", None)
        url = data.pop("backend_url", None)
        data.setdefault("expected_ports", [port] if port else [])
        data.setdefault("registered_backends", {self.name: [url]} if url else {})
        return JobState(**data)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_file
        if not state_file.exists():
            if self.name == "default" and self._legacy_state_file.exists():
                logger.info(
                    f"No {state_file} found; migrating legacy state from "
                    f"{self._legacy_state_file}"
                )
                state_file = self._legacy_state_file
            else:
                return

        try:
            with open(state_file) as f:
                data = json.load(f)

            if data.get("current_job"):
                self.current_job = self._job_from_dict(data["current_job"])
                logger.info(f"[{self.name}] Loaded current job: {self.current_job.job_id}")

            if data.get("pending_job"):
                self.pending_job = self._job_from_dict(data["pending_job"])
                logger.info(f"[{self.name}] Loaded pending job: {self.pending_job.job_id}")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            data = {
                "current_job": asdict(self.current_job) if self.current_job else None,
                "pending_job": asdict(self.pending_job) if self.pending_job else None,
                "updated_at": datetime.utcnow().isoformat(),
            }

            # Atomic write
            tmp_file = self.state_file.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
            tmp_file.replace(self.state_file)

        except Exception as e:
            logger.error(f"[{self.name}] Failed to save state: {e}")

    def _submit_job(self) -> str:
        """Submit a new Slurm job. Returns job_id."""
        submitted_at = datetime.utcnow()
        duration = _parse_wall_time(self.backend_config.slurm.wall_time)
        expires_at = submitted_at + duration  # noqa: F841 (kept for parity/clarity)

        # Render job script
        template = self.jinja_env.get_template(self.template_name)
        instances = self._build_instance_plan()
        script = template.render(
            partition=self.backend_config.slurm.partition,
            gres=self.backend_config.slurm.gres,
            wall_time=self.backend_config.slurm.wall_time,
            container_path=self.paths.container_path,
            gateway_host=self.gateway_host,
            gateway_port=self.gateway_port,
            memory=self.backend_config.slurm.memory,
            cpus_per_task=self.backend_config.slurm.cpus_per_task,
            instances=instances,
        )

        # Build sbatch args: add --exclude if we know which node is currently
        # serving this backend (to avoid port conflict when the old job's vLLM
        # is still alive briefly after its job expires).
        sbatch_args = ["sbatch", "--parsable"]
        if self.current_job and self.current_job.node:
            sbatch_args.append(f"--exclude={self.current_job.node}")

        # Submit via sbatch
        logger.info(f"[{self.name}] Submitting Slurm job")
        result = subprocess.run(
            sbatch_args,
            input=script.encode(),
            capture_output=True,
            check=True,
        )
        job_id = result.stdout.decode().strip()
        logger.info(f"[{self.name}] Job submitted: {job_id} (args: {' '.join(sbatch_args[1:])})")

        return job_id

    def _get_job_info(self, job_id: str) -> Optional[dict]:
        """Query squeue for job info."""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T %N"],
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout.strip()
            if not output:
                return None

            parts = output.split()
            return {
                "state": parts[0],  # PENDING, RUNNING, etc.
                "node": parts[1] if len(parts) > 1 and parts[1] != "(null)" else None,
            }
        except subprocess.CalledProcessError:
            return None

    def _cancel_job(self, job: JobState, reason: str) -> None:
        """scancel a job this manager is about to stop tracking, so it can't
        keep running untracked, holding GPUs and ports on its node."""
        logger.info(f"[{self.name}] Cancelling job {job.job_id}: {reason}")
        subprocess.run(["scancel", job.job_id], capture_output=True)

    def _restart_delay(self) -> int:
        """Seconds to wait before submitting the next replacement job."""
        if self._consecutive_failures == 0:
            return 0
        return min(
            RESTART_BACKOFF_BASE_SECONDS * 2 ** (self._consecutive_failures - 1),
            RESTART_BACKOFF_MAX_SECONDS,
        )

    def record_healthy(self) -> None:
        """Health monitor reached every backend this pool serves: the current
        job works, so end the failure streak and its backoff."""
        if self._consecutive_failures:
            logger.info(
                f"[{self.name}] Backend healthy, resetting restart backoff "
                f"after {self._consecutive_failures} failed job(s)"
            )
        self._consecutive_failures = 0

    def _submit_pending_job(self) -> Optional[JobState]:
        """Submit and track a job. An sbatch failure counts toward the backoff
        and returns None: slurmctld can be unreachable during the same network
        faults that kill jobs, and the caller retries after the delay."""
        try:
            return self._record_pending_job(self._submit_job())
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace").strip() if e.stderr else ""
            logger.error(f"[{self.name}] sbatch failed (exit {e.returncode}): {stderr}")
            self._consecutive_failures += 1
            return None

    async def start(self) -> None:
        """Start lifecycle manager."""
        self.running = True
        self._load_state()

        # Validate loaded state against squeue
        if self.current_job:
            info = self._get_job_info(self.current_job.job_id)
            if not info or info["state"] not in ("RUNNING", "PENDING"):
                logger.warning(
                    f"[{self.name}] Current job {self.current_job.job_id} no longer valid"
                )
                self.current_job = None
            else:
                # Rehydrate the gateway's registry so a restart doesn't briefly
                # 503 requests for an already-running backend.
                model_names = self._backend_model_names()
                for backend_name, urls in self.current_job.registered_backends.items():
                    for url in urls:
                        await self.gateway.set_backend(
                            backend_name,
                            url,
                            model_name=model_names.get(backend_name, ""),
                            job_id=self.current_job.job_id,
                        )

        # A pending job in saved state belonged to a rotation or cold start the
        # previous process was waiting on. Nothing resumes that wait, so cancel
        # it rather than let it start later, untracked, on some node.
        if self.pending_job:
            self._cancel_job(self.pending_job, "pending job from a previous gateway run")
            self.pending_job = None
            self._save_state()

        # Cold start: submit job immediately if no current job.
        # Run as a background task so uvicorn finishes startup and can
        # accept the /internal/register callback from the compute node.
        if not self.current_job:
            self._restart_in_progress = True
            asyncio.create_task(self._cold_start())

        self.task = asyncio.create_task(self._rotation_loop())
        logger.info(f"[{self.name}] Lifecycle manager started")

    async def stop(self) -> None:
        """Stop lifecycle manager."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info(f"[{self.name}] Lifecycle manager stopped")

    def _record_pending_job(self, job_id: str) -> JobState:
        """Track a freshly submitted job as pending and persist it."""
        submitted_at = datetime.utcnow()
        expires_at = submitted_at + _parse_wall_time(self.backend_config.slurm.wall_time)
        self.pending_job = JobState(
            job_id=job_id,
            node=None,
            status="PENDING",
            submitted_at=submitted_at.isoformat(),
            expires_at=expires_at.isoformat(),
            expected_ports=self.backend_config.instance_ports(),
        )
        self._save_state()
        return self.pending_job

    async def _cold_start(self) -> None:
        """Submit jobs until one registers every instance, backing off after
        each failed attempt.

        Callers set `_restart_in_progress` first. It is released however this
        ends, so a later failure can trigger another restart instead of being
        skipped forever.
        """
        try:
            while self.running:
                delay = self._restart_delay()
                if delay:
                    logger.warning(
                        f"[{self.name}] {self._consecutive_failures} consecutive failed job(s); "
                        f"waiting {delay}s before resubmitting"
                    )
                    await asyncio.sleep(delay)
                    if not self.running:
                        return
                logger.info(f"[{self.name}] Cold start: submitting initial job")
                job = self._submit_pending_job()
                if job is None:
                    continue
                await self._wait_for_job_ready(job)
                if self.current_job is job:
                    return
                self._consecutive_failures += 1
                if self.pending_job is job:
                    self.pending_job = None
                    self._save_state()
        finally:
            self._restart_in_progress = False

    async def _wait_for_job_ready(self, job: JobState) -> None:
        """Wait for job to reach RUNNING and for every planned instance
        (primary replicas and secondaries) to register with the gateway."""
        logger.info(f"[{self.name}] Waiting for job {job.job_id} to become ready")
        expected = Counter(inst.backend_name for inst in self._build_instance_plan())

        # Poll until RUNNING
        while self.running:
            info = self._get_job_info(job.job_id)
            if not info:
                logger.error(f"[{self.name}] Job {job.job_id} disappeared from queue")
                job.status = "FAILED"
                self._save_state()
                return

            if info["state"] == "RUNNING":
                job.status = "RUNNING"
                job.node = info["node"]
                logger.info(f"[{self.name}] Job {job.job_id} running on {job.node}")
                break

            await asyncio.sleep(5)

        # Wait for the job to register. Graph-mode (CUDA graphs / torch.compile)
        # cold starts are slow — first-run compile plus weight load and graph
        # capture can take ~15 min on a cold compile cache. 20 min matches the
        # job template's startup deadline, which covers every instance.
        start_time = datetime.utcnow()
        while self.running:
            if (datetime.utcnow() - start_time).total_seconds() > 1200:
                logger.error(f"[{self.name}] Job {job.job_id} failed to register within 20 minutes")
                # Still RUNNING in Slurm: left alone it would hold its GPUs and
                # ports untracked until wall time.
                self._cancel_job(job, "did not register within 20 minutes")
                job.status = "FAILED"
                self._save_state()
                return

            # Match registrations by job_id, not URL: the registered URL comes
            # from the job's own `hostname` (often FQDN) while squeue reports
            # the short name, so URLs never string-match.
            registered = {
                backend_name: self.gateway.get_backend_urls_by_job(backend_name, job.job_id)
                for backend_name in expected
            }
            if all(len(registered[name]) >= count for name, count in expected.items()):
                logger.info(f"[{self.name}] Job {job.job_id} registered all instances: {registered}")
                job.registered_backends = registered
                self.current_job = job
                self.pending_job = None
                self._save_state()
                self._squeue_fail_count = 0
                return

            await asyncio.sleep(2)

    async def _clear_gateway_backends(self) -> None:
        """Drop this job's registrations for the primary and every secondary."""
        for backend_name in self._backend_model_names():
            await self.gateway.clear_backend(backend_name)

    async def clear_backend_and_restart(self) -> None:
        """Clear stale state and cold-start a replacement.

        Called by the health monitor's unhealthy callback when a backend
        vLLM process is unreachable. Without this, a crashed backend lingers
        until the next scheduled rotation (up to 6 hours with the current
        lead_time), leaving clients with 503s the entire time.
        """
        logger.warning(f"[{self.name}] Health monitor reports backend unhealthy, triggering restart")
        if self._restart_in_progress:
            logger.warning(f"[{self.name}] Restart already in progress, skipping")
            return
        self._restart_in_progress = True
        self._consecutive_failures += 1
        await self._clear_gateway_backends()
        if self.current_job:
            # The job may still be RUNNING in Slurm with a hung vLLM. Cancel it
            # so it releases the GPU and port before the replacement starts.
            self._cancel_job(self.current_job, "backend unhealthy")
        if self.pending_job:
            # A rotation replacement still queued or starting. The cold start
            # supersedes it; left running it could land on the same node as the
            # replacement, and whichever registers second hides the other.
            self._cancel_job(self.pending_job, "superseded by cold start")
        self.current_job = None
        self.pending_job = None
        self._save_state()
        asyncio.create_task(self._cold_start())

    async def _rotation_loop(self) -> None:
        """Main rotation loop."""
        while self.running:
            if not self.current_job:
                await asyncio.sleep(10)
                continue

            # Check if the current job is still alive in the Slurm queue.
            # A job that FAILED, was CANCELLED, or timed out disappears from
            # squeue. squeue can transiently return empty (transient scheduler
            # hiccup, network glitch), so we require 3 consecutive misses
            # before triggering a restart (~90s at the 30s check interval).
            info = self._get_job_info(self.current_job.job_id)
            if not info or info["state"] not in ("RUNNING", "PENDING"):
                self._squeue_fail_count += 1
                if self._squeue_fail_count >= 3:
                    if self._restart_in_progress:
                        logger.debug(f"[{self.name}] Restart already in progress, skip duplicate")
                    else:
                        logger.warning(
                            f"[{self.name}] Current job {self.current_job.job_id} "
                            f"missing from squeue for {self._squeue_fail_count} checks, "
                            f"cold-starting replacement"
                        )
                        self._restart_in_progress = True
                        self._consecutive_failures += 1
                        await self._clear_gateway_backends()
                        self.current_job = None
                        self._save_state()
                        asyncio.create_task(self._cold_start())
                        await asyncio.sleep(10)
                        continue
                else:
                    logger.warning(
                        f"[{self.name}] squeue miss {self._squeue_fail_count}/3 "
                        f"(job state={info['state'] if info else 'GONE'})"
                    )
            else:
                # Reset counter on success
                self._squeue_fail_count = 0
            # Calculate time until rotation needed
            expires_at = datetime.fromisoformat(self.current_job.expires_at)
            lead_time = timedelta(minutes=self.backend_config.slurm.lead_time_minutes)
            rotation_time = expires_at - lead_time

            now = datetime.utcnow()
            if now >= rotation_time and not self.pending_job:
                # Submit next job
                logger.info(f"[{self.name}] Rotation window reached, submitting next job")

                job = self._submit_pending_job()
                if job is None:
                    await asyncio.sleep(self._restart_delay())
                    continue

                # Wait for pending job to be ready, then promote
                await self._wait_for_job_ready(job)
                if self.pending_job is job:
                    # Neither promoted nor superseded by a cold start: it died
                    # or never registered. Clear it so the next pass retries,
                    # after the backoff.
                    self.pending_job = None
                    self._save_state()
                    self._consecutive_failures += 1
                    await asyncio.sleep(self._restart_delay())

            await asyncio.sleep(30)  # Check every 30s
