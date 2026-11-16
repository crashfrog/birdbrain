"""Tests for lifecycle manager."""
@pytest.mark.asyncio
@patch("birdbrain.lifecycle.subprocess.run")
async def test_promotion_cancels_replaced_job_and_counts_expiry_from_start(
    mock_run, mock_gateway, tmp_path
):
    """A rotation replacement that registers scancels the job it replaces
    (else it holds its GPUs untracked until wall time), and its expiry counts
    from when it started running, not from when it was queued."""
    mock_gateway.get_backend_urls_by_job = lambda name, job_id: (
        ["http://g043:50001"] if name == "laguna-xs" else ["http://g043:50002"]
    )
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.running = True
    old = _job("100")
    manager.current_job = old
    job = manager._record_pending_job("101")
    job.expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat()  # stale queue-time estimate

    t0 = datetime(2026, 11, 16, 12, 0)
    fake_datetime = MagicMock()
    fake_datetime.utcnow.return_value = t0
    with patch.object(manager, "_get_job_info", return_value={"state": "RUNNING", "node": "g043"}), \
            patch("birdbrain.lifecycle.datetime", fake_datetime):
        await manager._wait_for_job_ready(job)

    assert manager.current_job is job
    assert job.expires_at == (t0 + timedelta(hours=23)).isoformat()  # default wall_time
    cancelled = [c.args[0] for c in mock_run.call_args_list if c.args[0][0] == "scancel"]
    assert cancelled == [["scancel", "100"]]



import json
import subprocess
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from birdbrain.config import Config
from birdbrain.lifecycle import JobState, LifecycleManager


@pytest.fixture
def mock_gateway():
    """Mock gateway."""
    gateway = MagicMock()
    gateway.get_backend = MagicMock(return_value=None)
    gateway.set_backend = AsyncMock()
    return gateway


@pytest.fixture
def lifecycle_manager(mock_gateway, tmp_path):
    """Create lifecycle manager with temp state file."""
    config = Config()
    config.paths.state_file = str(tmp_path / "state.json")
    config.paths.job_template = "birdbrain/templates/vllm_job.sh.j2"

    backend_config = config.backends[0]
    manager = LifecycleManager(
        backend_config, config.paths, "gw-host", 8765, mock_gateway
    )
    return manager


def test_job_state_serialization():
    """Test JobState can be serialized to JSON."""
    job = JobState(
        job_id="12345",
        node="gpu-node-01",
        status="RUNNING",
        submitted_at="2026-05-26T10:00:00",
        expires_at="2026-05-27T09:00:00",
        expected_ports=[50000],
        registered_backends={"default": ["http://gpu-node-01:50000"]},
    )

    # Should be serializable
    from dataclasses import asdict

    data = asdict(job)
    assert data["job_id"] == "12345"
    assert data["node"] == "gpu-node-01"
    assert data["expected_ports"] == [50000]
    assert data["registered_backends"] == {"default": ["http://gpu-node-01:50000"]}

    # Round-trip
    job2 = JobState(**data)
    assert job2.job_id == job.job_id


def test_save_and_load_state(lifecycle_manager):
    """Test state persistence."""
    # Create job state
    submitted_at = datetime.utcnow()
    expires_at = submitted_at + timedelta(hours=23)

    lifecycle_manager.current_job = JobState(
        job_id="12345",
        node="gpu-node-01",
        status="RUNNING",
        submitted_at=submitted_at.isoformat(),
        expires_at=expires_at.isoformat(),
        expected_ports=[50000],
        registered_backends={"default": ["http://gpu-node-01:50000"]},
    )

    # Save state
    lifecycle_manager._save_state()

    # Clear in-memory state
    lifecycle_manager.current_job = None

    # Load state
    lifecycle_manager._load_state()

    # Verify
    assert lifecycle_manager.current_job is not None
    assert lifecycle_manager.current_job.job_id == "12345"
    assert lifecycle_manager.current_job.node == "gpu-node-01"


def test_two_managers_do_not_share_state(mock_gateway, tmp_path):
    """Two LifecycleManagers for different backends must write to separate
    per-backend state files and not clobber each other's state."""
    config = Config()
    config.paths.state_file = str(tmp_path / "state.json")
    config.paths.job_template = "birdbrain/templates/vllm_job.sh.j2"

    backend_a = config.backends[0].model_copy(update={"name": "backend-a"})
    backend_b = config.backends[0].model_copy(update={"name": "backend-b"})

    manager_a = LifecycleManager(backend_a, config.paths, "gw-host", 8765, mock_gateway)
    manager_b = LifecycleManager(backend_b, config.paths, "gw-host", 8765, mock_gateway)

    assert manager_a.state_file != manager_b.state_file

    submitted_at = datetime.utcnow()
    expires_at = submitted_at + timedelta(hours=23)

    manager_a.current_job = JobState(
        job_id="aaa",
        node="g001",
        status="RUNNING",
        submitted_at=submitted_at.isoformat(),
        expires_at=expires_at.isoformat(),
        expected_ports=[50000],
        registered_backends={"backend-a": ["http://g001:50000"]},
    )
    manager_a._save_state()

    manager_b.current_job = JobState(
        job_id="bbb",
        node="g002",
        status="RUNNING",
        submitted_at=submitted_at.isoformat(),
        expires_at=expires_at.isoformat(),
        expected_ports=[50001],
        registered_backends={"backend-b": ["http://g002:50001"]},
    )
    manager_b._save_state()

    manager_a._load_state()
    manager_b._load_state()

    assert manager_a.current_job.job_id == "aaa"
    assert manager_b.current_job.job_id == "bbb"


def test_legacy_state_file_migration(mock_gateway, tmp_path):
    """A "default" backend with no per-backend state file yet should fall
    back to reading the legacy single-backend state.json, so an in-place
    upgrade doesn't lose track of an already-running job."""
    config = Config()
    config.paths.state_file = str(tmp_path / "state.json")
    config.paths.job_template = "birdbrain/templates/vllm_job.sh.j2"

    legacy_manager = LifecycleManager(
        config.backends[0], config.paths, "gw-host", 8765, mock_gateway
    )
    # Simulate the pre-upgrade shape: write old-format state directly
    legacy_manager.state_file = legacy_manager._legacy_state_file
    submitted_at = datetime.utcnow()
    expires_at = submitted_at + timedelta(hours=23)
    # Write old-format state (with port and backend_url, not expected_ports/registered_backends)
    legacy_state = {
        "current_job": {
            "job_id": "legacy-1",
            "node": "g999",
            "port": 50000,
            "status": "RUNNING",
            "submitted_at": submitted_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "backend_url": "http://g999:50000",
        },
        "pending_job": None,
        "updated_at": datetime.utcnow().isoformat(),
    }
    with open(legacy_manager._legacy_state_file, "w") as f:
        json.dump(legacy_state, f)

    # New manager, as constructed post-upgrade, uses the new per-backend path
    new_manager = LifecycleManager(
        config.backends[0], config.paths, "gw-host", 8765, mock_gateway
    )
    assert new_manager.name == "default"
    assert not new_manager.state_file.exists()

    new_manager._load_state()

    assert new_manager.current_job is not None
    assert new_manager.current_job.job_id == "legacy-1"
    # After migration, expected_ports should be populated
    assert new_manager.current_job.expected_ports == [50000]
    assert new_manager.current_job.registered_backends == {"default": ["http://g999:50000"]}


@patch("birdbrain.lifecycle.subprocess.run")
def test_submit_job(mock_subprocess_run, lifecycle_manager):
    """Test job submission."""
    # Mock sbatch response
    mock_subprocess_run.return_value = MagicMock(
        stdout=b"67890\n",
        returncode=0,
    )

    job_id = lifecycle_manager._submit_job()

    assert job_id == "67890"
    mock_subprocess_run.assert_called_once()

    # Verify sbatch was called with correct args
    call_args = mock_subprocess_run.call_args
    assert call_args[0][0] == ["sbatch", "--parsable"]


@patch("birdbrain.lifecycle.subprocess.run")
def test_get_job_info(mock_subprocess_run, lifecycle_manager):
    """Test querying job info from squeue."""
    # Mock squeue response
    mock_subprocess_run.return_value = MagicMock(
        stdout="RUNNING gpu-node-01\n",
        returncode=0,
    )

    info = lifecycle_manager._get_job_info("12345")

    assert info is not None
    assert info["state"] == "RUNNING"
    assert info["node"] == "gpu-node-01"


@patch("birdbrain.lifecycle.subprocess.run")
def test_get_job_info_pending(mock_subprocess_run, lifecycle_manager):
    """Test job info for pending job (no node allocated yet)."""
    mock_subprocess_run.return_value = MagicMock(
        stdout="PENDING (null)\n",
        returncode=0,
    )

    info = lifecycle_manager._get_job_info("12345")

    assert info is not None
    assert info["state"] == "PENDING"
    assert info["node"] is None


@patch("birdbrain.lifecycle.subprocess.run")
def test_get_job_info_not_found(mock_subprocess_run, lifecycle_manager):
    """Test job info when job no longer in queue."""
    mock_subprocess_run.return_value = MagicMock(
        stdout="",
        returncode=0,
    )

    info = lifecycle_manager._get_job_info("12345")

    assert info is None


def _colocated_manager(gateway, tmp_path, **slurm):
    config = Config(
        backends=[
            {
                "name": "laguna-xs",
                "model": {"name": "/m/laguna-xs"},
                "slurm": {"gpus_per_job": 1, "port": 50001, **slurm},
                "secondary_models": [
                    {"model_name": "/m/nomic", "backend_name": "nomic-embed", "aliases": ["embed"]}
                ],
            }
        ]
    )
    config.paths.state_file = str(tmp_path / "state.json")
    return LifecycleManager(config.backends[0], config.paths, "gw-host", 8765, gateway)


def test_instance_plan_splits_gpus_between_replicas(mock_gateway, tmp_path):
    """Replicas get disjoint GPU slices; a single replica inherits the whole
    allocation (so TP>1 keeps every GPU); secondaries follow, pinned."""
    manager = _colocated_manager(mock_gateway, tmp_path, gpus_per_job=4, replicas=2)
    plan = manager._build_instance_plan()
    assert [(i.port, i.backend_name, i.gpu_indices, i.secondary) for i in plan] == [
        (50001, "laguna-xs", [0, 1], False),
        (50002, "laguna-xs", [2, 3], False),
        (50003, "nomic-embed", [0], True),
    ]

    single = _colocated_manager(mock_gateway, tmp_path, gpus_per_job=2)
    assert single._build_instance_plan()[0].gpu_indices is None


@pytest.mark.asyncio
async def test_job_not_promoted_until_secondary_registers(mock_gateway, tmp_path):
    """Promotion waits for every planned backend, not just the primary."""
    registry = {"laguna-xs": ["http://g043:50001"]}
    mock_gateway.get_backend_urls_by_job = lambda name, job_id: registry.get(name, [])
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.running = True
    job = manager._record_pending_job("77")

    async def register_sidecar(_seconds):
        assert manager.current_job is None, "promoted before the secondary registered"
        registry["nomic-embed"] = ["http://g043:50002"]

    with patch.object(manager, "_get_job_info", return_value={"state": "RUNNING", "node": "g043"}), \
            patch("birdbrain.lifecycle.asyncio.sleep", side_effect=register_sidecar):
        await manager._wait_for_job_ready(job)

    assert manager.current_job is job
    assert job.registered_backends == {
        "laguna-xs": ["http://g043:50001"],
        "nomic-embed": ["http://g043:50002"],
    }


@pytest.mark.asyncio
async def test_cold_start_records_pending_job(mock_gateway, tmp_path):
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.running = True

    async def promote(job):
        manager.current_job = job

    with patch.object(manager, "_submit_job", return_value="88"), \
            patch.object(manager, "_wait_for_job_ready", side_effect=promote):
        await manager._cold_start()

    reloaded = _colocated_manager(mock_gateway, tmp_path)
    reloaded._load_state()
    assert reloaded.pending_job.job_id == "88"
    assert reloaded.pending_job.expected_ports == [50001, 50002]


def test_restart_delay_doubles_to_cap_and_resets_when_healthy(mock_gateway, tmp_path):
    manager = _colocated_manager(mock_gateway, tmp_path)
    delays = []
    for failures in range(8):
        manager._consecutive_failures = failures
        delays.append(manager._restart_delay())
    assert delays == [0, 60, 120, 240, 480, 960, 1800, 1800]

    manager.record_healthy()
    assert manager._restart_delay() == 0


@pytest.mark.asyncio
async def test_cold_start_backs_off_and_resubmits_after_failed_job(mock_gateway, tmp_path):
    """A replacement that never becomes current is followed by another one,
    only after the backoff delay; prior failures lengthen the first wait."""
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.running = True
    manager._restart_in_progress = True
    manager._consecutive_failures = 1  # e.g. the health monitor's restart

    async def first_fails_second_registers(job):
        if job.job_id == "2":
            manager.current_job = job

    sleep = AsyncMock()
    with patch.object(manager, "_submit_job", side_effect=["1", "2"]), \
            patch.object(manager, "_wait_for_job_ready", side_effect=first_fails_second_registers), \
            patch("birdbrain.lifecycle.asyncio.sleep", sleep):
        await manager._cold_start()

    assert [c.args[0] for c in sleep.call_args_list] == [60, 120]
    assert manager.current_job.job_id == "2"
    assert manager._restart_in_progress is False


@pytest.mark.asyncio
async def test_cold_start_retries_after_sbatch_failure(mock_gateway, tmp_path):
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.running = True

    async def promote(job):
        manager.current_job = job

    sbatch_error = subprocess.CalledProcessError(1, "sbatch", stderr=b"Unable to contact slurm controller")
    sleep = AsyncMock()
    with patch.object(manager, "_submit_job", side_effect=[sbatch_error, "9"]), \
            patch.object(manager, "_wait_for_job_ready", side_effect=promote), \
            patch("birdbrain.lifecycle.asyncio.sleep", sleep):
        await manager._cold_start()

    assert [c.args[0] for c in sleep.call_args_list] == [60]
    assert manager.current_job.job_id == "9"


@pytest.mark.asyncio
@patch("birdbrain.lifecycle.subprocess.run")
async def test_registration_timeout_cancels_running_job(mock_run, mock_gateway, tmp_path):
    """A job that reaches RUNNING but never registers is scancelled, not left
    holding its GPU and ports until wall time."""
    mock_gateway.get_backend_urls_by_job = lambda name, job_id: []
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.running = True
    job = manager._record_pending_job("77")

    t0 = datetime(2026, 10, 12, 8, 19)
    fake_datetime = MagicMock()
    fake_datetime.utcnow.side_effect = [t0, t0 + timedelta(minutes=21)]
    with patch.object(manager, "_get_job_info", return_value={"state": "RUNNING", "node": "g042"}), \
            patch("birdbrain.lifecycle.datetime", fake_datetime):
        await manager._wait_for_job_ready(job)

    assert job.status == "FAILED"
    assert manager.current_job is None
    mock_run.assert_any_call(["scancel", "77"], capture_output=True)


def _job(job_id: str, status: str = "RUNNING") -> JobState:
    return JobState(
        job_id=job_id,
        node="g042",
        status=status,
        submitted_at=datetime.utcnow().isoformat(),
        expires_at=(datetime.utcnow() + timedelta(hours=40)).isoformat(),
    )


@pytest.mark.asyncio
@patch("birdbrain.lifecycle.subprocess.run")
async def test_unhealthy_restart_cancels_current_and_pending_jobs(mock_run, mock_gateway, tmp_path):
    """The replacement supersedes a queued rotation job too; dropping it from
    state without scancel left it to start untracked on the same node."""
    mock_gateway.clear_backend = AsyncMock()
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.current_job = _job("100")
    manager.pending_job = _job("101", status="PENDING")

    await manager.clear_backend_and_restart()

    cancelled = [c.args[0] for c in mock_run.call_args_list if c.args[0][0] == "scancel"]
    assert cancelled == [["scancel", "100"], ["scancel", "101"]]
    assert manager.pending_job is None
    assert manager._consecutive_failures == 1


@pytest.mark.asyncio
@patch("birdbrain.lifecycle.subprocess.run")
async def test_start_cancels_pending_job_from_previous_run(mock_run, mock_gateway, tmp_path):
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.current_job = _job("200")
    manager.pending_job = _job("201", status="PENDING")
    manager._save_state()

    restarted = _colocated_manager(mock_gateway, tmp_path)
    with patch.object(restarted, "_get_job_info", return_value={"state": "RUNNING", "node": "g042"}):
        await restarted.start()
        await restarted.stop()

    mock_run.assert_any_call(["scancel", "201"], capture_output=True)
    reloaded = _colocated_manager(mock_gateway, tmp_path)
    reloaded._load_state()
    assert reloaded.current_job.job_id == "200"
    assert reloaded.pending_job is None


@pytest.mark.asyncio
async def test_start_rehydrates_every_registered_backend(mock_gateway, tmp_path):
    """A gateway restart re-registers the primary and the secondary from state."""
    manager = _colocated_manager(mock_gateway, tmp_path)
    manager.current_job = JobState(
        job_id="55",
        node="g043",
        status="RUNNING",
        submitted_at=datetime.utcnow().isoformat(),
        expires_at=(datetime.utcnow() + timedelta(hours=40)).isoformat(),
        expected_ports=[50001, 50002],
        registered_backends={
            "laguna-xs": ["http://g043:50001"],
            "nomic-embed": ["http://g043:50002"],
        },
    )
    manager._save_state()

    restarted = _colocated_manager(mock_gateway, tmp_path)
    with patch.object(restarted, "_get_job_info", return_value={"state": "RUNNING", "node": "g043"}):
        await restarted.start()
    await restarted.stop()

    calls = {(c.args[0], c.args[1], c.kwargs["model_name"]) for c in mock_gateway.set_backend.call_args_list}
    assert calls == {
        ("laguna-xs", "http://g043:50001", "/m/laguna-xs"),
        ("nomic-embed", "http://g043:50002", "/m/nomic"),
    }
