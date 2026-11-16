"""Tests for config module."""

import tempfile

import pytest

import datetime as dt

from birdbrain.config import BackendPoolConfig, Config, ScheduleConfig, load_config


def test_config_defaults():
    """Test default configuration."""
    config = Config()

    assert config.gateway.host == "127.0.0.1"
    assert config.gateway.port == 8765

    # No [[backends]] and default legacy slurm/model -> migrated to "default"
    assert len(config.backends) == 1
    default_backend = config.backends[0]
    assert default_backend.name == "default"
    assert default_backend.slurm.partition == "gpu"
    assert default_backend.slurm.gpus_per_job == 2
    assert default_backend.slurm.wall_time == "23:00:00"
    assert default_backend.slurm.lead_time_minutes == 90
    assert default_backend.model.name == "meta-llama/Llama-3.1-8B"

    assert config.paths.api_keys_file == "~/.birdbrain/api_keys.json"


def test_config_from_dict(sample_config_data):
    """Test config creation from dict (legacy single-model shape)."""
    config = Config(**sample_config_data)

    assert config.gateway.port == 8765
    assert len(config.backends) == 1
    assert config.backends[0].name == "default"
    assert config.backends[0].slurm.partition == "gpu"
    assert config.backends[0].model.name == "meta-llama/Llama-3.1-8B"


def test_config_migration_from_legacy_shape(sample_config_data):
    """Test that old-style [slurm]/[model] TOML (no [[backends]]) migrates
    into a single "default" backend pool, so existing deployments don't
    need to change their config file."""
    config = Config(**sample_config_data)

    assert config.backends == [
        BackendPoolConfig(
            name="default",
            model=config.model,
            slurm=config.slurm,
        )
    ]


def test_multi_backend_config(sample_multi_backend_config_data):
    """Test explicit [[backends]] config with aliases."""
    config = Config(**sample_multi_backend_config_data)

    assert len(config.backends) == 2
    names = {b.name for b in config.backends}
    assert names == {"llama-3.3-70b", "small-model-a"}

    llama = next(b for b in config.backends if b.name == "llama-3.3-70b")
    assert llama.aliases == ["llama", "default"]
    assert llama.model.name == "meta-llama/Llama-3.3-70B-Instruct"

    small = next(b for b in config.backends if b.name == "small-model-a")
    assert small.aliases == ["small", "fast"]


def test_duplicate_backend_names_rejected(sample_multi_backend_config_data):
    """Backend pool names must be unique."""
    data = sample_multi_backend_config_data
    data["backends"][1]["name"] = data["backends"][0]["name"]

    with pytest.raises(ValueError):
        Config(**data)


def test_duplicate_aliases_rejected(sample_multi_backend_config_data):
    """Aliases must be unique across all backends."""
    data = sample_multi_backend_config_data
    data["backends"][1]["aliases"] = ["small", "llama"]

    with pytest.raises(ValueError):
        Config(**data)


def test_alias_colliding_with_backend_name_rejected(sample_multi_backend_config_data):
    """Aliases must not collide with any backend's canonical name or model name."""
    data = sample_multi_backend_config_data
    data["backends"][1]["aliases"] = ["small", "llama-3.3-70b"]

    with pytest.raises(ValueError):
        Config(**data)


def test_replicas_must_divide_gpus(sample_multi_backend_config_data):
    """Each replica gets an equal GPU slice, so gpus_per_job % replicas must be 0."""
    data = sample_multi_backend_config_data
    data["backends"][0]["slurm"].update(gpus_per_job=2, replicas=3)

    with pytest.raises(ValueError, match="divisible"):
        Config(**data)


def test_secondary_gpu_outside_allocation_rejected(sample_multi_backend_config_data):
    """A secondary pinned to a GPU index the job doesn't own is a config error."""
    data = sample_multi_backend_config_data
    data["backends"][1]["secondary_models"] = [
        {"model_name": "nomic-ai/nomic-embed-text-v1.5", "backend_name": "embed-pool", "gpu_device": 1}
    ]

    with pytest.raises(ValueError, match="outside"):
        Config(**data)


def test_secondary_ports_overlapping_another_pool_rejected(sample_multi_backend_config_data):
    """llama owns 50000; adding a replica there would take 50001, small-model-a's port."""
    data = sample_multi_backend_config_data
    data["backends"][0]["slurm"]["replicas"] = 2

    with pytest.raises(ValueError, match="overlap"):
        Config(**data)


def test_secondary_backend_name_colliding_with_pool_rejected(sample_multi_backend_config_data):
    """Secondary backend_names share the gateway's registry with pool names."""
    data = sample_multi_backend_config_data
    data["backends"][1]["slurm"]["port"] = 50005
    data["backends"][1]["secondary_models"] = [
        {"model_name": "nomic-ai/nomic-embed-text-v1.5", "backend_name": "llama-3.3-70b"}
    ]

    with pytest.raises(ValueError, match="unique"):
        Config(**data)


def test_wall_time_validation():
    """Test wall_time format validation."""
    from birdbrain.config import SlurmConfig

    # Valid formats
    SlurmConfig(wall_time="23:00:00")
    SlurmConfig(wall_time="1-12:30:00")

    # Invalid formats
    with pytest.raises(ValueError):
        SlurmConfig(wall_time="invalid")

    with pytest.raises(ValueError):
        SlurmConfig(wall_time="23:00")  # Missing seconds


def test_load_config_from_file(sample_config_data, tmp_path):
    """Test loading config from TOML file."""
    import tomli_w

    config_file = tmp_path / "test-config.toml"
    with open(config_file, "wb") as f:
        tomli_w.dump(sample_config_data, f)

    config = load_config(config_file)

    assert config.gateway.port == 8765
    assert config.backends[0].slurm.partition == "gpu"


def test_load_config_default_when_no_file():
    """Test loading default config when no file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temp dir where no config exists
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            config = load_config()
            assert config.gateway.port == 8765  # Default value
        finally:
            os.chdir(old_cwd)

def test_schedule_is_optional_and_default():
    """No schedule means models run around the clock."""
    assert Config().schedule is None


def test_schedule_window_weekday_and_overnight():
    """Within-window returns stop in UTC; excluded day returns None; overnight
    wraps cross midnight into the next calendar day."""
    sched = ScheduleConfig(timezone="America/New_York", start=dt.time(22), stop=dt.time(2))
    # 23:00 NY (03:00 UTC) Tuesday, inside a window that started 22:00 Monday
    now = dt.datetime(2026, 11, 17, 3, 0)
    assert sched.is_active(now)
    assert sched.window_end(now) == dt.datetime(2026, 11, 17, 7, 0)  # 02:00 Tue NY
    # Monday 21:00 NY (02:00 UTC): window started Sunday 22:00, Sunday excluded
    mon = dt.datetime(2026, 11, 16, 2, 0)
    assert not sched.is_active(mon)
    assert sched.window_end(mon) is None


def test_schedule_rejects_equal_start_and_stop():
    with pytest.raises(ValueError):
        ScheduleConfig(start=dt.time(9), stop=dt.time(9))


def test_schedule_rejects_unknown_timezone_and_days():
    with pytest.raises(ValueError):
        ScheduleConfig(timezone="bogus", start=dt.time(9), stop=dt.time(10))
    with pytest.raises(ValueError):
        ScheduleConfig(start=dt.time(9), stop=dt.time(10), days=["funday"])
    with pytest.raises(ValueError):
        ScheduleConfig(start=dt.time(9), stop=dt.time(10), days=[])


def test_schedule_days_subset_enforced():
    """A weekday-only window is inactive on Saturday."""
    sched = ScheduleConfig(
        timezone="America/New_York", start=dt.time(7), stop=dt.time(19),
        days=["mon", "tue", "wed", "thu", "fri"],
    )
    sat = dt.datetime(2026, 11, 14, 16, 0)  # 11:00 NY Saturday
    assert not sched.is_active(sat)
    assert sched.window_end(sat) is None
    fri = dt.datetime(2026, 11, 13, 16, 0)  # 11:00 NY Friday
    assert sched.is_active(fri)
