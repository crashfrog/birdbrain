"""Configuration loading and validation."""

import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class GatewayConfig(BaseModel):
    """Gateway server configuration."""

    host: str = "127.0.0.1"
    port: int = 8765
    # Hostname compute nodes use to reach this gateway for /internal/register
    # callbacks. `host` is the uvicorn bind address and is frequently
    # "0.0.0.0"/"::" (bind all interfaces) — not a routable address a compute
    # node can dial. Leave unset to auto-resolve via socket.gethostname() when
    # `host` is a bind-all address, or reuse `host` otherwise.
    advertise_host: Optional[str] = None


class SlurmConfig(BaseModel):
    """Slurm job configuration."""

    partition: str = "gpu"
    gres: str = "gpu:h100:2"
    gpus_per_job: int = 2
    wall_time: str = "23:00:00"
    lead_time_minutes: int = 90
    port: int = 50000  # Port of the first instance; further instances use port+1, port+2, ...
    memory: Optional[str] = None  # Slurm --mem, e.g. "128GB"; omitted if unset
    cpus_per_task: Optional[int] = None  # Slurm --cpus-per-task; omitted if unset
    # Independent vLLM replicas of the primary model inside one Slurm job; the
    # gateway round-robins across them. The job's GPUs are split evenly, so
    # each replica gets gpus_per_job // replicas GPUs (its tensor-parallel
    # size). With replicas = 1 the primary sees every GPU in the allocation.
    replicas: int = Field(default=1, ge=1)

    @field_validator("wall_time")
    @classmethod
    def validate_wall_time(cls, v: str) -> str:
        """Validate wall time format (HH:MM:SS or D-HH:MM:SS)."""
        parts = v.split("-")
        if len(parts) == 2:
            time_part = parts[1]
        elif len(parts) == 1:
            time_part = parts[0]
        else:
            raise ValueError("Invalid wall_time format")

        time_components = time_part.split(":")
        if len(time_components) != 3:
            raise ValueError("wall_time must be HH:MM:SS or D-HH:MM:SS")
        return v


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = "meta-llama/Llama-3.1-8B"
    vllm_extra_args: str = "--tensor-parallel-size 2 --gpu-memory-utilization 0.95"
    # Environment variables to set inside the vLLM container, keyed by bare
    # name (no SINGULARITYENV_ prefix — the template adds it so the value
    # crosses into Singularity). Use for build-specific kernel-stability
    # toggles, e.g. {"VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER": "0"} to disable an
    # unstable FP8 GEMM path on a given container. Empty by default.
    env: dict[str, str] = Field(default_factory=dict)


class SecondaryModelConfig(BaseModel):
    """A secondary model co-located in a backend pool's Slurm job."""

    model_name: str  # What vLLM receives as --model
    backend_name: str  # Name used in /internal/register
    aliases: list[str] = Field(default_factory=list)
    vllm_extra_args: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    # Index into the job's GPU allocation this model runs on. Shares that GPU
    # with the primary model, so both sides' --gpu-memory-utilization must sum
    # to less than 1.0.
    gpu_device: int = Field(default=0, ge=0)


class BackendPoolConfig(BaseModel):
    """A backend pool: one Slurm job lineage serving a primary model (optionally
    replicated) plus zero or more co-located secondary models.

    Every instance shares the job's lifecycle: rotation, health, restart.
    """

    name: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    aliases: list[str] = Field(default_factory=list)
    secondary_models: list[SecondaryModelConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_gpu_layout(self) -> "BackendPoolConfig":
        gpus = self.slurm.gpus_per_job
        if gpus % self.slurm.replicas != 0:
            raise ValueError(
                f"[{self.name}] gpus_per_job ({gpus}) must be divisible by replicas "
                f"({self.slurm.replicas})"
            )
        for sec in self.secondary_models:
            if sec.gpu_device >= gpus:
                raise ValueError(
                    f"[{self.name}] secondary model {sec.backend_name!r} gpu_device "
                    f"{sec.gpu_device} is outside the job's {gpus} GPU(s)"
                )
        return self

    def instance_ports(self) -> list[int]:
        """Ports of every instance in launch order: primary replicas, then secondaries."""
        count = self.slurm.replicas + len(self.secondary_models)
        return [self.slurm.port + i for i in range(count)]


class PathsConfig(BaseModel):
    """Path configuration."""

    container_path: str = "~/.birdbrain/containers/vllm-openai.sif"
    job_template: str = "birdbrain/templates/vllm_job.sh.j2"
    state_file: str = "~/.birdbrain/state.json"
    log_dir: str = "~/.birdbrain/logs"
    api_keys_file: str = "~/.birdbrain/api_keys.json"


class Config(BaseModel):
    """Root configuration."""

    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    backends: list[BackendPoolConfig] = Field(default_factory=list)

    # Legacy singular fields. Only used to migrate old-style single-model
    # TOML (no [[backends]]) into a single "default" backend pool so
    # existing deployments don't need to change their config file.
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    @model_validator(mode="after")
    def _migrate_legacy_shape(self) -> "Config":
        """Synthesize a single "default" backend pool from legacy [slurm]/[model]
        tables when no [[backends]] entries are configured."""
        if not self.backends:
            self.backends = [
                BackendPoolConfig(name="default", model=self.model, slurm=self.slurm)
            ]
        return self

    @model_validator(mode="after")
    def _validate_backends(self) -> "Config":
        """Backend names, aliases and instance ports must be unique across every
        pool and secondary model; aliases must not shadow a name or model id."""
        names = [b.name for b in self.backends]
        names += [sec.backend_name for b in self.backends for sec in b.secondary_models]
        if len(names) != len(set(names)):
            raise ValueError(
                "Backend names (pool names and secondary_models.backend_name) must be unique"
            )

        aliases = [a for b in self.backends for a in b.aliases]
        aliases += [a for b in self.backends for sec in b.secondary_models for a in sec.aliases]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Aliases must be unique across all backends and secondary models")

        model_names = {b.model.name for b in self.backends}
        model_names |= {sec.model_name for b in self.backends for sec in b.secondary_models}
        collisions = set(aliases) & (set(names) | model_names)
        if collisions:
            raise ValueError(
                f"Aliases must not collide with a backend name or model name: {collisions}"
            )

        # Two pools' jobs can land on the same node, so their instance port
        # ranges must never overlap.
        ports = [p for b in self.backends for p in b.instance_ports()]
        if len(ports) != len(set(ports)):
            raise ValueError(
                "Instance ports overlap between backend pools; each pool uses "
                "slurm.port .. slurm.port + replicas + len(secondary_models) - 1"
            )

        return self



def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from TOML file.

    Search order:
    1. Provided path
    2. ./birdbrain.toml
    3. ~/.birdbrain/birdbrain.toml

    Returns default config if no file found.
    """
    if config_path and config_path.exists():
        paths_to_try = [config_path]
    else:
        paths_to_try = [
            Path("birdbrain.toml"),
            Path.home() / ".birdbrain" / "birdbrain.toml",
        ]

    for path in paths_to_try:
        if path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            return Config(**data)

    # Return default config if no file found
    return Config()
