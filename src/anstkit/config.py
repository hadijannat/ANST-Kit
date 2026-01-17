"""Centralized configuration for ANST-Kit.

This module consolidates all configurable parameters in one place, supporting
environment variable overrides via pydantic-settings. All modules should import
settings from here rather than using hardcoded values.

Usage:
    from anstkit.config import get_settings

    settings = get_settings()
    weights_path = settings.pinn_weights_path
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Package root for relative path resolution
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent.parent


class PhysicsConfig(BaseSettings):
    """Physics simulation parameters for the tank system.

    These parameters define the tank dynamics:
        dH/dt = (k_in * u_pump - k_out * u_valve * sqrt(H)) / area
    """

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_PHYSICS_")

    k_in: float = Field(default=1.0, description="Inlet flow coefficient")
    k_out: float = Field(default=0.6, description="Outlet flow coefficient")
    area: float = Field(default=1.0, description="Tank cross-sectional area")
    eps: float = Field(default=1e-6, description="Numerical stability epsilon")

    # PINN inference parameters
    horizon: float = Field(default=5.0, description="Prediction horizon (time units)")
    n_eval: int = Field(default=64, description="Number of evaluation points")
    h_max: float = Field(default=1.0, description="Maximum safe tank level")
    residual_threshold: float = Field(default=5e-3, description="Max acceptable physics residual")
    uncertainty_threshold: Optional[float] = Field(
        default=0.05, description="Max acceptable ensemble uncertainty"
    )


class PINNConfig(BaseSettings):
    """PINN training and architecture configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_PINN_")

    hidden_dim: int = Field(default=64, description="Hidden layer dimension")
    depth: int = Field(default=3, description="Number of hidden layers")
    training_steps: int = Field(default=1500, description="Default training steps")
    learning_rate: float = Field(default=3e-4, description="Adam learning rate")
    n_ensemble: int = Field(default=5, description="Number of ensemble members")


class PlantConfig(BaseSettings):
    """Demo plant topology configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_PLANT_")

    tank_id: str = Field(default="T1", description="Tank node ID")
    pump_id: str = Field(default="P1", description="Pump node ID")
    inlet_valve_id: str = Field(default="V1", description="Inlet valve node ID")
    outlet_valve_id: str = Field(default="V2", description="Outlet valve node ID")


class AgentConfig(BaseSettings):
    """Demo agent behavior configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_AGENT_")

    p_hallucinate_id: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Probability of hallucinating invalid target IDs"
    )
    p_out_of_range: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Probability of proposing out-of-range values"
    )


class OrchestratorConfig(BaseSettings):
    """Orchestrator behavior configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_ORCHESTRATOR_")

    max_iterations: int = Field(default=3, ge=1, description="Max propose-verify iterations")


class PolicyGateConfig(BaseSettings):
    """Policy gate configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_POLICY_")

    max_delta: float = Field(default=0.5, ge=0.0, le=1.0, description="Max allowed change per step")
    enforce_action_allowlist: bool = Field(default=True, description="Enforce action type allowlist")
    enforce_target_allowlist: bool = Field(default=False, description="Enforce target ID allowlist")
    enforce_value_bounds: bool = Field(default=True, description="Enforce value bounds")


class APIConfig(BaseSettings):
    """API service configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_API_")

    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_key: Optional[str] = Field(default=None, description="API key for authentication (None = disabled)")
    require_auth: bool = Field(default=False, description="Require API key authentication")


class AuditConfig(BaseSettings):
    """Audit store configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_AUDIT_")

    db_path: str = Field(default=":memory:", description="SQLite DB path (':memory:' for in-memory)")
    use_thread_local: bool = Field(
        default=True, description="Use thread-local connections for safety"
    )


class MLFlowConfig(BaseSettings):
    """MLflow tracking configuration."""

    model_config = SettingsConfigDict(env_prefix="ANSTKIT_MLFLOW_")

    tracking_uri: Optional[str] = Field(
        default=None, description="MLflow tracking URI (None = disabled)"
    )
    experiment_name: str = Field(default="anstkit-experiments", description="Experiment name")


class Settings(BaseSettings):
    """Root settings class aggregating all configuration.

    Environment variables:
        ANSTKIT_SEED: Global random seed
        ANSTKIT_PINN_WEIGHTS_PATH: Path to PINN model weights
        ANSTKIT_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)

    Nested config environment variables use prefixes:
        ANSTKIT_PHYSICS_* - Physics parameters
        ANSTKIT_PINN_* - PINN config
        ANSTKIT_PLANT_* - Plant topology
        ANSTKIT_AGENT_* - Agent behavior
        ANSTKIT_ORCHESTRATOR_* - Orchestrator settings
        ANSTKIT_POLICY_* - Policy gate
        ANSTKIT_API_* - API service
        ANSTKIT_AUDIT_* - Audit store
        ANSTKIT_MLFLOW_* - MLflow tracking
    """

    model_config = SettingsConfigDict(
        env_prefix="ANSTKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Global settings
    seed: int = Field(default=42, description="Global random seed for reproducibility")
    log_level: str = Field(default="INFO", description="Logging level")
    pinn_weights_path: Path = Field(
        default=_PROJECT_ROOT / "models" / "tank_pinn.pt",
        description="Path to PINN model weights",
    )

    # Nested configurations
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    pinn: PINNConfig = Field(default_factory=PINNConfig)
    plant: PlantConfig = Field(default_factory=PlantConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    policy: PolicyGateConfig = Field(default_factory=PolicyGateConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    mlflow: MLFlowConfig = Field(default_factory=MLFlowConfig)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper

    @field_validator("pinn_weights_path", mode="before")
    @classmethod
    def resolve_weights_path(cls, v):
        """Convert string path to Path object."""
        if isinstance(v, str):
            return Path(v)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    This function is cached to ensure singleton behavior. Settings are
    loaded once from environment variables and .env file.

    Returns:
        Settings instance with all configuration.
    """
    return Settings()


def configure_logging(settings: Optional[Settings] = None) -> None:
    """Configure logging based on settings.

    Args:
        settings: Settings instance, or None to use get_settings().
    """
    settings = settings or get_settings()
    level = getattr(logging, settings.log_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Convenience function to get a specific config value
def get_default_weights_path() -> Path:
    """Get the default PINN weights path."""
    return get_settings().pinn_weights_path
