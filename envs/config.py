"""Dataclass configurations for the elevator environment.

Designed to play well with Hydra: every parameter is a plain dataclass field
with a default, so Hydra can override via CLI or YAML without surprises.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BuildingConfig:
    num_floors: int = 10
    num_elevators: int = 3
    capacity: int = 12
    floor_height_m: float = 3.5
    speed_mps: float = 2.0
    door_open_seconds: float = 2.0
    door_close_seconds: float = 2.0
    boarding_seconds_per_passenger: float = 1.0
    accel_penalty_per_change: float = 1.0


@dataclass(frozen=True)
class TrafficConfig:
    """Time-varying Poisson arrival generator.

    Three canonical patterns; pick one via `pattern`:
      - "uniform":  uniform pairwise OD, low rate
      - "morning":  most arrivals from lobby (floor 0) up
      - "lunch":    bidirectional from middle floors
      - "evening":  most arrivals to lobby
    """

    pattern: str = "uniform"
    base_rate_per_minute: float = 8.0
    seed: int = 0


@dataclass
class RewardWeights:
    waiting: float = 0.01
    in_car: float = 0.005
    energy: float = 0.001
    direction_change: float = 0.1
    completion_bonus: float = 1.0


@dataclass(frozen=True)
class ElevatorEnvConfig:
    building: BuildingConfig = field(default_factory=BuildingConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    dt_seconds: float = 1.0
    horizon_steps: int = 3600
    max_passengers: int = 200
