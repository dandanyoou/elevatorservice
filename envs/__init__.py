"""Custom Gymnasium environments for elevator scheduling."""
from __future__ import annotations

from gymnasium.envs.registration import register

from envs.config import BuildingConfig, ElevatorEnvConfig, TrafficConfig
from envs.elevator_env import ElevatorEnv

__all__ = [
    "BuildingConfig",
    "ElevatorEnv",
    "ElevatorEnvConfig",
    "TrafficConfig",
]

register(
    id="ElevatorRL-v0",
    entry_point="envs.elevator_env:ElevatorEnv",
    max_episode_steps=3600,
)
