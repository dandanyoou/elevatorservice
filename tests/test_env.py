"""Unit tests for the elevator environment.

These cover three things the user spec calls out explicitly:
    1. determinism — same seed ⇒ identical trajectories
    2. reward sign — wait/in-car/energy components are negative; completion is positive
    3. action space integrity — actions are MultiDiscrete([3]*N) and bounds are honored
"""
from __future__ import annotations

import numpy as np
import pytest

import envs  # noqa: F401 — registers ElevatorRL-v0
from agents.baseline import NearestCarPolicy, ScanPolicy
from envs.config import (
    BuildingConfig,
    ElevatorEnvConfig,
    RewardWeights,
    TrafficConfig,
)
from envs.elevator_env import ElevatorEnv


def _make_env(
    num_floors: int = 8,
    num_elevators: int = 2,
    horizon: int = 200,
    seed: int = 0,
    pattern: str = "uniform",
) -> ElevatorEnv:
    cfg = ElevatorEnvConfig(
        building=BuildingConfig(num_floors=num_floors, num_elevators=num_elevators, capacity=8),
        traffic=TrafficConfig(pattern=pattern, base_rate_per_minute=20.0, seed=seed),
        horizon_steps=horizon,
    )
    return ElevatorEnv(cfg)


def test_action_space_shape() -> None:
    env = _make_env()
    assert env.action_space.shape == (2,)
    assert list(env.action_space.nvec) == [3, 3]


def test_observation_space_shape() -> None:
    env = _make_env(num_floors=10, num_elevators=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_determinism_under_fixed_seed() -> None:
    """Two envs with identical config + seed produce identical trajectories."""
    env_a = _make_env(seed=42, horizon=100)
    env_b = _make_env(seed=42, horizon=100)

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)

    rewards_a, rewards_b = [], []
    for _ in range(100):
        action = np.array([2, 0])  # arbitrary fixed action
        oa, ra, term_a, trunc_a, _ = env_a.step(action)
        ob, rb, term_b, trunc_b, _ = env_b.step(action)
        rewards_a.append(ra)
        rewards_b.append(rb)
        np.testing.assert_array_equal(oa, ob)
        assert term_a == term_b and trunc_a == trunc_b
    assert rewards_a == rewards_b


def test_truncation_at_horizon() -> None:
    env = _make_env(horizon=5)
    env.reset(seed=0)
    truncated = False
    for _ in range(5):
        _, _, _, truncated, _ = env.step(env.action_space.sample())
    assert truncated, "expected truncation at horizon"


def test_reward_sign_components() -> None:
    """A pure-cost step (no completions) must yield non-positive reward."""
    env = _make_env(horizon=200, pattern="morning")
    env.reset(seed=1)
    # take many HOLD actions so nothing ever gets serviced beyond initial loads
    hold = np.array([1, 1])
    saw_pure_cost_step = False
    for _ in range(50):
        _, r, _, _, info = env.step(hold)
        if info["arrivals"] == 0:
            assert r <= 0.0, f"pure-cost step but reward {r} > 0"
            saw_pure_cost_step = True
    assert saw_pure_cost_step


def test_completion_bonus_positive_contribution() -> None:
    """The completion bonus is positive and equals weight * arrivals."""
    cfg = ElevatorEnvConfig(
        building=BuildingConfig(num_floors=4, num_elevators=1, capacity=4),
        traffic=TrafficConfig(pattern="uniform", base_rate_per_minute=60.0, seed=7),
        reward=RewardWeights(
            waiting=0.0, in_car=0.0, energy=0.0, direction_change=0.0, completion_bonus=10.0
        ),
        horizon_steps=400,
    )
    env = ElevatorEnv(cfg)
    env.reset(seed=7)
    policy = ScanPolicy(env)
    saw_completion = False
    for _ in range(400):
        action = policy.act()
        _, r, _, trunc, info = env.step(action)
        if info["arrivals"] > 0:
            assert r == pytest.approx(10.0 * info["arrivals"])
            saw_completion = True
            break
        if trunc:
            break
    assert saw_completion, "expected at least one completion under SCAN"


def test_action_bounds_enforced() -> None:
    env = _make_env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.array([2, 1, 0]))  # too many actions


def test_scan_completes_passengers() -> None:
    env = _make_env(num_floors=6, num_elevators=2, horizon=600, seed=3)
    env.reset(seed=3)
    policy = ScanPolicy(env)
    for _ in range(600):
        env.step(policy.act())
    metrics = env.metrics()
    assert metrics["completed"] >= 5, f"SCAN failed to deliver any passengers: {metrics}"
    assert metrics["avg_wait_s"] >= 0.0


def test_nearest_car_completes_passengers() -> None:
    env = _make_env(num_floors=6, num_elevators=2, horizon=600, seed=3)
    env.reset(seed=3)
    policy = NearestCarPolicy(env)
    for _ in range(600):
        env.step(policy.act())
    metrics = env.metrics()
    assert metrics["completed"] >= 5, f"Nearest-Car failed to deliver passengers: {metrics}"


def test_no_passengers_above_capacity() -> None:
    env = _make_env(num_floors=8, num_elevators=1, horizon=600, seed=11)
    env.reset(seed=11)
    policy = ScanPolicy(env)
    cap = env.cfg.building.capacity
    for _ in range(600):
        env.step(policy.act())
        for e in env.building.elevators:
            assert e.num_passengers <= cap


def test_render_ascii_runs() -> None:
    env = _make_env(num_floors=5, num_elevators=2, horizon=10)
    env.reset(seed=0)
    out = env.render_ascii()
    assert isinstance(out, str)
    assert len(out) > 0
