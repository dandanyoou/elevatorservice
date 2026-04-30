"""Poisson passenger arrival generator with time-varying OD matrices.

Patterns are deterministic given the seed. Returns a list of new passenger
(origin, destination) pairs for the current simulation step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from envs.config import TrafficConfig


@dataclass
class Passenger:
    pid: int
    origin: int
    destination: int
    spawn_time: float
    pickup_time: float | None = None
    arrival_time: float | None = None

    @property
    def direction(self) -> int:
        return 1 if self.destination > self.origin else -1


def od_matrix(num_floors: int, pattern: str) -> NDArray[np.float64]:
    """Return an unnormalized OD matrix; the diagonal is zero."""
    f = num_floors
    od = np.zeros((f, f), dtype=np.float64)

    if pattern == "uniform":
        od = np.ones((f, f), dtype=np.float64)
    elif pattern == "morning":
        # heavy lobby->upper, light upper->upper
        od[0, 1:] = 6.0
        for i in range(1, f):
            for j in range(f):
                if i != j:
                    od[i, j] = 0.3
    elif pattern == "lunch":
        # mid floors source/sink heavily
        mid = f // 2
        for i in range(f):
            for j in range(f):
                if i == j:
                    continue
                weight = 1.0
                if i in (0, mid) or j in (0, mid):
                    weight = 4.0
                od[i, j] = weight
    elif pattern == "evening":
        od[1:, 0] = 6.0
        for i in range(f):
            for j in range(1, f):
                if i != j:
                    od[i, j] += 0.3
    else:
        raise ValueError(f"unknown traffic pattern: {pattern!r}")

    np.fill_diagonal(od, 0.0)
    total = od.sum()
    if total <= 0:
        raise ValueError("OD matrix sums to zero")
    return od / total


class PoissonTrafficGenerator:
    """Generate passenger arrivals at each simulated time step.

    Arrivals follow a homogeneous Poisson process with rate λ
    (events / second). Each event is then assigned (origin, destination)
    by sampling from the OD distribution.
    """

    def __init__(self, num_floors: int, cfg: TrafficConfig, dt_seconds: float) -> None:
        self.num_floors = num_floors
        self.cfg = cfg
        self.dt_seconds = dt_seconds
        self.lambda_per_step = cfg.base_rate_per_minute / 60.0 * dt_seconds
        self.od = od_matrix(num_floors, cfg.pattern)
        self._od_flat = self.od.flatten()
        self._next_pid = 0
        self._rng = np.random.default_rng(cfg.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self.cfg.seed
        self._rng = np.random.default_rng(seed)
        self._next_pid = 0

    def step(self, sim_time_s: float) -> list[Passenger]:
        n = int(self._rng.poisson(self.lambda_per_step))
        if n == 0:
            return []
        choices = self._rng.choice(self._od_flat.size, size=n, p=self._od_flat)
        out: list[Passenger] = []
        for c in choices:
            origin = int(c) // self.num_floors
            destination = int(c) % self.num_floors
            if origin == destination:  # safety; shouldn't happen due to zero diagonal
                continue
            p = Passenger(
                pid=self._next_pid,
                origin=origin,
                destination=destination,
                spawn_time=sim_time_s,
            )
            self._next_pid += 1
            out.append(p)
        return out
