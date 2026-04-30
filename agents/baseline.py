"""Rule-based baselines for the elevator env.

These act as comparison points for any learned policy. Each baseline returns
a ``MultiDiscrete([3] * num_elevators)`` action where 0=DOWN, 1=HOLD, 2=UP.

Baselines peek at the underlying :class:`Building` rather than only consuming
the observation vector, because their decision logic is naturally expressed
in elevator coordinates (floors, directions, calls). Learned policies get the
flat observation; baselines get the structured truth.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from envs.building import ACTION_DOWN, ACTION_HOLD, ACTION_UP, Building, ElevatorState
from envs.elevator_env import ElevatorEnv


class BasePolicy(ABC):
    def __init__(self, env: ElevatorEnv) -> None:
        self.env = env

    def reset(self) -> None:  # noqa: B027 — default no-op is intentional
        return None

    @abstractmethod
    def act(self) -> np.ndarray: ...


class RandomPolicy(BasePolicy):
    """Sanity-check baseline: random direction per step."""

    def __init__(self, env: ElevatorEnv, seed: int = 0) -> None:
        super().__init__(env)
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self) -> np.ndarray:
        n = self.env.cfg.building.num_elevators
        return self._rng.integers(low=0, high=3, size=n)


class ScanPolicy(BasePolicy):
    """SCAN ('elevator algorithm') baseline.

    Each elevator continues in its current direction while any demand
    (hall calls or internal calls) remains ahead. When demand ahead is
    exhausted, it reverses if there is demand behind, otherwise idles.
    Initial direction at rest defaults to UP.
    """

    def act(self) -> np.ndarray:
        b = self.env.building
        n = b.num_elevators
        actions = np.full(n, ACTION_HOLD, dtype=np.int64)

        for i, e in enumerate(b.elevators):
            actions[i] = self._action_for(e, b)
        return actions

    def _action_for(self, e: ElevatorState, b: Building) -> int:
        if e.is_servicing:
            return ACTION_HOLD

        # mid-flight: continue
        if e.direction > 0:
            return ACTION_UP
        if e.direction < 0:
            return ACTION_DOWN

        # at rest: choose direction based on demand
        above = self._has_demand_above(e, b)
        below = self._has_demand_below(e, b)

        # prefer continuing in last direction
        if e.last_direction >= 0 and above:
            return ACTION_UP
        if e.last_direction <= 0 and below:
            return ACTION_DOWN
        if above:
            return ACTION_UP
        if below:
            return ACTION_DOWN
        return ACTION_HOLD

    @staticmethod
    def _has_demand_above(e: ElevatorState, b: Building) -> bool:
        f = e.floor
        if any(p.destination > f for p in e.passengers):
            return True
        if b.hall_up[f + 1 :].any() or b.hall_down[f + 1 :].any():
            return True
        return False

    @staticmethod
    def _has_demand_below(e: ElevatorState, b: Building) -> bool:
        f = e.floor
        if any(p.destination < f for p in e.passengers):
            return True
        if b.hall_up[:f].any() or b.hall_down[:f].any():
            return True
        return False


class NearestCarPolicy(BasePolicy):
    """Nearest-Car dispatch.

    When an elevator is at rest, send it to the closest unanswered hall call.
    When committed to a direction, behave like SCAN. Internal calls always
    take precedence (the elevator must finish its current load).

    Assignment is recomputed every step (no sticky assignment table).
    """

    def act(self) -> np.ndarray:
        b = self.env.building
        n = b.num_elevators
        actions = np.full(n, ACTION_HOLD, dtype=np.int64)

        # Pass 1: continue motion / hold during service.
        for i, e in enumerate(b.elevators):
            if e.is_servicing:
                actions[i] = ACTION_HOLD
                continue
            if e.direction > 0:
                actions[i] = ACTION_UP
                continue
            if e.direction < 0:
                actions[i] = ACTION_DOWN
                continue

        # Pass 2: resting elevators with passengers head toward nearest internal call.
        for i, e in enumerate(b.elevators):
            if e.direction != 0 or e.is_servicing or not e.passengers:
                continue
            target = min((p.destination for p in e.passengers), key=lambda d: abs(d - e.floor))
            if target > e.floor:
                actions[i] = ACTION_UP
            elif target < e.floor:
                actions[i] = ACTION_DOWN

        # Pass 3: greedy assignment of pending hall calls to remaining idle elevators.
        pending: list[tuple[int, int]] = []
        for f in range(b.num_floors):
            if b.hall_up[f]:
                pending.append((f, +1))
            if b.hall_down[f]:
                pending.append((f, -1))

        idle = [
            i
            for i, e in enumerate(b.elevators)
            if not e.is_servicing
            and e.direction == 0
            and not e.passengers
            and actions[i] == ACTION_HOLD
        ]
        for call in pending:
            if not idle:
                break
            best = min(idle, key=lambda i: abs(b.elevators[i].floor - call[0]))
            target_floor, _ = call
            e = b.elevators[best]
            if target_floor > e.floor:
                actions[best] = ACTION_UP
            elif target_floor < e.floor:
                actions[best] = ACTION_DOWN
            # at-floor case stays HOLD; env will auto-service same-direction call
            idle.remove(best)

        return actions
