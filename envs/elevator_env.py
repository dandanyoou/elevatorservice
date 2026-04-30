"""Gymnasium environment for multi-elevator scheduling.

Action space:
    MultiDiscrete([3] * num_elevators)
    per elevator: 0 = DOWN, 1 = HOLD, 2 = UP

Observation space:
    Box(low=-1, high=1, shape=(obs_dim,), dtype=float32)
    obs_dim = num_elevators * (num_floors + 3)        # per-elevator block
            + 2 * num_floors                          # hall calls (up/down)
            + 1                                        # time fraction

Reward (per step), accumulated cost:
    r_t = -α * Σ(active waits)
        - β * Σ(in-car ride seconds)
        - γ * total floor-equivalents moved
        - δ * direction reversals at stops
        + ε * completed arrivals this step
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.building import ACTION_HOLD, Building
from envs.config import ElevatorEnvConfig
from envs.traffic import PoissonTrafficGenerator


class ElevatorEnv(gym.Env[np.ndarray, np.ndarray]):
    """Multi-elevator scheduling MDP."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, cfg: ElevatorEnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ElevatorEnvConfig()
        self.render_mode = render_mode

        self.building = Building(self.cfg.building, self.cfg.dt_seconds)
        self.traffic = PoissonTrafficGenerator(
            num_floors=self.cfg.building.num_floors,
            cfg=self.cfg.traffic,
            dt_seconds=self.cfg.dt_seconds,
        )

        f = self.cfg.building.num_floors
        n = self.cfg.building.num_elevators

        self.action_space: spaces.MultiDiscrete = spaces.MultiDiscrete([3] * n)

        # observation layout (see module docstring)
        self.obs_dim: int = n * (f + 3) + 2 * f + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # episode bookkeeping
        self._step_count: int = 0
        self._sim_time_s: float = 0.0

    # --------------- public API ---------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.traffic.reset(seed=seed)
        else:
            self.traffic.reset()
        self.building.reset()
        self._step_count = 0
        self._sim_time_s = 0.0
        obs = self._observe()
        return obs, self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != self.cfg.building.num_elevators:
            raise ValueError(
                f"action length {action.shape[0]} != num_elevators "
                f"{self.cfg.building.num_elevators}"
            )

        # 1) admit new arrivals at start of this step
        new_passengers = self.traffic.step(self._sim_time_s)
        self.building.admit_arrivals(new_passengers)

        # 2) advance dynamics
        events = self.building.step(action, self._sim_time_s)

        # 3) advance time
        self._sim_time_s += self.cfg.dt_seconds
        self._step_count += 1

        # 4) reward
        w = self.cfg.reward
        wait_pen = w.waiting * self.cfg.dt_seconds * self.building.num_waiting()
        in_car_pen = w.in_car * self.cfg.dt_seconds * self.building.num_in_car()
        energy_pen = w.energy * events.motion_floors
        dir_pen = w.direction_change * events.direction_changes
        completion = w.completion_bonus * len(events.arrivals)
        reward = float(-wait_pen - in_car_pen - energy_pen - dir_pen + completion)

        terminated = False
        truncated = self._step_count >= self.cfg.horizon_steps

        obs = self._observe()
        info = self._info()
        info.update(
            {
                "boarded": events.boarded,
                "arrivals": len(events.arrivals),
                "wait_penalty": wait_pen,
                "in_car_penalty": in_car_pen,
                "energy_penalty": energy_pen,
                "direction_change_penalty": dir_pen,
                "completion_bonus": completion,
            }
        )
        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self.render_ascii()
        if self.render_mode == "human":
            print(self.render_ascii())
            return None
        return None

    # --------------- helpers ---------------

    def render_ascii(self) -> str:
        f = self.cfg.building.num_floors
        n = self.cfg.building.num_elevators
        rows: list[str] = []
        header = f"  step={self._step_count}  t={self._sim_time_s:.0f}s  waiting={self.building.num_waiting()}  in_car={self.building.num_in_car()}  done={len(self.building.completed)}"
        rows.append(header)
        for floor in reversed(range(f)):
            cells: list[str] = []
            for e in self.building.elevators:
                if e.floor == floor and not e.is_servicing:
                    if e.direction > 0:
                        cells.append(f"[^{e.num_passengers:>2}]")
                    elif e.direction < 0:
                        cells.append(f"[v{e.num_passengers:>2}]")
                    else:
                        cells.append(f"[ {e.num_passengers:>2}]")
                elif e.floor == floor and e.is_servicing:
                    cells.append(f"[*{e.num_passengers:>2}]")
                else:
                    cells.append("     ")
            up = "^" if self.building.hall_up[floor] else " "
            dn = "v" if self.building.hall_down[floor] else " "
            wait_n = len(self.building.waiting[floor])
            rows.append(f"  {floor:>2} {up}{dn} {' '.join(cells)}   waiting={wait_n}")
        return "\n".join(rows)

    def _observe(self) -> np.ndarray:
        f = self.cfg.building.num_floors
        n = self.cfg.building.num_elevators

        out = np.zeros(self.obs_dim, dtype=np.float32)
        idx = 0
        for e in self.building.elevators:
            # one-hot floor
            out[idx + e.floor] = 1.0
            idx += f
            # direction in {-1,0,1}
            out[idx] = float(e.direction)
            idx += 1
            # load
            out[idx] = e.num_passengers / max(self.cfg.building.capacity, 1)
            idx += 1
            # servicing flag
            out[idx] = 1.0 if e.is_servicing else 0.0
            idx += 1

        # hall calls up/down
        out[idx : idx + f] = self.building.hall_up.astype(np.float32)
        idx += f
        out[idx : idx + f] = self.building.hall_down.astype(np.float32)
        idx += f

        # time fraction
        out[idx] = self._step_count / max(self.cfg.horizon_steps, 1)

        return out

    def _info(self) -> dict[str, Any]:
        return {
            "sim_time_s": self._sim_time_s,
            "num_waiting": self.building.num_waiting(),
            "num_in_car": self.building.num_in_car(),
            "num_completed": len(self.building.completed),
            "cumulative_wait_s": self.building.cumulative_wait_seconds,
            "cumulative_in_car_s": self.building.cumulative_in_car_seconds,
        }

    def metrics(self) -> dict[str, float]:
        """End-of-episode aggregate metrics (average wait + journey times)."""
        completed = self.building.completed
        if not completed:
            return {
                "completed": 0.0,
                "avg_wait_s": 0.0,
                "avg_journey_s": 0.0,
                "throughput": 0.0,
            }
        waits = [
            (p.pickup_time or p.spawn_time) - p.spawn_time for p in completed
        ]
        journeys = [
            (p.arrival_time or p.spawn_time) - p.spawn_time for p in completed
        ]
        return {
            "completed": float(len(completed)),
            "avg_wait_s": float(np.mean(waits)),
            "avg_journey_s": float(np.mean(journeys)),
            "throughput": float(len(completed) / max(self._sim_time_s, 1.0)),
        }
