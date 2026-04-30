"""Discrete-time building dynamics.

The :class:`Building` owns elevator state, hall-call bookkeeping, and the
boarding logic. The Gymnasium environment wraps it and produces observations,
actions, rewards, and metrics. This split keeps physics testable in isolation.

Action semantics (per elevator, discrete):
    0 = DOWN, 1 = HOLD, 2 = UP

Per-step rules:
- An elevator currently servicing a floor (door cycle running) ignores its
  action; the door cycle counts down until boarding/alighting completes.
- An elevator at rest at an integer floor adopts the agent's commanded
  direction. ``HOLD`` keeps it idle.
- A moving elevator continues toward the next integer floor (no mid-flight
  reversal). When it arrives, the env decides whether to stop:
    * stop if any internal passenger wants to alight here,
    * stop if a hall call in the elevator's direction exists here,
    * stop if the new action is HOLD or opposite to the current direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from envs.config import BuildingConfig
from envs.traffic import Passenger

ACTION_DOWN, ACTION_HOLD, ACTION_UP = 0, 1, 2


@dataclass
class ElevatorState:
    floor: int = 0
    pos_frac: float = 0.0  # 0..1 distance toward `floor + direction`
    direction: int = 0  # -1, 0, +1
    last_direction: int = 0
    service_timer: float = 0.0  # >0 means door cycle / boarding in progress
    passengers: list[Passenger] = field(default_factory=list)

    @property
    def is_servicing(self) -> bool:
        return self.service_timer > 1e-9

    @property
    def num_passengers(self) -> int:
        return len(self.passengers)

    def position(self) -> float:
        return self.floor + self.direction * self.pos_frac if self.direction != 0 else float(self.floor)

    def internal_call_mask(self, num_floors: int) -> np.ndarray:
        mask = np.zeros(num_floors, dtype=np.int8)
        for p in self.passengers:
            mask[p.destination] = 1
        return mask


@dataclass
class StepEvents:
    boarded: int = 0
    arrivals: list[Passenger] = field(default_factory=list)
    direction_changes: int = 0
    motion_floors: float = 0.0  # total floor-equivalents moved this step


class Building:
    """Owns elevator and hall-call state. Stateful; reset() before reuse."""

    def __init__(self, cfg: BuildingConfig, dt_seconds: float) -> None:
        self.cfg = cfg
        self.dt = dt_seconds

        # 1 floor takes (floor_height / speed) seconds at full speed.
        self.seconds_per_floor: float = cfg.floor_height_m / max(cfg.speed_mps, 1e-6)
        self.floors_per_step: float = self.dt / self.seconds_per_floor

        # service time = open + close + (boarding handled additively below)
        self.base_service_seconds: float = cfg.door_open_seconds + cfg.door_close_seconds

        self.num_floors = cfg.num_floors
        self.num_elevators = cfg.num_elevators

        self.elevators: list[ElevatorState] = []
        # hall calls: shape (F, 2) — col 0 = up call pending, col 1 = down call pending
        self.hall_up: np.ndarray = np.zeros(self.num_floors, dtype=np.int8)
        self.hall_down: np.ndarray = np.zeros(self.num_floors, dtype=np.int8)
        # waiting passengers at each floor (FIFO per floor)
        self.waiting: list[list[Passenger]] = [[] for _ in range(self.num_floors)]

        # bookkeeping for metrics + reward computation
        self.completed: list[Passenger] = []
        self.cumulative_wait_seconds: float = 0.0
        self.cumulative_in_car_seconds: float = 0.0

        self.reset()

    def reset(self) -> None:
        self.elevators = [ElevatorState(floor=0) for _ in range(self.num_elevators)]
        self.hall_up.fill(0)
        self.hall_down.fill(0)
        self.waiting = [[] for _ in range(self.num_floors)]
        self.completed = []
        self.cumulative_wait_seconds = 0.0
        self.cumulative_in_car_seconds = 0.0

    # ---------- inbound passengers ----------

    def admit_arrivals(self, new_passengers: list[Passenger]) -> None:
        for p in new_passengers:
            self.waiting[p.origin].append(p)
            if p.direction > 0:
                self.hall_up[p.origin] = 1
            else:
                self.hall_down[p.origin] = 1

    # ---------- per-step dynamics ----------

    def step(self, actions: np.ndarray, sim_time_s: float) -> StepEvents:
        ev = StepEvents()

        # 1) Tick service timers; finalize boarding when timer hits zero.
        for e in self.elevators:
            if e.is_servicing:
                e.service_timer = max(0.0, e.service_timer - self.dt)
                if not e.is_servicing:
                    self._service_floor(e, sim_time_s, ev)

        # 2) Advance motion / take new commands for resting elevators.
        for idx, e in enumerate(self.elevators):
            if e.is_servicing:
                continue
            action = int(actions[idx])
            if e.direction == 0:
                # at rest at an integer floor: adopt action
                if action == ACTION_UP and e.floor < self.num_floors - 1:
                    e.direction = 1
                    e.last_direction = 1
                elif action == ACTION_DOWN and e.floor > 0:
                    e.direction = -1
                    e.last_direction = -1
                else:
                    # HOLD or boundary; remain idle
                    continue

            # advance in current direction; cannot reverse mid-flight
            e.pos_frac += self.floors_per_step
            ev.motion_floors += self.floors_per_step

            if e.pos_frac >= 1.0:
                # arrived at next integer floor
                next_floor = e.floor + e.direction
                # snap to floor
                e.floor = next_floor
                e.pos_frac = 0.0

                if self._should_stop(e, action):
                    e.service_timer = self._compute_service_seconds(e)
                    # mark direction-change penalty if action is opposite
                    if action != ACTION_HOLD and action - 1 == -e.direction:
                        ev.direction_changes += 1
                    e.direction = 0
                    # service this floor immediately if timer == 0 (no boarders)
                    if e.service_timer <= 0:
                        self._service_floor(e, sim_time_s, ev)

        # 3) Accumulate wait/in-car time for everyone still in the system.
        for floor_q in self.waiting:
            self.cumulative_wait_seconds += self.dt * len(floor_q)
        for e in self.elevators:
            self.cumulative_in_car_seconds += self.dt * len(e.passengers)

        return ev

    # ---------- internal helpers ----------

    def _should_stop(self, e: ElevatorState, action: int) -> bool:
        # stop if internal passenger alights here
        if any(p.destination == e.floor for p in e.passengers):
            return True
        # stop if same-direction hall call here
        if e.direction > 0 and self.hall_up[e.floor]:
            return True
        if e.direction < 0 and self.hall_down[e.floor]:
            return True
        # stop if agent commanded HOLD or opposite direction
        if action == ACTION_HOLD:
            return True
        if action == ACTION_DOWN and e.direction > 0:
            return True
        if action == ACTION_UP and e.direction < 0:
            return True
        # stop if at terminal floor
        if e.floor == 0 and e.direction < 0:
            return True
        if e.floor == self.num_floors - 1 and e.direction > 0:
            return True
        return False

    def _compute_service_seconds(self, e: ElevatorState) -> float:
        # crude estimate before boarding; refined when service finalizes
        # base door open/close + 1s per anticipated alight + 1s per boarder slot
        anticipated_alights = sum(1 for p in e.passengers if p.destination == e.floor)
        anticipated_boards = min(
            self.cfg.capacity - (len(e.passengers) - anticipated_alights),
            len(self.waiting[e.floor]),
        )
        n = anticipated_alights + max(0, anticipated_boards)
        return self.base_service_seconds + self.cfg.boarding_seconds_per_passenger * n

    def _service_floor(self, e: ElevatorState, sim_time_s: float, ev: StepEvents) -> None:
        """Execute alighting + boarding at the elevator's current floor."""
        # 1) alight
        still_riding: list[Passenger] = []
        for p in e.passengers:
            if p.destination == e.floor:
                p.arrival_time = sim_time_s
                self.completed.append(p)
                ev.arrivals.append(p)
            else:
                still_riding.append(p)
        e.passengers = still_riding

        floor_idx = e.floor
        queue = self.waiting[floor_idx]

        # If the car emptied, reset directional context so the next boarders
        # can set a fresh direction (otherwise an elevator returning empty to
        # the lobby in a downward motion would refuse all upbound boarders).
        if not e.passengers:
            # Also at terminal floors the only sensible direction is the
            # reverse of last_direction.
            if e.floor == 0:
                e.last_direction = 0
            elif e.floor == self.num_floors - 1:
                e.last_direction = 0

        if not queue:
            self._update_hall_lights(floor_idx)
            return

        # candidate_dir is either fixed by remaining riders or open (0) to be
        # set by the first boarder.
        candidate_dir = e.last_direction
        boarded_any = False
        new_queue: list[Passenger] = []
        for p in queue:
            if e.num_passengers >= self.cfg.capacity:
                new_queue.append(p)
                continue
            if candidate_dir == 0 or p.direction == candidate_dir:
                p.pickup_time = sim_time_s
                e.passengers.append(p)
                if candidate_dir == 0:
                    candidate_dir = p.direction
                    e.last_direction = p.direction
                ev.boarded += 1
                boarded_any = True
            else:
                new_queue.append(p)

        self.waiting[floor_idx] = new_queue
        self._update_hall_lights(floor_idx)

        if boarded_any:
            if candidate_dir > 0:
                self.hall_up[floor_idx] = 1 if any(p.direction > 0 for p in new_queue) else 0
            elif candidate_dir < 0:
                self.hall_down[floor_idx] = 1 if any(p.direction < 0 for p in new_queue) else 0

    def _update_hall_lights(self, floor_idx: int) -> None:
        q = self.waiting[floor_idx]
        self.hall_up[floor_idx] = 1 if any(p.direction > 0 for p in q) else 0
        self.hall_down[floor_idx] = 1 if any(p.direction < 0 for p in q) else 0

    # ---------- diagnostics ----------

    def num_waiting(self) -> int:
        return sum(len(q) for q in self.waiting)

    def num_in_car(self) -> int:
        return sum(len(e.passengers) for e in self.elevators)
