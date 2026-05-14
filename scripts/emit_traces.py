"""Emit per-step JSON traces for the web dashboard.

For each policy in {scan, nearest, random} and each seed in a configurable
range, run the env for ``--steps`` and dump a JSON trace conforming to the
schema in the project design doc.

Usage:
    python -m scripts.emit_traces --out web/traces --seeds 10 --steps 1200

Trace shape (per file):
    {
      "schema_version": 1,
      "policy": "scan",
      "episode_id": "scan-seed0007",
      "seed": 7,
      "traffic_pattern": "morning",
      "num_elevators": 3,
      "num_floors": 10,
      "total_steps": 1200,
      "simulator_version": "0.1.0",
      "metrics": { "avg_wait_s": 31.2, "avg_journey_s": 58.0, ... },
      "steps": [
        {
          "t": 0,
          "elevator_positions": [0, 0, 0],
          "elevator_loads": [0, 0, 0],
          "elevator_directions": [0, 0, 0],
          "elevator_servicing": [false, false, false],
          "hall_calls_up": [0, 0, ...],
          "hall_calls_down": [0, 0, ...],
          "waiting_per_floor": [0, 0, ...],
          "arrivals_this_step": 0,
          "completions_this_step": 0,
          "reward": 0.0
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.baseline import BasePolicy, NearestCarPolicy, RandomPolicy, ScanPolicy
from envs.config import BuildingConfig, ElevatorEnvConfig, TrafficConfig
from envs.elevator_env import ElevatorEnv

SCHEMA_VERSION = 1
SIMULATOR_VERSION = "0.1.0"

POLICIES: dict[str, type[BasePolicy]] = {
    "scan": ScanPolicy,
    "nearest": NearestCarPolicy,
    "random": RandomPolicy,
}


def emit_one(policy_name: str, seed: int, args: argparse.Namespace) -> dict:
    cfg = ElevatorEnvConfig(
        building=BuildingConfig(
            num_floors=args.floors,
            num_elevators=args.elevators,
            capacity=args.capacity,
        ),
        traffic=TrafficConfig(
            pattern=args.pattern,
            base_rate_per_minute=args.rate,
            seed=seed,
        ),
        horizon_steps=args.steps,
    )
    env = ElevatorEnv(cfg)
    policy = POLICIES[policy_name](env)
    env.reset(seed=seed)

    def _system_total(b) -> int:
        return (
            sum(len(q) for q in b.waiting)
            + sum(len(e.passengers) for e in b.elevators)
            + len(b.completed)
        )

    prev_total = _system_total(env.building)
    steps: list[dict] = []
    for t in range(args.steps):
        action = policy.act()
        _, reward, _, _, info = env.step(action)
        b = env.building
        total_now = _system_total(b)
        new_arrivals = max(0, total_now - prev_total)
        prev_total = total_now
        # NOTE: env labels destination-arrivals as "arrivals" in info, and
        # boardings as "boarded". This trace renames for clarity.
        steps.append(
            {
                "t": t,
                "elevator_positions": [round(float(e.position()), 3) for e in b.elevators],
                "elevator_loads": [e.num_passengers for e in b.elevators],
                "elevator_directions": [int(e.direction) for e in b.elevators],
                "elevator_servicing": [bool(e.is_servicing) for e in b.elevators],
                "hall_calls_up": b.hall_up.astype(int).tolist(),
                "hall_calls_down": b.hall_down.astype(int).tolist(),
                "waiting_per_floor": [len(q) for q in b.waiting],
                "arrivals_this_step": int(new_arrivals),
                "boardings_this_step": int(info["boarded"]),
                "completions_this_step": int(info["arrivals"]),
                "reward": round(float(reward), 4),
            }
        )

    metrics = env.metrics()
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy_name,
        "episode_id": f"{policy_name}-seed{seed:04d}",
        "seed": seed,
        "traffic_pattern": args.pattern,
        "num_elevators": args.elevators,
        "num_floors": args.floors,
        "total_steps": args.steps,
        "simulator_version": SIMULATOR_VERSION,
        "metrics": {k: round(float(v), 3) for k, v in metrics.items()},
        "steps": steps,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="web/traces", help="output directory")
    p.add_argument("--seeds", type=int, default=10, help="number of seeds per policy")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--floors", type=int, default=10)
    p.add_argument("--elevators", type=int, default=3)
    p.add_argument("--capacity", type=int, default=12)
    p.add_argument("--rate", type=float, default=18.0)
    p.add_argument("--pattern", default="morning")
    p.add_argument(
        "--policies",
        nargs="+",
        default=list(POLICIES.keys()),
        help="subset of policies to emit",
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict]] = {"episodes": []}
    for policy_name in args.policies:
        for s in range(args.seeds):
            seed = args.seed_base + s
            trace = emit_one(policy_name, seed, args)
            fname = f"{policy_name}-seed{seed:04d}.json"
            path = out_dir / fname
            with path.open("w") as f:
                json.dump(trace, f, separators=(",", ":"))
            manifest["episodes"].append(
                {
                    "file": fname,
                    "policy": policy_name,
                    "seed": seed,
                    "metrics": trace["metrics"],
                }
            )
            print(
                f"  emitted {fname}  AWT={trace['metrics']['avg_wait_s']:.2f}s  "
                f"completed={trace['metrics']['completed']:.0f}"
            )

    manifest["config"] = {
        "schema_version": SCHEMA_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "pattern": args.pattern,
        "num_elevators": args.elevators,
        "num_floors": args.floors,
        "total_steps": args.steps,
        "policies": args.policies,
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest.json with {len(manifest['episodes'])} episodes")


if __name__ == "__main__":
    main()
