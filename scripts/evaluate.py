"""Compare baselines on identical seeds. Reports AWT, AJT, throughput.

Usage:
    python -m scripts.evaluate --episodes 20 --steps 1800
"""
from __future__ import annotations

import argparse
from statistics import mean, stdev

from agents.baseline import BasePolicy, NearestCarPolicy, RandomPolicy, ScanPolicy
from envs.config import BuildingConfig, ElevatorEnvConfig, TrafficConfig
from envs.elevator_env import ElevatorEnv

POLICIES: dict[str, type[BasePolicy]] = {
    "scan": ScanPolicy,
    "nearest": NearestCarPolicy,
    "random": RandomPolicy,
}


def run_episode(policy_name: str, seed: int, args: argparse.Namespace) -> dict[str, float]:
    cfg = ElevatorEnvConfig(
        building=BuildingConfig(
            num_floors=args.floors, num_elevators=args.elevators, capacity=args.capacity
        ),
        traffic=TrafficConfig(pattern=args.pattern, base_rate_per_minute=args.rate, seed=seed),
        horizon_steps=args.steps,
    )
    env = ElevatorEnv(cfg)
    policy_cls = POLICIES[policy_name]
    policy = policy_cls(env)

    env.reset(seed=seed)
    for _ in range(args.steps):
        env.step(policy.act())
    return env.metrics()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--floors", type=int, default=10)
    p.add_argument("--elevators", type=int, default=3)
    p.add_argument("--capacity", type=int, default=12)
    p.add_argument("--rate", type=float, default=18.0)
    p.add_argument("--pattern", default="morning")
    p.add_argument("--seed-base", type=int, default=1000)
    args = p.parse_args()

    results: dict[str, dict[str, list[float]]] = {
        name: {"avg_wait_s": [], "avg_journey_s": [], "throughput": [], "completed": []}
        for name in POLICIES
    }

    for ep in range(args.episodes):
        seed = args.seed_base + ep
        for name in POLICIES:
            m = run_episode(name, seed, args)
            for k, v in m.items():
                results[name].setdefault(k, []).append(v)

    print(f"\n=== Baseline comparison ({args.episodes} episodes, pattern={args.pattern}) ===")
    print(f"{'policy':<10} {'AWT(s)':>10} {'AJT(s)':>10} {'thru/s':>10} {'completed':>12}")
    for name, runs in results.items():
        awt = mean(runs["avg_wait_s"])
        awt_sd = stdev(runs["avg_wait_s"]) if len(runs["avg_wait_s"]) > 1 else 0.0
        ajt = mean(runs["avg_journey_s"])
        thr = mean(runs["throughput"])
        comp = mean(runs["completed"])
        print(
            f"{name:<10} {awt:>7.2f}±{awt_sd:<3.1f}  {ajt:>9.2f}  {thr:>9.3f}  {comp:>10.1f}"
        )


if __name__ == "__main__":
    main()
