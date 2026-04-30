"""Run a short simulation and produce both an ASCII trace and a matplotlib animation.

Usage:
    python -m scripts.visualize --policy scan --floors 8 --elevators 2 --steps 300
    python -m scripts.visualize --policy nearest --gif out/run.gif

If matplotlib animation export fails (no ffmpeg), a multi-frame PNG is written instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; works in CI / Windows without DISPLAY
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

from agents.baseline import NearestCarPolicy, RandomPolicy, ScanPolicy  # noqa: E402
from envs.config import BuildingConfig, ElevatorEnvConfig, TrafficConfig  # noqa: E402
from envs.elevator_env import ElevatorEnv  # noqa: E402

POLICIES = {
    "scan": ScanPolicy,
    "nearest": NearestCarPolicy,
    "random": RandomPolicy,
}


def build_env(args: argparse.Namespace) -> ElevatorEnv:
    cfg = ElevatorEnvConfig(
        building=BuildingConfig(
            num_floors=args.floors, num_elevators=args.elevators, capacity=args.capacity
        ),
        traffic=TrafficConfig(
            pattern=args.pattern, base_rate_per_minute=args.rate, seed=args.seed
        ),
        horizon_steps=args.steps,
    )
    return ElevatorEnv(cfg)


def run(args: argparse.Namespace) -> tuple[list[np.ndarray], dict[str, float], list[str]]:
    env = build_env(args)
    policy_cls = POLICIES[args.policy]
    policy = policy_cls(env)

    obs, _ = env.reset(seed=args.seed)
    frames: list[np.ndarray] = []
    ascii_frames: list[str] = []

    f = env.cfg.building.num_floors
    n = env.cfg.building.num_elevators

    for t in range(args.steps):
        action = policy.act() if hasattr(policy, "act") else env.action_space.sample()
        obs, _r, _term, trunc, _info = env.step(action)
        if t % args.snapshot_every == 0:
            grid = render_grid(env, f, n)
            frames.append(grid)
            ascii_frames.append(env.render_ascii())
        if trunc:
            break

    metrics = env.metrics()
    return frames, metrics, ascii_frames


def render_grid(env: ElevatorEnv, f: int, n: int) -> np.ndarray:
    """Build a small RGB image: rows = floors (top→down), columns = elevators + waiting bar."""
    cell = 24
    width_cols = n + 1  # last column shows waiting count per floor
    img = np.full((f * cell, width_cols * cell, 3), 240, dtype=np.uint8)

    for floor in range(f):
        # invert Y: floor 0 at bottom
        y_top = (f - 1 - floor) * cell
        # waiting bar
        wait_n = len(env.building.waiting[floor])
        wait_intensity = min(wait_n * 50, 255)
        img[y_top : y_top + cell, n * cell : (n + 1) * cell] = (
            255,
            max(0, 255 - wait_intensity),
            max(0, 255 - wait_intensity),
        )

    for i, e in enumerate(env.building.elevators):
        # interpolate vertical position
        floor_pos = e.floor + e.direction * e.pos_frac
        y = (f - 1 - floor_pos) * cell
        y0, y1 = int(y), int(y) + cell
        x0, x1 = i * cell, (i + 1) * cell
        # color by load
        load = e.num_passengers / max(env.cfg.building.capacity, 1)
        r = int(80 + 175 * load)
        g = 80 if e.is_servicing else int(180 - 100 * load)
        b = int(220 * (1 - load))
        img[max(0, y0) : min(f * cell, y1), x0:x1] = (r, g, b)
        # door open marker (white border)
        if e.is_servicing:
            img[max(0, y0) : max(0, y0) + 2, x0:x1] = (255, 255, 255)
            img[min(f * cell - 1, y1 - 2) : min(f * cell, y1), x0:x1] = (255, 255, 255)

    return img


def write_animation(frames: list[np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i: int) -> tuple[plt.Artist]:
        im.set_data(frames[i])
        ax.set_title(f"frame {i}/{len(frames)}")
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=120, blit=True)
    try:
        anim.save(out_path, writer="pillow")
        print(f"[visualize] wrote animation: {out_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[visualize] gif write failed ({exc}); falling back to grid PNG")
        grid_path = out_path.with_suffix(".png")
        cols = min(8, len(frames))
        rows = (len(frames) + cols - 1) // cols
        fig2, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 2.5))
        axes = np.atleast_2d(axes)
        for k, frame in enumerate(frames):
            axes[k // cols, k % cols].imshow(frame)
            axes[k // cols, k % cols].axis("off")
        for k in range(len(frames), rows * cols):
            axes[k // cols, k % cols].axis("off")
        fig2.tight_layout()
        fig2.savefig(grid_path, dpi=110)
        print(f"[visualize] wrote grid: {grid_path}")
    finally:
        plt.close("all")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a baseline policy and visualize the result")
    p.add_argument("--policy", choices=POLICIES.keys(), default="scan")
    p.add_argument("--floors", type=int, default=8)
    p.add_argument("--elevators", type=int, default=3)
    p.add_argument("--capacity", type=int, default=10)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--rate", type=float, default=20.0, help="passengers per minute")
    p.add_argument("--pattern", choices=("uniform", "morning", "lunch", "evening"), default="morning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot-every", type=int, default=2)
    p.add_argument("--gif", default="out/run.gif", type=str)
    p.add_argument("--ascii-tail", type=int, default=3, help="print last N ASCII frames")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames, metrics, ascii_frames = run(args)

    print("\n=== last ASCII frames ===")
    for af in ascii_frames[-args.ascii_tail :]:
        print(af)
        print("-" * 40)

    print("\n=== episode metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    if frames:
        write_animation(frames, Path(args.gif))


if __name__ == "__main__":
    main()
