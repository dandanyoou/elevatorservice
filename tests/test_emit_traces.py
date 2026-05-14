"""Tests for scripts.emit_traces — schema, determinism, manifest integrity."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.emit_traces import SCHEMA_VERSION, emit_one


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        out="ignored",
        seeds=2,
        seed_base=1000,
        steps=60,
        floors=6,
        elevators=2,
        capacity=8,
        rate=12.0,
        pattern="morning",
        policies=["scan", "nearest", "random"],
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_trace_has_required_header_fields() -> None:
    trace = emit_one("scan", seed=1234, args=_args())
    for key in (
        "schema_version",
        "policy",
        "episode_id",
        "seed",
        "traffic_pattern",
        "num_elevators",
        "num_floors",
        "total_steps",
        "simulator_version",
        "metrics",
        "steps",
    ):
        assert key in trace, f"missing field: {key}"
    assert trace["schema_version"] == SCHEMA_VERSION
    assert trace["policy"] == "scan"
    assert trace["seed"] == 1234
    assert trace["episode_id"] == "scan-seed1234"
    assert len(trace["steps"]) == 60


def test_step_has_required_per_step_fields() -> None:
    trace = emit_one("nearest", seed=5, args=_args())
    step = trace["steps"][0]
    for key in (
        "t",
        "elevator_positions",
        "elevator_loads",
        "elevator_directions",
        "elevator_servicing",
        "hall_calls_up",
        "hall_calls_down",
        "waiting_per_floor",
        "arrivals_this_step",
        "boardings_this_step",
        "completions_this_step",
        "reward",
    ):
        assert key in step, f"missing per-step field: {key}"

    assert len(step["elevator_positions"]) == trace["num_elevators"]
    assert len(step["elevator_loads"]) == trace["num_elevators"]
    assert len(step["elevator_directions"]) == trace["num_elevators"]
    assert len(step["elevator_servicing"]) == trace["num_elevators"]
    assert len(step["hall_calls_up"]) == trace["num_floors"]
    assert len(step["hall_calls_down"]) == trace["num_floors"]
    assert len(step["waiting_per_floor"]) == trace["num_floors"]


def test_same_seed_same_trace_for_scan() -> None:
    """Determinism contract: same seed + same policy → identical trace."""
    a = emit_one("scan", seed=42, args=_args())
    b = emit_one("scan", seed=42, args=_args())
    assert a["steps"] == b["steps"]
    assert a["metrics"] == b["metrics"]


def test_different_seeds_diverge_eventually() -> None:
    """Sanity: seeds 1 and 2 should not produce identical traces."""
    a = emit_one("scan", seed=1, args=_args(steps=200))
    b = emit_one("scan", seed=2, args=_args(steps=200))
    assert a["steps"] != b["steps"], "different seeds produced identical traces"


def test_emit_traces_writes_manifest(tmp_path: Path) -> None:
    """End-to-end: emit a tiny batch and verify manifest + per-trace files."""
    out = tmp_path / "traces"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.emit_traces",
            "--out",
            str(out),
            "--seeds",
            "2",
            "--steps",
            "30",
            "--floors",
            "5",
            "--elevators",
            "2",
            "--policies",
            "scan",
            "random",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    if result.returncode != 0:
        pytest.fail(f"emit_traces failed:\nstdout={result.stdout}\nstderr={result.stderr}")

    manifest_path = out / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "episodes" in manifest
    assert "config" in manifest
    assert len(manifest["episodes"]) == 4  # 2 policies × 2 seeds

    for entry in manifest["episodes"]:
        f = out / entry["file"]
        assert f.exists(), f"missing trace file {f}"
        trace = json.loads(f.read_text())
        assert trace["policy"] == entry["policy"]
        assert trace["seed"] == entry["seed"]


def test_completions_sum_is_nonneg_and_bounded() -> None:
    """Sanity: completions, boardings, arrivals are non-negative and bounded."""
    trace = emit_one("nearest", seed=7, args=_args(steps=200))
    cap = trace["num_elevators"] * 8  # _args sets capacity=8
    total_completions = 0
    total_arrivals = 0
    for step in trace["steps"]:
        assert step["completions_this_step"] >= 0
        assert step["completions_this_step"] <= cap
        assert step["boardings_this_step"] >= 0
        assert step["boardings_this_step"] <= cap
        assert step["arrivals_this_step"] >= 0
        total_completions += step["completions_this_step"]
        total_arrivals += step["arrivals_this_step"]
    # arrivals (button presses) must be >= completions (destinations reached)
    assert total_arrivals >= total_completions
    # final metric must match summed completions exactly
    assert int(trace["metrics"]["completed"]) == total_completions
