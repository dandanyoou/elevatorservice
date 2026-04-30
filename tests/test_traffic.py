"""Traffic generator tests: OD properties + reproducibility."""
from __future__ import annotations

import numpy as np
import pytest

from envs.config import TrafficConfig
from envs.traffic import PoissonTrafficGenerator, od_matrix


def test_od_matrix_zero_diagonal_and_normalized() -> None:
    for pattern in ("uniform", "morning", "lunch", "evening"):
        m = od_matrix(num_floors=8, pattern=pattern)
        assert m.shape == (8, 8)
        assert (np.diag(m) == 0).all()
        assert m.sum() == pytest.approx(1.0)


def test_od_matrix_unknown_pattern_raises() -> None:
    with pytest.raises(ValueError):
        od_matrix(num_floors=4, pattern="rocket")


def test_traffic_determinism() -> None:
    cfg = TrafficConfig(pattern="morning", base_rate_per_minute=30.0, seed=123)
    a = PoissonTrafficGenerator(num_floors=8, cfg=cfg, dt_seconds=1.0)
    b = PoissonTrafficGenerator(num_floors=8, cfg=cfg, dt_seconds=1.0)
    a_arrivals = []
    b_arrivals = []
    for t in range(200):
        a_arrivals.extend((p.origin, p.destination) for p in a.step(float(t)))
        b_arrivals.extend((p.origin, p.destination) for p in b.step(float(t)))
    assert a_arrivals == b_arrivals


def test_morning_pattern_skews_lobby() -> None:
    cfg = TrafficConfig(pattern="morning", base_rate_per_minute=120.0, seed=0)
    gen = PoissonTrafficGenerator(num_floors=8, cfg=cfg, dt_seconds=1.0)
    lobby_origins = 0
    total = 0
    for t in range(600):
        for p in gen.step(float(t)):
            total += 1
            if p.origin == 0:
                lobby_origins += 1
    assert total > 50
    assert lobby_origins / total > 0.5
