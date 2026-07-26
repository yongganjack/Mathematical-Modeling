from __future__ import annotations

import numpy as np


def test_pso_is_reproducible_and_improves_sphere() -> None:
    from question2.model import solve_pso

    def objective(position: np.ndarray) -> float:
        return -float(np.dot(position, position))

    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    first = solve_pso(
        objective, bounds, np.random.default_rng(123), particles=12, iterations=10
    )
    second = solve_pso(
        objective, bounds, np.random.default_rng(123), particles=12, iterations=10
    )

    assert np.array_equal(first.best_position, second.best_position)
    assert first.best_score == second.best_score
    assert first.history == second.history
    assert first.best_score >= first.history[0]["best"]
    assert first.best_score > -1.0
    assert first.evaluations == 12 * 11
