from __future__ import annotations

import numpy as np
import pytest

import megalap


def test_linear_sum_assignment_finds_known_optimum() -> None:
    cost = np.array(
        [
            [4.0, 1.0, 3.0],
            [2.0, 0.0, 5.0],
            [3.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    row_ind, col_ind, total_cost = megalap.linear_sum_assignment(cost)

    assert row_ind.tolist() == [0, 1, 2]
    assert col_ind.tolist() == [1, 0, 2]
    assert total_cost == pytest.approx(5.0)


def test_snap_to_grid_returns_unique_assignment_for_non_rectangular_count() -> None:
    points = np.array(
        [
            [0.10, 0.20],
            [0.85, 0.15],
            [0.20, 0.80],
            [0.75, 0.70],
            [0.50, 0.45],
        ],
        dtype=np.float64,
    )

    grid_points, assignment, grid_size = megalap.snap_to_grid(points)

    assert grid_points.shape == points.shape
    assert assignment.shape == (points.shape[0],)
    assert len(set(assignment.tolist())) == points.shape[0]
    assert grid_size[0] * grid_size[1] >= points.shape[0]


def test_window_cleanup_returns_expected_result_shape() -> None:
    points = np.array(
        [
            [0.03, 0.03],
            [0.97, 0.03],
            [0.03, 0.97],
            [0.97, 0.97],
        ],
        dtype=np.float64,
    )
    initial_assignment = np.arange(4, dtype=np.int64)

    result = megalap.window_cleanup(
        points,
        initial_assignment,
        rows=2,
        cols=2,
        budget_seconds=0.0,
        num_threads=1,
    )

    assert result["assignment"].dtype == np.int64
    assert result["assignment"].shape == (4,)
    assert sorted(result["assignment"].tolist()) == [0, 1, 2, 3]
    assert result["passes_completed"] >= 1
    assert result["elapsed_s"] >= 0.0
