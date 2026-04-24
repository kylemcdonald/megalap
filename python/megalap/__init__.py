from __future__ import annotations

import math

import numpy as np

from ._core import _auction_grid_assignment, _linear_sum_assignment, _window_cleanup

DEFAULT_EXACT_POINT_LIMIT = 10_000
DEFAULT_ITERATIVE_SECONDS = 10.0

__all__ = [
    "DEFAULT_EXACT_POINT_LIMIT",
    "DEFAULT_ITERATIVE_SECONDS",
    "linear_sum_assignment",
    "window_cleanup",
    "snap_to_grid",
]


def _build_target_grid(width: int, height: int, margin: float) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width > 1:
        xs = np.linspace(margin, 1.0 - margin, width, dtype=np.float64)
    else:
        xs = np.array([0.5], dtype=np.float64)
    if height > 1:
        ys = np.linspace(margin, 1.0 - margin, height, dtype=np.float64)
    else:
        ys = np.array([0.5], dtype=np.float64)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])


def _choose_grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be positive")

    exact_candidates: list[tuple[int, int]] = []
    root = int(math.isqrt(n))
    for height in range(1, root + 1):
        if n % height != 0:
            continue
        width = n // height
        if width < height:
            width, height = height, width
        if width / height <= 2.0:
            exact_candidates.append((width, height))

    if exact_candidates:
        return min(exact_candidates, key=lambda wh: (wh[0] - wh[1], wh[0]))

    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int] | None = None
    max_height = int(math.ceil(math.sqrt(n)))
    for height in range(1, max_height + 1):
        width = math.ceil(n / height)
        if width < height:
            width, height = height, width
        if width / height > 2.0:
            continue
        area = width * height
        key = (area, width - height, width)
        if best_key is None or key < best_key:
            best_key = key
            best = (width, height)

    if best is None:
        side = int(math.ceil(math.sqrt(n)))
        return side, side
    return best


def linear_sum_assignment(cost_matrix):
    cost = np.asarray(cost_matrix, dtype=np.float64, order="C")
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost_matrix must be a square 2D array")
    row_ind, col_ind, total_cost = _linear_sum_assignment(cost)
    return (
        np.asarray(row_ind, dtype=np.int64),
        np.asarray(col_ind, dtype=np.int64),
        float(total_cost),
    )


def _auction_grid_lap(points: np.ndarray, rows: int, cols: int, margin: float) -> np.ndarray:
    _, col_ind, _ = _auction_grid_assignment(points, int(rows), int(cols), float(margin))
    return np.asarray(col_ind, dtype=np.int64)


def _seed_grid_assignment(points: np.ndarray, ghost_count: int) -> np.ndarray:
    n_total = int(points.shape[0])
    n_real = n_total - int(ghost_count)
    assignment = np.empty(n_total, dtype=np.int64)

    if n_real > 0:
        order = np.lexsort((points[:n_real, 0], points[:n_real, 1]))
        assignment[order] = np.arange(n_real, dtype=np.int64)

    if ghost_count:
        assignment[n_real:] = np.arange(n_real, n_total, dtype=np.int64)

    return assignment


def _normalize_num_threads(num_threads: int | None) -> int:
    if num_threads is None:
        return 0
    num_threads = int(num_threads)
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    return num_threads


def window_cleanup(
    points,
    initial_assignment,
    rows: int,
    cols: int,
    budget_seconds: float,
    window_size: int = 6,
    margin: float = 0.03,
    num_threads: int | None = None,
    fixed_suffix_count: int = 0,
):
    pts = np.asarray(points, dtype=np.float64, order="C")
    assignment = np.asarray(initial_assignment, dtype=np.int64, order="C")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if assignment.ndim != 1 or assignment.shape[0] != pts.shape[0]:
        raise ValueError("initial_assignment must have shape (n,)")
    if int(rows) * int(cols) != pts.shape[0]:
        raise ValueError("rows * cols must equal the number of points")
    result = _window_cleanup(
        pts,
        assignment,
        int(rows),
        int(cols),
        float(budget_seconds),
        int(window_size),
        float(margin),
        int(fixed_suffix_count),
        _normalize_num_threads(num_threads),
    )
    result["assignment"] = np.asarray(result["assignment"], dtype=np.int64)
    return result


def snap_to_grid(
    points,
    width: int | None = None,
    height: int | None = None,
    cleanup_seconds: float | None = None,
    window_size: int = 6,
    margin: float = 0.03,
    num_threads: int | None = None,
    exact_point_limit: int = DEFAULT_EXACT_POINT_LIMIT,
):
    pts = np.asarray(points, dtype=np.float64, order="C")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    n_real = int(pts.shape[0])
    if width is None and height is None:
        width, height = _choose_grid_shape(n_real)
    elif width is None or height is None:
        raise ValueError("width and height must be provided together")
    else:
        width = int(width)
        height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    total_cells = width * height
    if total_cells < n_real:
        raise ValueError("width * height must be at least the number of source points")

    target_points = _build_target_grid(width, height, float(margin))
    ghost_count = total_cells - n_real

    if ghost_count:
        ghost_points = target_points[-ghost_count:].copy()
        augmented_points = np.vstack([pts, ghost_points])
    else:
        augmented_points = pts

    if total_cells < int(exact_point_limit):
        assignment = _auction_grid_lap(augmented_points, rows=height, cols=width, margin=margin)
        cleanup_budget = 0.0 if cleanup_seconds is None else float(cleanup_seconds)
    else:
        assignment = _seed_grid_assignment(augmented_points, ghost_count)
        cleanup_budget = DEFAULT_ITERATIVE_SECONDS if cleanup_seconds is None else float(cleanup_seconds)

    if cleanup_budget > 0.0:
        cleanup = window_cleanup(
            augmented_points,
            assignment,
            rows=height,
            cols=width,
            budget_seconds=cleanup_budget,
            window_size=window_size,
            margin=margin,
            num_threads=num_threads,
            fixed_suffix_count=ghost_count,
        )
        assignment = cleanup["assignment"]

    assignment = np.asarray(assignment, dtype=np.int64)
    grid_points = target_points[assignment[:n_real]]
    return grid_points, assignment[:n_real].copy(), (width, height)
