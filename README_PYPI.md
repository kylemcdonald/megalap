# megalap

`megalap` is a Python package with a native C++ core and `nanobind` bindings for point-to-grid assignment.

The public API has three functions:

1. `linear_sum_assignment(cost_matrix)`
2. `window_cleanup(points, initial_assignment, rows, cols, budget_seconds, ...)`
3. `snap_to_grid(points, width=None, height=None, cleanup_seconds=None, ...)`

## Install

```bash
python -m pip install megalap
```

To run the matplotlib example from the source tree:

```bash
python -m pip install -e '.[examples]'
python examples/basic_usage.py
```

## API

### `linear_sum_assignment(cost_matrix)`

Solve a dense square linear assignment problem with the native C++ Jonker-Volgenant implementation.

Returns:

- `row_ind`: `int64` NumPy array of shape `(n,)`
- `col_ind`: `int64` NumPy array of shape `(n,)`
- `total_cost`: Python `float`

### `window_cleanup(points, initial_assignment, rows, cols, budget_seconds, ...)`

Run the overlapping-window cleanup kernel using native C++ threads.

Key options:

- `window_size=6`
- `num_threads=None` to use `std::thread::hardware_concurrency()`
- `num_threads=1` to force serial execution
- `fixed_suffix_count` to keep a suffix of target cells fixed

Returns a dict with:

- `assignment`
- `passes_completed`
- `elapsed_s`
- `final_cost`

### `snap_to_grid(points, width=None, height=None, cleanup_seconds=None, ...)`

High-level wrapper for snapping a 2D point cloud onto a destination grid.

Behavior:

- chooses a destination grid automatically when `width` and `height` are omitted
- prefers exact rectangular sizes with aspect ratio in `[1:1, 2:1]`
- falls back to a near-square enclosing grid in that same band when exact factors do not exist
- pads with edge ghost points when the chosen grid has more cells than real points
- uses the native auction LAP solver by default when the padded assignment problem has fewer than `10000` cells
- uses the native iterative cleanup solver for `10s` by default when the padded assignment problem has `10000` cells or more
- pass `cleanup_seconds` to override the default iterative budget; `cleanup_seconds=0.0` disables cleanup
- pass `exact_point_limit` to adjust the automatic exact/iterative threshold

Returns:

- `grid_points`: `(n, 2)` float64 NumPy array of assigned destination points in original point order
- `assignment`: `(n,)` int64 NumPy array of destination-grid indices in original point order
- `(width, height)`: destination-grid size tuple

## More

- Source repository: https://github.com/kylemcdonald/megalap
- Issue tracker: https://github.com/kylemcdonald/megalap/issues
- Example scripts: https://github.com/kylemcdonald/megalap/tree/main/examples
