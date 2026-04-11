from __future__ import annotations

import argparse
import pathlib
import struct
import zlib

import numpy as np

import megalap


def make_meandering_points(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0, scale=1.0, size=(n, 2))
    points = np.cumsum(steps, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, np.finfo(np.float64).eps)
    return np.asarray((points - mins) / span, dtype=np.float64)


def build_target_grid(width: int, height: int, margin: float) -> np.ndarray:
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


def solve_leaf(points: np.ndarray, target_points: np.ndarray, point_ids: np.ndarray, target_ids: np.ndarray, assignment: np.ndarray) -> None:
    local_points = points[point_ids]
    local_targets = target_points[target_ids]
    diffs = local_points[:, None, :] - local_targets[None, :, :]
    cost = np.sum(diffs * diffs, axis=2, dtype=np.float64)
    _, col_ind, _ = megalap.linear_sum_assignment(cost)
    assignment[point_ids] = target_ids[np.asarray(col_ind, dtype=np.int64)]


def recursive_seed(
    points: np.ndarray,
    target_points: np.ndarray,
    width: int,
    height: int,
    leaf_size: int = 8,
) -> np.ndarray:
    n = width * height
    assignment = np.empty(n, dtype=np.int64)

    def recurse(point_ids: np.ndarray, row0: int, rows: int, col0: int, cols: int) -> None:
        if rows <= leaf_size and cols <= leaf_size:
            target_rows = np.arange(row0, row0 + rows, dtype=np.int64)
            target_cols = np.arange(col0, col0 + cols, dtype=np.int64)
            grid_rows, grid_cols = np.meshgrid(target_rows, target_cols, indexing="ij")
            target_ids = (grid_rows * width + grid_cols).reshape(-1)
            solve_leaf(points, target_points, point_ids, target_ids, assignment)
            return

        if cols >= rows and cols > leaf_size:
            left_cols = cols // 2
            split_len = rows * left_cols
            order = np.lexsort((points[point_ids, 1], points[point_ids, 0]))
            sorted_ids = point_ids[order]
            recurse(sorted_ids[:split_len], row0, rows, col0, left_cols)
            recurse(sorted_ids[split_len:], row0, rows, col0 + left_cols, cols - left_cols)
            return

        top_rows = rows // 2
        split_len = top_rows * cols
        order = np.lexsort((points[point_ids, 0], points[point_ids, 1]))
        sorted_ids = point_ids[order]
        recurse(sorted_ids[:split_len], row0, top_rows, col0, cols)
        recurse(sorted_ids[split_len:], row0 + top_rows, rows - top_rows, col0, cols)

    recurse(np.arange(n, dtype=np.int64), 0, height, 0, width)
    return assignment


def lab_to_srgb(points: np.ndarray) -> np.ndarray:
    l = np.full(points.shape[0], 72.0, dtype=np.float64)
    a = (points[:, 0] * 2.0 - 1.0) * 80.0
    b = (points[:, 1] * 2.0 - 1.0) * 80.0

    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def invf(t: np.ndarray) -> np.ndarray:
        t3 = t * t * t
        return np.where(t3 > epsilon, t3, (116.0 * t - 16.0) / kappa)

    x = 0.95047 * invf(fx)
    y = invf(fy)
    z = 1.08883 * invf(fz)

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb_lin = np.clip(np.column_stack([r_lin, g_lin, b_lin]), 0.0, 1.0)

    threshold = 0.0031308
    rgb = np.where(
        rgb_lin <= threshold,
        12.92 * rgb_lin,
        1.055 * np.power(rgb_lin, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.round(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def render_points(
    source_points: np.ndarray,
    dest_points: np.ndarray,
    width: int,
    height: int,
    interp: float,
    colors: np.ndarray,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    interp_points = source_points + interp * (dest_points - source_points)

    px = np.rint(interp_points[:, 0] * (width - 1)).astype(np.int32)
    py = np.rint((1.0 - interp_points[:, 1]) * (height - 1)).astype(np.int32)
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    order = np.random.default_rng(0).permutation(interp_points.shape[0])
    canvas[py[order], px[order]] = colors[order]
    return canvas


def render_triptych(
    source_points: np.ndarray,
    dest_points: np.ndarray,
    panel_width: int,
    panel_height: int,
    mid_interp: float,
) -> np.ndarray:
    colors = lab_to_srgb(source_points)
    panels = [
        render_points(source_points, dest_points, panel_width, panel_height, 0.0, colors),
        render_points(source_points, dest_points, panel_width, panel_height, mid_interp, colors),
        render_points(source_points, dest_points, panel_width, panel_height, 1.0, colors),
    ]
    return np.concatenate(panels, axis=1)


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", crc)
    )


def write_png(path: pathlib.Path, rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be a uint8 array of shape (height, width, 3)")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a megalap showcase PNG without matplotlib.")
    parser.add_argument("--grid-width", type=int, default=512)
    parser.add_argument("--grid-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--leaf-size", type=int, default=8)
    parser.add_argument("--cleanup-seconds", type=float, default=30.0)
    parser.add_argument("--mid-interp", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "assets" / "showcase_triptych_512.png",
    )
    args = parser.parse_args()

    n = args.grid_width * args.grid_height
    points = make_meandering_points(n, seed=args.seed)
    target_points = build_target_grid(args.grid_width, args.grid_height, args.margin)
    assignment = recursive_seed(
        points,
        target_points,
        args.grid_width,
        args.grid_height,
        leaf_size=args.leaf_size,
    )

    if args.cleanup_seconds > 0.0:
        cleanup = megalap.window_cleanup(
            points,
            assignment,
            rows=args.grid_height,
            cols=args.grid_width,
            budget_seconds=args.cleanup_seconds,
            window_size=6,
            margin=args.margin,
            num_threads=None if args.num_threads == 0 else args.num_threads,
        )
        assignment = np.asarray(cleanup["assignment"], dtype=np.int64)
        print(
            f"cleanup passes={cleanup['passes_completed']} "
            f"elapsed_s={cleanup['elapsed_s']:.3f} "
            f"final_cost={cleanup['final_cost']:.6f}"
        )

    dest_points = target_points[assignment]
    rgb = render_triptych(
        points,
        dest_points,
        args.image_width,
        args.image_height,
        args.mid_interp,
    )
    write_png(args.output, rgb)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
