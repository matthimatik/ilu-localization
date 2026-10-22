import itertools

import numpy as np

# Corner layout of an 8-corner box: the six quads below are planar faces for
# both known conventions — the generator's pose.jsonl (vertical-edge pairs
# 0-1, 2-3, 4-5, 6-7) and the pose backend's bottom/top layout (0-3 bottom,
# 4-7 top).
_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)

# All 8! corner correspondences (pred-index -> gt-index), evaluated at once.
# The GT generator and the pose backend number corners differently (including
# reflections), so the search must cover the full symmetric group — box
# symmetries alone are not enough.
_CORNER_PERMS = np.array(list(itertools.permutations(range(8))), dtype=int)


def box_center(corners: np.ndarray) -> np.ndarray:
    """Mean of the 8 corners, shape (3,)."""
    return np.asarray(corners, dtype=float).reshape(8, 3).mean(axis=0)


def best_corner_errors(pred: np.ndarray, gt: np.ndarray) -> tuple[int, np.ndarray]:
    """Corner-wise distances (8,) under the best of the 8! corner matchings.

    Returns `(perm_index, errors)` where `errors[j]` is the distance between
    the prediction corner mapped onto gt corner `j`.
    """
    pred = np.asarray(pred, dtype=float).reshape(8, 3)
    gt = np.asarray(gt, dtype=float).reshape(8, 3)
    dist = np.linalg.norm(pred[None, :, :] - gt[:, None, :], axis=2)  # (gt, pred)
    totals = dist[_CORNER_PERMS, np.arange(8)].sum(axis=1)
    best = int(np.argmin(totals))
    perm = _CORNER_PERMS[best]
    errors = np.empty(8)
    errors[perm] = dist[perm, np.arange(8)]
    return best, errors


def _inside_mask(points: np.ndarray, corners: np.ndarray, tol: float) -> np.ndarray:
    """Vectorised half-space test: True where `points` lie inside the box."""
    centroid = corners.mean(axis=0)
    inside = np.ones(len(points), dtype=bool)
    for face in _FACES:
        v0, v1, v2 = corners[list(face[:3])]
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        if float(normal @ centroid - normal @ v0) < 0:
            normal = -normal
        signed = points @ normal - float(normal @ v0)
        inside &= signed >= -tol
    return inside


def voxel_iou3d(pred: np.ndarray, gt: np.ndarray, voxel_m: float = 0.1) -> float:
    """IoU of two 8-corner boxes on a voxel grid (convex half-space tests)."""
    pred = np.asarray(pred, dtype=float).reshape(8, 3)
    gt = np.asarray(gt, dtype=float).reshape(8, 3)
    lower = np.minimum(pred.min(axis=0), gt.min(axis=0))
    upper = np.maximum(pred.max(axis=0), gt.max(axis=0))
    if np.any(upper - lower <= 0):
        return 0.0
    steps = np.ceil((upper - lower) / voxel_m).astype(int)
    if int(np.prod(steps)) > 2_000_000:
        return float("nan")
    xs = lower[0] + (np.arange(steps[0]) + 0.5) * voxel_m
    ys = lower[1] + (np.arange(steps[1]) + 0.5) * voxel_m
    zs = lower[2] + (np.arange(steps[2]) + 0.5) * voxel_m
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    tol = voxel_m * 0.5
    in_pred = _inside_mask(grid, pred, tol)
    in_gt = _inside_mask(grid, gt, tol)
    union = int(np.count_nonzero(in_pred | in_gt))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(in_pred & in_gt) / union)
