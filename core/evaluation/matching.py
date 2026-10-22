import numpy as np


def greedy_match(
    center_errors: np.ndarray, threshold: float
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by ascending center error.

    `center_errors[i, j]` is the distance between gt `i` and prediction `j`.
    Pairs are accepted in ascending order while both sides are free and the
    error is at most `threshold`. Returns `[(gt_index, pred_index), ...]`.
    """
    center_errors = np.asarray(center_errors, dtype=float)
    if center_errors.size == 0:
        return []
    order = np.argsort(center_errors, axis=None)
    gt_taken = np.zeros(center_errors.shape[0], dtype=bool)
    pred_taken = np.zeros(center_errors.shape[1], dtype=bool)
    pairs: list[tuple[int, int]] = []
    for flat in order:
        i, j = np.unravel_index(flat, center_errors.shape)
        if center_errors[i, j] > threshold:
            break
        if gt_taken[i] or pred_taken[j]:
            continue
        gt_taken[i] = True
        pred_taken[j] = True
        pairs.append((int(i), int(j)))
    return sorted(pairs)


def average_precision(
    errors: np.ndarray, confidences: np.ndarray, threshold: float
) -> float:
    """All-point-interpolated AP over a single error threshold.

    A prediction is a true positive when its (matching) error is at most
    `threshold`; ranking is by descending confidence. Unmatched predictions
    carry `inf` error and count as false positives.
    """
    errors = np.asarray(errors, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    n_pos = int(np.count_nonzero(np.isfinite(errors) & (errors <= threshold)))
    if n_pos == 0:
        return 0.0
    if len(errors) == 0:
        return 0.0
    order = np.argsort(-confidences, kind="stable")
    sorted_errors = errors[order]
    tp = np.cumsum(sorted_errors <= threshold)
    fp = np.cumsum(sorted_errors > threshold)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = 0.0
    for k in range(len(tp) - 1, -1, -1):
        if sorted_errors[k] <= threshold:
            ap += (recall[k] - (recall[k - 1] if k > 0 else 0.0)) * precision[k]
    return float(ap)
