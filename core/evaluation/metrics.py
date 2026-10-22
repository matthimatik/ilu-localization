import numpy as np


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def accuracy_or_none(flags: list[bool | None]) -> float | None:
    """Mean of the non-None flags; None when nothing is comparable."""
    known = [flag for flag in flags if flag is not None]
    return float(np.mean(known)) if known else None


def detection_prf(
    n_matched: int, n_pred: int, n_gt: int
) -> tuple[float, float, float]:
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1
