import numpy as np

from core.evaluation.geometry import best_corner_errors, box_center, voxel_iou3d
from core.evaluation.matching import average_precision, greedy_match
from core.evaluation.metrics import (
    accuracy_or_none,
    detection_prf,
    mean_or_none,
    median_or_none,
)
from schemas.evaluation import EvalReport, GTBoxes, MatchedPair, SceneMetrics
from schemas.ilu import ILURecord, ILURecords

DEFAULT_CENTER_THRESHOLD_M = 2.0
DEFAULT_VOXEL_M = 0.1
DEFAULT_AP_THRESHOLDS = (0.5, 1.0, 1.5, 2.0)


def _normalize_iso(code: str | None) -> str | None:
    if code is None:
        return None
    normalized = code.replace(" ", "").upper()
    return normalized or None


def evaluate(
    records: ILURecords,
    world_by_id: dict[str, np.ndarray],
    gt: GTBoxes,
    *,
    center_threshold_m: float = DEFAULT_CENTER_THRESHOLD_M,
    voxel_m: float = DEFAULT_VOXEL_M,
    ap_thresholds: tuple[float, ...] = DEFAULT_AP_THRESHOLDS,
    include_occluded_gt: bool = False,
) -> EvalReport:
    """End-to-end 3D localization + identification metrics for one scene.

    Predictions are the run's `ILURecord`s with PnP world corners (from
    `pose/raw`); GT boxes come from the scene's `pose.jsonl`. Matching is
    greedy one-to-one by ascending center distance up to `center_threshold_m`.
    """
    gt_boxes = [
        box
        for box in gt.boxes
        if include_occluded_gt or any(box.visible)
    ]

    preds: list[tuple[str, np.ndarray, ILURecord]] = []
    pred_ids_without_world: list[str] = []
    for record in records.records:
        world = world_by_id.get(record.ilu_id)
        if world is None or np.asarray(world).reshape(-1, 3).shape != (8, 3):
            pred_ids_without_world.append(record.ilu_id)
            continue
        preds.append((record.ilu_id, np.asarray(world, dtype=float).reshape(8, 3), record))

    n_gt = len(gt.boxes)
    n_gt_eval = len(gt_boxes)
    n_pred = len(records.records)
    n_pred_world = len(preds)

    if gt_boxes and preds:
        gt_centers = np.stack([box_center(np.asarray(b.world_corners)) for b in gt_boxes])
        pred_centers = np.stack([box_center(w) for _, w, _ in preds])
        center_error_matrix = np.linalg.norm(
            gt_centers[:, None, :] - pred_centers[None, :, :], axis=2
        )
    else:
        center_error_matrix = np.zeros((len(gt_boxes), len(preds)))

    matches = greedy_match(center_error_matrix, center_threshold_m)

    pairs: list[MatchedPair] = []
    errors_for_ap = np.full(n_pred, np.inf)
    confidences = np.zeros(n_pred)
    for j, (_, _, record) in enumerate(preds):
        confidences[j] = float(record.confidences.get("pose", 0.0))
    for i, j in matches:
        errors_for_ap[j] = center_error_matrix[i, j]

    gt_flags_id: list[bool | None] = []
    gt_flags_id_relaxed: list[bool | None] = []
    gt_flags_size: list[bool | None] = []
    center_errors: list[float] = []
    corner_means: list[float] = []

    for i, j in matches:
        gt_box = gt_boxes[i]
        pred_id, pred_world, record = preds[j]
        gt_world = np.asarray(gt_box.world_corners, dtype=float)
        _, corner_errors = best_corner_errors(pred_world, gt_world)
        center_error = float(center_error_matrix[i, j])
        iou = voxel_iou3d(pred_world, gt_world, voxel_m)

        gt_iso = _normalize_iso(gt_box.iso_code)
        pred_iso = _normalize_iso(record.iso_code)
        id_correct = (
            None if gt_iso is None or pred_iso is None else gt_iso == pred_iso
        )
        # ISO 6346: 4-letter owner + 4-digit serial + check digit (index -1).
        # Relaxed matching drops the check digit, so OCR check-digit errors
        # still count as identified.
        id_relaxed = (
            None
            if gt_iso is None or pred_iso is None or len(gt_iso) < 10 or len(pred_iso) < 10
            else gt_iso[:-1] == pred_iso[:-1]
        )
        gt_size = gt_box.size_class
        pred_size = record.size_class
        size_correct = None if gt_size is None else pred_size == gt_size

        gt_flags_id.append(id_correct)
        gt_flags_id_relaxed.append(id_relaxed)
        gt_flags_size.append(size_correct)
        center_errors.append(center_error)
        corner_means.append(float(corner_errors.mean()))

        pairs.append(
            MatchedPair(
                gt_index=gt_box.index,
                pred_ilu_id=pred_id,
                center_error_m=center_error,
                corner_errors_m=[float(e) for e in corner_errors],
                mean_corner_error_m=float(corner_errors.mean()),
                max_corner_error_m=float(corner_errors.max()),
                iou3d=iou,
                pose_confidence=confidences[j],
                gt_iso_code=gt_iso,
                pred_iso_code=pred_iso,
                id_correct=id_correct,
                id_relaxed_correct=id_relaxed,
                gt_size_class=gt_size,
                pred_size_class=pred_size,
                size_correct=size_correct,
            )
        )

    ap_values = [
        average_precision(errors_for_ap, confidences, t) for t in ap_thresholds
    ]
    ap_center = float(np.mean(ap_values)) if ap_values else None

    precision, recall, f1 = detection_prf(len(matches), n_pred_world, n_gt_eval)
    metrics = SceneMetrics(
        n_gt=n_gt,
        n_gt_visible=n_gt_eval,
        n_pred=n_pred,
        n_pred_with_world=n_pred_world,
        n_pred_without_world=len(pred_ids_without_world),
        n_matched=len(matches),
        center_threshold_m=center_threshold_m,
        precision=precision,
        recall=recall,
        f1=f1,
        ap_center=ap_center,
        mean_center_error_m=mean_or_none(center_errors),
        median_center_error_m=median_or_none(center_errors),
        mean_corner_error_m=mean_or_none(corner_means),
        id_accuracy=accuracy_or_none(gt_flags_id),
        id_relaxed_accuracy=accuracy_or_none(gt_flags_id_relaxed),
        size_accuracy=accuracy_or_none(gt_flags_size),
    )

    matched_gt = {i for i, _ in matches}
    matched_pred = {j for _, j in matches}
    return EvalReport(
        image_path=records.image_path,
        gt_source=gt.source,
        params={
            "center_threshold_m": center_threshold_m,
            "voxel_m": voxel_m,
            **{f"ap_threshold_{t}": t for t in ap_thresholds},
        },
        pairs=pairs,
        metrics=metrics,
        unmatched_gt_indices=[
            gt_boxes[i].index for i in range(n_gt_eval) if i not in matched_gt
        ],
        unmatched_pred_ids=[
            preds[j][0] for j in range(n_pred_world) if j not in matched_pred
        ],
        pred_ids_without_world=pred_ids_without_world,
    )
