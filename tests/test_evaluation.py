import numpy as np

from core.evaluation import (
    average_precision,
    best_corner_errors,
    box_center,
    evaluate,
    greedy_match,
    load_pose_jsonl,
    parse_iso_from_label,
    voxel_iou3d,
)
from schemas.evaluation import GTBox, GTBoxes
from schemas.ilu import (
    IdentificationMethod,
    ILURecord,
    ILURecords,
)
from schemas.vision import Corner, PoseBox


def make_world_box(
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    length: float = 6.0,
    width: float = 2.44,
    height: float = 2.6,
) -> np.ndarray:
    """Axis-aligned box, corners 0-3 bottom then 4-7 top (layout of _FACES)."""
    x0, x1 = cx - length / 2, cx + length / 2
    y0, y1 = cy - width / 2, cy + width / 2
    bottom = [(x0, y0, cz), (x1, y0, cz), (x1, y1, cz), (x0, y1, cz)]
    top = [(x, y, cz + height) for x, y, _ in bottom]
    return np.asarray(bottom + top, dtype=float)


def make_gt(
    index: int = 0,
    corners: np.ndarray | None = None,
    visible: bool = True,
    size_class: str | None = "20",
    label: str = "KLJU237271 3",
) -> GTBox:
    corners = make_world_box() if corners is None else corners
    return GTBox(
        index=index,
        world_corners=[tuple(map(float, c)) for c in corners],
        visible=[visible] * 8,
        size_class=size_class,
        label=label,
        iso_code=parse_iso_from_label(label),
    )


def make_record(
    ilu_id: str = "det-0",
    iso_code: str | None = "KLJU2372713",
    size_class: str | None = "20",
    pose_conf: float = 0.9,
) -> ILURecord:
    corners = [
        Corner(position_px=(float(i), float(i)), confidence=0.9) for i in range(8)
    ]
    return ILURecord(
        ilu_id=ilu_id,
        category="container",
        iso_code=iso_code,
        size_class=size_class,
        identification_method=(
            IdentificationMethod.code if iso_code else IdentificationMethod.unknown
        ),
        pose=PoseBox(
            box_id=ilu_id,
            corners_px=corners,
            size_class=size_class,
            method="pnp",
            confidence=pose_conf,
        ),
        confidences={"pose": pose_conf},
    )


def make_records(records: list[ILURecord]) -> ILURecords:
    return ILURecords(image_path="img.png", records=records)


def make_gt_boxes(boxes: list[GTBox], source: str = "pose.jsonl") -> GTBoxes:
    return GTBoxes(source=source, boxes=boxes)


def test_parse_iso_from_label():
    assert parse_iso_from_label("KLJU237271 3") == "KLJU2372713"
    assert parse_iso_from_label("BMOU2372713") == "BMOU2372713"
    assert parse_iso_from_label("") is None
    assert parse_iso_from_label("ugly") is None


def test_load_pose_jsonl(tmp_path):
    path = tmp_path / "pose.jsonl"
    corners = make_world_box()
    lines = []
    for label, visible_first in [("KLJU237271 3", True), ("IGQU575075 2", False)]:
        entries = []
        for i, corner in enumerate(corners):
            image_position = [100 + i, 200] if (visible_first or i > 0) else None
            entries.append(
                {
                    "world_position": [i + 1, [float(v) for v in corner]],
                    "image_position": image_position,
                }
            )
        lines.append(
            {"corners": entries, "model": "20", "label": label}
        )
    import json

    path.write_text("\n".join(json.dumps(row) for row in lines))

    gt = load_pose_jsonl(path)
    assert len(gt.boxes) == 2
    assert gt.boxes[0].iso_code == "KLJU2372713"
    assert gt.boxes[0].size_class == "20"
    assert all(gt.boxes[0].visible)
    assert not gt.boxes[1].visible[0]
    assert gt.boxes[1].visible[1]
    assert np.allclose(gt.boxes[0].world_corners, corners)


def test_greedy_match_threshold_and_one_to_one():
    errors = np.array([[0.1, 0.2], [1.5, 0.15]])
    pairs = greedy_match(errors, threshold=0.5)
    assert pairs == [(0, 0), (1, 1)]

    assert greedy_match(errors, threshold=0.12) == [(0, 0)]
    assert greedy_match(np.zeros((0, 2)), threshold=1.0) == []


def test_best_corner_errors_rotation_invariant():
    gt = make_world_box()
    # rotate corner numbering by 1 within each face
    rotation = [1, 2, 3, 0, 5, 6, 7, 4]
    rotated = gt[rotation]
    _, errors = best_corner_errors(rotated, gt)
    assert float(errors.max()) < 1e-9


def test_best_corner_errors_face_swap_invariant():
    gt = make_world_box()
    swapped = np.concatenate([gt[4:], gt[:4]])
    _, errors = best_corner_errors(swapped, gt)
    assert float(errors.max()) < 1e-9


def test_best_corner_errors_reflection_invariant():
    # pose.jsonl numbers corners as vertical-edge pairs with the opposite
    # winding to the backend; that correspondence is a reflection, outside
    # any cyclic rotation/face-swap group (seen on v3_reachstacker_50).
    gt = make_world_box()
    reflected = gt[[3, 2, 1, 0, 7, 6, 5, 4]]
    _, errors = best_corner_errors(reflected, gt)
    assert float(errors.max()) < 1e-9


def test_voxel_iou3d():
    box = make_world_box()
    assert voxel_iou3d(box, box) > 0.99
    shifted = make_world_box(cx=100.0)
    assert voxel_iou3d(box, shifted) == 0.0
    partial = make_world_box(cx=3.0)
    iou = voxel_iou3d(box, partial)
    assert 0.0 < iou < 1.0


def test_box_center():
    box = make_world_box(cx=2.0, cy=3.0, cz=4.0)
    assert np.allclose(box_center(box), [2.0, 3.0, 4.0 + 1.3])


def test_average_precision_perfect_ranking():
    errors = np.array([0.1, 0.2, np.inf])
    confidences = np.array([0.9, 0.8, 0.7])
    assert average_precision(errors, confidences, threshold=0.5) == 1.0
    assert average_precision(np.array([np.inf]), np.array([0.5]), 0.5) == 0.0


def test_evaluate_perfect_match():
    corners = make_world_box()
    gt = make_gt_boxes([make_gt(corners=corners)])
    records = make_records([make_record()])
    world = {"det-0": corners}

    report = evaluate(records, world, gt)
    m = report.metrics
    assert m.n_matched == 1
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.ap_center == 1.0
    assert m.mean_center_error_m < 1e-9
    assert m.mean_corner_error_m < 1e-9
    assert m.id_accuracy == 1.0
    assert m.id_relaxed_accuracy == 1.0
    assert m.size_accuracy == 1.0
    assert report.pairs[0].iou3d > 0.99
    assert report.unmatched_gt_indices == []
    assert report.unmatched_pred_ids == []


def test_evaluate_stacked_boxes_matched_by_z():
    bottom = make_world_box(cz=0.0)
    top = make_world_box(cz=2.6)
    gt = make_gt_boxes(
        [make_gt(index=0, corners=bottom), make_gt(index=1, corners=top)]
    )
    # predictions in reverse order and slightly perturbed
    world = {
        "det-0": top + 0.05,
        "det-1": bottom + 0.05,
    }
    records = make_records([make_record("det-0"), make_record("det-1")])

    report = evaluate(records, world, gt)
    assert report.metrics.n_matched == 2
    pairs = {p.pred_ilu_id: p.gt_index for p in report.pairs}
    assert pairs == {"det-0": 1, "det-1": 0}


def test_evaluate_records_without_world_are_excluded():
    gt = make_gt_boxes([make_gt()])
    records = make_records([make_record("det-0"), make_record("det-1")])
    world = {"det-0": make_world_box()}  # det-1 has no PnP corners

    report = evaluate(records, world, gt)
    m = report.metrics
    assert m.n_pred == 2
    assert m.n_pred_with_world == 1
    assert m.n_pred_without_world == 1
    assert report.pred_ids_without_world == ["det-1"]
    assert m.n_matched == 1
    assert m.recall == 1.0


def test_evaluate_id_relaxed_check_digit():
    corners = make_world_box()
    gt = make_gt_boxes([make_gt(corners=corners)])  # iso KLJU2372713
    records = make_records([make_record(iso_code="KLJU2372719")])  # wrong check
    world = {"det-0": corners}

    report = evaluate(records, world, gt)
    pair = report.pairs[0]
    assert pair.id_correct is False
    assert pair.id_relaxed_correct is True
    assert report.metrics.id_accuracy == 0.0
    assert report.metrics.id_relaxed_accuracy == 1.0


def test_evaluate_fully_occluded_gt_excluded_by_default():
    visible = make_gt(index=0, corners=make_world_box(cx=0.0), visible=True)
    occluded = make_gt(index=1, corners=make_world_box(cx=50.0), visible=False)
    gt = make_gt_boxes([visible, occluded])
    records = make_records([make_record()])
    world = {"det-0": make_world_box(cx=0.0)}

    report = evaluate(records, world, gt)
    assert report.metrics.n_gt == 2
    assert report.metrics.n_gt_visible == 1
    assert report.metrics.recall == 1.0

    report_all = evaluate(records, world, gt, include_occluded_gt=True)
    assert report_all.metrics.n_gt_visible == 2
    assert report_all.metrics.recall == 0.5
    assert report_all.unmatched_gt_indices == [1]


def test_evaluate_false_positive_lowers_precision():
    gt = make_gt_boxes([make_gt(corners=make_world_box(cx=0.0))])
    records = make_records([make_record("det-0"), make_record("det-1")])
    world = {
        "det-0": make_world_box(cx=0.0),
        "det-1": make_world_box(cx=30.0),  # far away: no match
    }

    report = evaluate(records, world, gt)
    assert report.metrics.n_matched == 1
    assert report.metrics.precision == 0.5
    assert report.metrics.recall == 1.0
    assert report.unmatched_pred_ids == ["det-1"]


def test_evaluate_missing_iso_sets_none():
    corners = make_world_box()
    gt = make_gt_boxes([make_gt(corners=corners, label="not-an-id")])
    records = make_records([make_record(iso_code=None)])
    world = {"det-0": corners}

    report = evaluate(records, world, gt)
    pair = report.pairs[0]
    assert pair.id_correct is None
    assert pair.gt_iso_code is None
    assert report.metrics.id_accuracy is None


def test_evaluate_size_mismatch():
    corners = make_world_box()
    gt = make_gt_boxes([make_gt(corners=corners, size_class="20")])
    records = make_records([make_record(size_class="40")])
    world = {"det-0": corners}

    report = evaluate(records, world, gt)
    assert report.pairs[0].size_correct is False
    assert report.metrics.size_accuracy == 0.0
