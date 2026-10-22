"""End-to-end evaluation of a completed run against synthetic-scene ground truth."""

from core.evaluation.api import evaluate
from core.evaluation.ground_truth import load_pose_jsonl, parse_iso_from_label
from core.evaluation.geometry import best_corner_errors, box_center, voxel_iou3d
from core.evaluation.matching import average_precision, greedy_match

__all__ = [
    "evaluate",
    "load_pose_jsonl",
    "parse_iso_from_label",
    "best_corner_errors",
    "box_center",
    "voxel_iou3d",
    "average_precision",
    "greedy_match",
]
