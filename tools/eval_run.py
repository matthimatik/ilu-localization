#!/usr/bin/env python3
"""Evaluate a completed pipeline run against synthetic-scene ground truth.

Metrics (world 3D, on `fuse/ilu_records.json` + `pose/raw` PnP corners):
    matching     greedy one-to-one by ascending center distance
    detection    precision / recall / F1 at the center threshold
    ranking      AP over center-distance thresholds (pose confidence ranked)
    quality      mean/median center error, mean corner error (best of all
                 8! corner matchings, so GT/pred corner orderings need not
                 agree), voxel IoU
    identification  ISO 6346 exact / check-digit-relaxed accuracy and size
                 accuracy, scored on matched pairs only

Writes <run>/eval/eval_report.json and prints a summary.

Usage:
    uv run python tools/eval_run.py                              # newest run
    uv run python tools/eval_run.py --run-id 20261017-153634-653302
    uv run python tools/eval_run.py --gt path/to/pose.jsonl
    uv run python tools/eval_run.py --center-thres-m 1.5 --voxel-m 0.05
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.evaluation import evaluate, load_pose_jsonl
from core.visualization import find_latest_run, load_pose_raw, load_run


def _default_gt_path(image_path: Path) -> Path | None:
    candidate = image_path.parent / "pose.jsonl"
    return candidate if candidate.exists() else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _print_summary(report) -> None:
    m = report.metrics
    print(f"\n=== {report.run_id or report.image_path} ===")
    print(
        f"gt {m.n_gt} ({m.n_gt_visible} visible) · "
        f"pred {m.n_pred} ({m.n_pred_with_world} with world, "
        f"{m.n_pred_without_world} without)"
    )
    print(f"matched {m.n_matched} @ center<={m.center_threshold_m:.1f} m")
    print(
        f"precision {m.precision:.3f} · recall {m.recall:.3f} · f1 {m.f1:.3f}"
        + (f" · ap_center {_fmt(m.ap_center)}" if m.ap_center is not None else "")
    )
    print(
        f"center err mean/median "
        f"{_fmt(m.mean_center_error_m)}/{_fmt(m.median_center_error_m)} m · "
        f"corner err mean {_fmt(m.mean_corner_error_m)} m"
    )
    print(
        f"id exact {_fmt(m.id_accuracy)} · "
        f"id relaxed {_fmt(m.id_relaxed_accuracy)} · "
        f"size {_fmt(m.size_accuracy)} "
        "(on matched pairs)"
    )
    if report.pairs:
        print("\n  gt#  pred      center_m  corner_m  iou3d   id     size  conf")
        for pair in report.pairs:
            id_mark = {True: "ok", False: "X", None: "-"}[pair.id_correct]
            size_mark = {True: "ok", False: "X", None: "-"}[pair.size_correct]
            print(
                f"  {pair.gt_index:>3}  {pair.pred_ilu_id:<8} "
                f"{pair.center_error_m:>8.3f}  {pair.mean_corner_error_m:>8.3f}  "
                f"{_fmt(pair.iou3d, 2):>5}  {id_mark:<4} {size_mark:<4} "
                f"{_fmt(pair.pose_confidence, 2)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a completed run against pose.jsonl ground truth",
    )
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--run-id", default=None, help="default: newest successful run")
    parser.add_argument(
        "--gt",
        default=None,
        help="path to pose.jsonl or a scene dir containing it; "
        "default: pose.jsonl next to the run's image",
    )
    parser.add_argument(
        "--center-thres-m",
        type=float,
        default=2.0,
        help="match / recall threshold on center error in meters (default 2.0)",
    )
    parser.add_argument(
        "--voxel-m",
        type=float,
        default=0.1,
        help="voxel size for 3D IoU in meters (default 0.1)",
    )
    parser.add_argument(
        "--include-fully-occluded-gt",
        action="store_true",
        help="also count GT boxes with no visible corners (default: skip)",
    )
    parser.add_argument(
        "--out", default=None, help="default: <run>/eval/eval_report.json"
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.run_id if args.run_id else find_latest_run(runs_root)
    if not run_dir.is_dir():
        raise SystemExit(f"run not found: {run_dir}")

    artifacts = load_run(run_dir, base_dir=REPO_ROOT)
    if not artifacts.world_by_id:
        raise SystemExit(
            f"no PnP world corners under {run_dir}/pose/raw — "
            "3D eval needs a pose-backend run (fixture runs have no world boxes)"
        )

    if args.gt:
        gt_path = Path(args.gt)
        if gt_path.is_dir():
            gt_path = gt_path / "pose.jsonl"
        if not gt_path.exists():
            raise SystemExit(f"ground truth not found: {gt_path}")
    else:
        gt_path = _default_gt_path(artifacts.image_path)
        if gt_path is None:
            raise SystemExit(
                f"no pose.jsonl next to {artifacts.image_path} — pass --gt"
            )
    gt = load_pose_jsonl(gt_path)

    report = evaluate(
        artifacts.records,
        artifacts.world_by_id,
        gt,
        center_threshold_m=args.center_thres_m,
        voxel_m=args.voxel_m,
        include_occluded_gt=args.include_fully_occluded_gt,
    )
    report.run_id = run_dir.name

    out_path = Path(args.out) if args.out else run_dir / "eval" / "eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2))
    print(f"report: {out_path}")

    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
