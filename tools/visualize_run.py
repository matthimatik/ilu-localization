#!/usr/bin/env python3
"""Visualize the end result of a completed pipeline run.

Writes into <run>/viz/ (override with --out):
    overlay.jpg   annotated image (masks, instances, keypoints, OCR quads,
                  id badges) + full detail table below, keyed by badge id
    scene.html    interactive 3D world scene incl. text spots, when available
    report.html   summary counts, record table, inlined overlay, scene link

Usage:
    uv run python tools/visualize_run.py                        # newest successful run
    uv run python tools/visualize_run.py --run-id 20261017-153634-653302
    uv run python tools/visualize_run.py --labels all --spots all
    uv run python tools/visualize_run.py --out /tmp/viz --open
"""

import argparse
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2

from core.visualization import (
    build_overlay,
    build_report,
    find_latest_run,
    load_run,
    plot_scene,
    write_html,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize the end result of a pipeline run",
    )
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--run-id", default=None, help="default: newest successful run")
    parser.add_argument("--out", default=None, help="default: <run>/viz")
    parser.add_argument(
        "--labels",
        choices=["ids", "all", "none"],
        default="ids",
        help="on-image record labels: id badges (default), full blocks, or none",
    )
    parser.add_argument(
        "--spots",
        choices=["associated", "all", "none"],
        default="associated",
        help="OCR quads: associated only (default), all, or none",
    )
    parser.add_argument(
        "--seg",
        choices=["on", "none"],
        default="on",
        help="instance-segmentation mask layer: on (default) or none",
    )
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--open", action="store_true", help="open report.html in a browser")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.run_id if args.run_id else find_latest_run(runs_root)
    if not run_dir.is_dir():
        raise SystemExit(f"run not found: {run_dir}")
    out_dir = Path(args.out) if args.out else run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_run(run_dir, base_dir=REPO_ROOT)
    image = cv2.imread(str(artifacts.image_path))
    if image is None:
        raise SystemExit(f"failed to read image: {artifacts.image_path}")

    title = f"ILU localization · {run_dir.name} · {artifacts.image_path.name}"
    overlay = build_overlay(
        image,
        artifacts.records,
        text_spots=artifacts.text_spots,
        associations=artifacts.associations,
        instances=artifacts.instances if args.seg != "none" else None,
        title=title,
        label_mode=args.labels,
        spot_mode=args.spots,
    )
    overlay_path = out_dir / "overlay.jpg"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise SystemExit(f"failed to write {overlay_path}")
    print(f"overlay: {overlay_path}")

    links: dict[str, str] = {}
    if not args.no_scene:
        fig = plot_scene(
            artifacts.records,
            artifacts.world_by_id,
            artifacts.camera,
            title=title,
            text_spots=artifacts.text_spots,
            associations=artifacts.associations,
        )
        if fig is not None:
            scene_path = write_html(fig, out_dir / "scene.html")
            links["3D scene"] = scene_path.name
            print(f"scene: {scene_path}")
        else:
            print("scene: skipped (no world corners or box centres to plot)")

    if not args.no_report:
        ok, jpeg = cv2.imencode(".jpg", overlay)
        report_path = write_report(
            build_report(
                artifacts.records,
                overlay_jpeg=jpeg.tobytes() if ok else None,
                links=links,
                associations=artifacts.associations,
                text_spots=artifacts.text_spots,
                title=title,
            ),
            out_dir / "report.html",
        )
        print(f"report: {report_path}")
        if args.open:
            webbrowser.open(report_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
