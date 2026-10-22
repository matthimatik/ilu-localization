#!/usr/bin/env python3
"""Generate an example pipeline config for a synthetic scene of ../ilu-pose-estimation.

    uv run python tools/make_example_config.py v3_drone_137
    uv run python tools/make_example_config.py v3_human_60 --out /tmp/example.yaml

The camera of a scene is read from its ``meta.json`` (camera_locations / camera_rotations, Blender
XYZ Euler degrees); the intrinsics are the repo's 4K camera (50 mm on a 36 mm sensor) scaled to the
image size. The generated config runs the real backends (pose, instance segmentation, OCR) and the
identify stage, like the other files in ``configs/examples``.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET = REPO_ROOT / ".." / "ilu-pose-estimation" / "datasets" / "synthetic-data-generation"
DEFAULT_OUT_DIR = REPO_ROOT / "configs" / "examples"
FOCAL_MM = 50.0
SENSOR_MM = 36.0

TEMPLATE = """pipeline: {pipeline}
stages:
  - name: load_image
    impl: load_image
    cfg:
      image: {image}
      camera:
        intrinsics:
          fx: {fx}
          fy: {fy}
          cx: {cx}
          cy: {cy}
          width: {width}
          height: {height}
        extrinsics:
          position_m: {position}
          rotation_euler_deg: {rotation}
  - name: pose
    impl: pose
    inputs:
      image: load_image/image_ref
    cfg:
      backend_dir: ../ilu-pose-estimation
      method: {method}
      min_corners: {min_corners}
  - name: instanceseg
    impl: instanceseg
    inputs:
      image: load_image/image_ref
    cfg:
      backend_dir: ../ilu-detection
  - name: textspots
    impl: mmocr
    inputs:
      image: load_image/image_ref
    cfg:
      backend_dir: ../mmocr
  - name: embeddings
    impl: fixture_embeddings
    inputs:
      image: load_image/image_ref
    cfg:
      source: tests/fixtures/embeddings_empty.json
  - name: associate
    impl: associate
    inputs:
      text_spots: textspots/text_spots
      pose_boxes: pose/pose_boxes
      embeddings: embeddings/embeddings
  - name: identify
    impl: identify
    inputs:
      associations: associate/associations
      text_spots: textspots/text_spots
      pose_boxes: pose/pose_boxes
    cfg:
      backend_dir: ../ilu-identification
  - name: fuse
    impl: fuse
    inputs:
      associations: associate/associations
      text_spots: textspots/text_spots
      pose_boxes: pose/pose_boxes
      embeddings: embeddings/embeddings
      image_ref: load_image/image_ref
      ilu_ids: identify/ilu_ids
"""


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image file."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img.shape[1], img.shape[0]


def config_text(scene: str, dataset_dir: Path, method: str, min_corners: int, pipeline: str) -> str:
    """The config for one scene, with the camera taken from the scene's meta.json."""
    scene_dir = dataset_dir / scene
    meta_path = scene_dir / "meta.json"
    image_path = scene_dir / "rgb.jpg"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no meta.json for scene {scene}: {meta_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"no image for scene {scene}: {image_path}")
    meta = json.loads(meta_path.read_text())
    width, height = image_size(image_path)
    focal = FOCAL_MM / SENSOR_MM * width
    position = [float(v) for v in meta["camera_locations"][0]]
    rotation = [float(v) for v in meta["camera_rotations"][0]]
    return TEMPLATE.format(
        pipeline=pipeline,
        image=f"../ilu-pose-estimation/datasets/synthetic-data-generation/{scene}/rgb.jpg",
        fx=repr(focal),
        fy=repr(focal),
        cx=repr(width / 2),
        cy=repr(height / 2),
        width=width,
        height=height,
        position=f"[{position[0]}, {position[1]}, {position[2]}]",
        rotation=f"[{rotation[0]}, {rotation[1]}, {rotation[2]}]",
        method=method,
        min_corners=min_corners,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scene', help="scene folder name, e.g. v3_drone_137")
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET,
                        help=f"scene root (default {DEFAULT_DATASET})")
    parser.add_argument('--out', type=Path, default=None,
                        help="output file (default configs/examples/backends_<scene>.yaml)")
    parser.add_argument('--method', default='upright-tier', help="PnP method of the pose backend")
    parser.add_argument('--min-corners', type=int, default=2, help="minimum visible corners per box")
    parser.add_argument('--pipeline', default='backends-example', help="pipeline name in the config")
    args = parser.parse_args()

    try:
        text = config_text(args.scene, args.dataset_dir, args.method, args.min_corners, args.pipeline)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    out = args.out or DEFAULT_OUT_DIR / f"backends_{args.scene}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    # The generated config must load like a hand-written one.
    from orchestrator.config import build_pipeline

    pipeline = build_pipeline(out, REPO_ROOT)
    pose = next(s for s in pipeline.specs if s.name == 'pose')
    print(f"{out}: {pipeline.name}, pose method {pose.cfg['method']}, "
          f"min_corners {pose.cfg['min_corners']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
