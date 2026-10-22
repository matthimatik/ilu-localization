#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orchestrator.stage import StageContext
from orchestrator.stages.pose import PoseBackendStage
from orchestrator.store import ArtifactStore
from schemas.image import CameraExtrinsics, CameraIntrinsics, CameraModel, ImageRef
from schemas.vision import PoseBoxes

DEFAULT_IMAGE = (
    REPO_ROOT
    / ".."
    / "ilu-pose-estimation"
    / "datasets"
    / "synthetic-data-generation"
    / "v3_drone_0"
    / "rgb.jpg"
)
DEFAULT_META = DEFAULT_IMAGE.parent / "meta.json"
DEFAULT_BACKEND = REPO_ROOT / ".." / "ilu-pose-estimation"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the pose backend adapter")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--backend-dir", default=str(DEFAULT_BACKEND))
    parser.add_argument("--method", default="upright-tier",
                        help="PnP method of the backend (upright-tier in the shipped configs)")
    parser.add_argument("--min-corners", type=int, default=2,
                        help="minimum visible corners per box (2 in the shipped configs)")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    ref = ImageRef(path=str(image_path))
    meta_path = Path(args.meta) if args.meta else None
    if meta_path is not None and meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        ref.camera = CameraModel(
            intrinsics=CameraIntrinsics(
                fx=50 / 36 * 3840,
                fy=50 / 36 * 3840,
                cx=1920.0,
                cy=1080.0,
                width=3840,
                height=2160,
            ),
            extrinsics=CameraExtrinsics(
                position_m=tuple(meta["camera_locations"][0]),
                rotation_euler_deg=tuple(meta["camera_rotations"][0]),
            ),
        )

    store = ArtifactStore(REPO_ROOT / "runs", "smoke-pose")
    store.write_model("input", "image_ref", ref)
    ctx = StageContext(
        store=store,
        stage_name="pose",
        cfg={"backend_dir": args.backend_dir, "method": args.method,
             "min_corners": args.min_corners, "_base_dir": str(REPO_ROOT)},
        inputs={"image": store.path_for("input", "image_ref")},
    )
    PoseBackendStage().run(ctx)
    boxes = store.read_model("pose", "pose_boxes", PoseBoxes)
    with_pnp = sum(1 for b in boxes.boxes if b.center_m is not None)
    print(f"{image_path.name}: {len(boxes.boxes)} pose boxes ({with_pnp} with world centres)")
    for box in boxes.boxes[:5]:
        print(
            f"  {box.box_id}: {box.size_class} conf={box.confidence:.2f} "
            f"method={box.method} center_m={box.center_m}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
