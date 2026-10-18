#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orchestrator.stage import StageContext
from orchestrator.stages.instanceseg import InstancesegBackendStage
from orchestrator.store import ArtifactStore
from schemas.image import ImageRef
from schemas.vision import Instances

DEFAULT_IMAGE = (
    REPO_ROOT
    / ".."
    / "ilu-pose-estimation"
    / "datasets"
    / "synthetic-data-generation"
    / "v3_drone_0"
    / "rgb.jpg"
)
DEFAULT_BACKEND = REPO_ROOT / ".." / "ilu-detection"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the instance-segmentation backend adapter")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--backend-dir", default=str(DEFAULT_BACKEND))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    ref = ImageRef(path=str(image_path))

    store = ArtifactStore(REPO_ROOT / "runs", "smoke-instanceseg")
    store.write_model("input", "image_ref", ref)
    ctx = StageContext(
        store=store,
        stage_name="instanceseg",
        cfg={
            "backend_dir": args.backend_dir,
            "conf": args.conf,
            "_base_dir": str(REPO_ROOT),
        },
        inputs={"image": store.path_for("input", "image_ref")},
    )
    InstancesegBackendStage().run(ctx)
    instances = store.read_model("instanceseg", "instances", Instances)
    counts: dict[str, int] = {}
    for inst in instances.instances:
        counts[inst.label] = counts.get(inst.label, 0) + 1
    print(f"{image_path.name}: {len(instances.instances)} instances {counts}")
    for inst in instances.instances[:5]:
        x1, y1, x2, y2 = inst.bbox_px
        print(
            f"  {inst.instance_id}: {inst.label} conf={inst.confidence:.2f} "
            f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) poly_pts={len(inst.polygon_px)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
