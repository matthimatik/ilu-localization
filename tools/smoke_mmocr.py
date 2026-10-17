#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from orchestrator.stage import StageContext
from orchestrator.stages.mmocr_backend import MmocrBackendStage
from orchestrator.store import ArtifactStore
from schemas.image import ImageRef
from schemas.vision import TextSpots

DEFAULT_IMAGE = (
    REPO_ROOT
    / ".."
    / "ilu-pose-estimation"
    / "datasets"
    / "synthetic-data-generation"
    / "v3_drone_0"
    / "rgb.jpg"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the mmocr backend adapter")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--backend-dir", default=str(REPO_ROOT / ".." / "mmocr"))
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    store = ArtifactStore(REPO_ROOT / "runs", "smoke-mmocr")
    store.write_model("input", "image_ref", ImageRef(path=str(image_path)))
    ctx = StageContext(
        store=store,
        stage_name="textspots",
        cfg={"backend_dir": args.backend_dir, "_base_dir": str(REPO_ROOT)},
        inputs={"image": store.path_for("input", "image_ref")},
    )
    MmocrBackendStage().run(ctx)
    spots = store.read_model("textspots", "text_spots", TextSpots)
    print(f"{image_path.name}: {len(spots.spots)} text spots")
    for spot in sorted(spots.spots, key=lambda s: -s.confidence)[:8]:
        print(f"  {spot.confidence:.2f} {spot.text!r} ({spot.text_type})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
