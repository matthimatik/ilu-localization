#!/usr/bin/env python3
"""Smoke-test the identify stage alone against the sibling ilu-identification repo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.stage import StageContext
from orchestrator.stages.identify import IdentifyStage
from orchestrator.store import ArtifactStore
from schemas.association import Association, Associations
from schemas.identification import IdentifiedILUs
from tests.builders import make_box, make_poseboxes, make_spot, make_textspots

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND = (REPO_ROOT / ".." / "ilu-identification").resolve()


def main() -> int:
    store = ArtifactStore(REPO_ROOT / "runs", "smoke-identify")
    spots = make_textspots(
        [
            make_spot(text="BMOU 237271 3", confidence=0.94),
            make_spot(text="45G1", center=(300, 100), confidence=0.87),
        ],
        image_path="smoke.png",
    )
    boxes = make_poseboxes([make_box(box_id="box-A", size_class="40")], image_path="smoke.png")
    assoc = Associations(
        associations=[Association(pose_box_id="box-A", text_spot_indices=[0, 1])]
    )
    store.write_model("smoke-assoc", "associations", assoc)
    store.write_model("smoke-spots", "text_spots", spots)
    store.write_model("smoke-boxes", "pose_boxes", boxes)

    ctx = StageContext(
        store=store,
        stage_name="identify",
        cfg={"backend_dir": str(BACKEND), "_base_dir": str(REPO_ROOT)},
        inputs={
            "associations": store.path_for("smoke-assoc", "associations"),
            "text_spots": store.path_for("smoke-spots", "text_spots"),
            "pose_boxes": store.path_for("smoke-boxes", "pose_boxes"),
        },
    )
    IdentifyStage().run(ctx)
    artifact = store.read_model("identify", "ilu_ids", IdentifiedILUs)
    for entry in artifact.results:
        st = entry.size_type.code if entry.size_type else None
        print(
            f"{entry.ilu_id}: {entry.id_code} ({entry.id_scheme}) "
            f"conf={entry.id_confidence:.3f} size_type={st} method={entry.method}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
