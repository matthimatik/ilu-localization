"""Load the JSON artifacts of a completed run for visualization."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orchestrator.stages.pose import match_pnp
from schemas.association import Associations
from schemas.ilu import ILURecords
from schemas.image import ImageRef
from schemas.vision import TextSpots


@dataclass
class RunArtifacts:
    run_dir: Path
    records: ILURecords
    image_path: Path
    text_spots: TextSpots | None = None
    associations: Associations | None = None
    world_by_id: dict[str, np.ndarray] = field(default_factory=dict)
    camera: dict | None = None


def _read_model(path: Path, model_cls):
    if not path.exists():
        return None
    return model_cls.model_validate_json(path.read_text())


def load_run(run_dir: Path | str, base_dir: Path | str) -> RunArtifacts:
    """Read a run's artifacts. `base_dir` resolves relative image paths.

    World corners for the 3D scene come from the pose backend's raw output
    (`pose/raw/<stem>/detections.json` + `boxes.json`), which only exists for
    pose-backend runs; fixture runs load without them.
    """
    run_dir = Path(run_dir)
    base_dir = Path(base_dir)
    records_path = run_dir / "fuse" / "ilu_records.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"no fuse/ilu_records.json under {run_dir} — did the run complete?"
        )
    records = ILURecords.model_validate_json(records_path.read_text())

    image_path = records.image_path
    if not image_path:
        ref = _read_model(run_dir / "load_image" / "image_ref.json", ImageRef)
        image_path = ref.path if ref is not None else ""
    resolved = Path(image_path)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"image not found: {resolved}")

    world_by_id, camera = _load_pose_raw(run_dir)
    return RunArtifacts(
        run_dir=run_dir,
        records=records,
        image_path=resolved,
        text_spots=_read_model(run_dir / "textspots" / "text_spots.json", TextSpots),
        associations=_read_model(
            run_dir / "associate" / "associations.json", Associations
        ),
        world_by_id=world_by_id,
        camera=camera,
    )


def _load_pose_raw(run_dir: Path) -> tuple[dict[str, np.ndarray], dict | None]:
    """Map `det-N` ids to PnP world corners via the same uv match as the stage."""
    world_by_id: dict[str, np.ndarray] = {}
    camera: dict | None = None
    for det_path in sorted(run_dir.glob("pose/raw/*/detections.json")):
        boxes_path = det_path.with_name("boxes.json")
        if not boxes_path.exists():
            continue
        detections = json.loads(det_path.read_text())
        boxes_json = json.loads(boxes_path.read_text())
        pnp_boxes = boxes_json.get("boxes") or []
        if camera is None and boxes_json.get("camera"):
            camera = boxes_json["camera"]
        for index, det in enumerate(detections.get("detections", [])):
            match = match_pnp(det, pnp_boxes)
            if match is not None and match.get("world"):
                world_by_id.setdefault(
                    f"det-{index}", np.asarray(match["world"], dtype=float)
                )
    return world_by_id, camera


def find_latest_run(runs_root: Path | str) -> Path:
    """Newest successfully finished run directory under `runs_root`."""
    runs_root = Path(runs_root)
    candidates: list[Path] = []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        if manifest.get("status") == "success":
            candidates.append(manifest_path.parent)
    if not candidates:
        raise FileNotFoundError(
            f"no successful run under {runs_root} — run tools/run_pipeline.py first"
        )
    return max(candidates, key=lambda p: p.name)
