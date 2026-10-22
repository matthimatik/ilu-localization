import os
from pathlib import Path

import pytest

from orchestrator.stage import StageContext
from orchestrator.stages.pose import PoseBackendStage, pose_boxes_from_backend
from orchestrator.store import ArtifactStore
from schemas.image import CameraExtrinsics, CameraIntrinsics, CameraModel, ImageRef
from schemas.vision import PoseBoxes

REPO_ROOT = Path(__file__).resolve().parents[1]
DRONE_IMAGE = (
    REPO_ROOT
    / ".."
    / "ilu-pose-estimation"
    / "datasets"
    / "synthetic-data-generation"
    / "v3_drone_0"
    / "rgb.jpg"
)
DRONE_META = DRONE_IMAGE.parent / "meta.json"

CANNED_DET = {
    "image": "img.png",
    "size": [3840, 2160],
    "weights": "w.pt",
    "conf": 0.25,
    "kpt_conf": 0.5,
    "n_type_hints": 0,
    "n_type_hints_given": 0,
    "detections": [
        {
            "cls": "container",
            "conf": 0.9,
            "uv": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0],
                   [1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]],
            "kpt_conf": [0.9] * 8,
            "visible": [True, True, True, True, True, True, True, False],
            "sizes": ["20", "40"],
            "type_hint": None,
        },
        {
            "cls": "trailer",
            "conf": 0.8,
            "uv": [[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0],
                   [21.0, 21.0], [29.0, 21.0], [29.0, 29.0], [21.0, 29.0]],
            "kpt_conf": [0.8] * 8,
            "visible": [True] * 8,
            "sizes": ["trailer"],
            "type_hint": None,
        },
    ],
}

CANNED_BOXES = {
    "image": "img.png",
    "method": "pnp",
    "min_corners": 4,
    "camera": {"r_cw": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "cam": [0, 0, 10],
               "K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "image_size": [3840, 2160]},
    "boxes": [
        {
            "cls": "container",
            "size": "40",
            "type_hint": None,
            "n_corners": 7,
            "rms": 1.5,
            "tilt": 0.2,
            "uv": CANNED_DET["detections"][0]["uv"],
            "visible": CANNED_DET["detections"][0]["visible"],
            "world": [[0.0, 0.0, 0.0], [12.0, 0.0, 0.0], [12.0, 2.4, 0.0],
                      [0.0, 2.4, 0.0], [0.0, 0.0, 2.6], [12.0, 0.0, 2.6],
                      [12.0, 2.4, 2.6], [0.0, 2.4, 2.6]],
        }
    ],
}


def test_parse_detections_only():
    result = pose_boxes_from_backend(CANNED_DET, None, "img.png")
    assert len(result.boxes) == 2
    first, second = result.boxes
    assert first.box_id == "det-0"
    assert first.size_class == "unknown"
    assert first.method == "yolo"
    assert first.center_m is None
    assert first.corners_px[7].position_px is None
    assert second.size_class == "trailer"
    assert second.confidence == pytest.approx(0.8)


def test_parse_with_pnp_boxes():
    result = pose_boxes_from_backend(CANNED_DET, CANNED_BOXES, "img.png")
    first, second = result.boxes
    assert first.size_class == "40"
    assert first.method == "pnp"
    assert first.center_m == pytest.approx((6.0, 1.2, 1.3))
    assert second.center_m is None
    assert second.method == "yolo"


@pytest.mark.skipif(
    os.environ.get("RUN_BACKEND_TESTS") != "1",
    reason="needs GPU + pose backend (set RUN_BACKEND_TESTS=1)",
)
def test_pose_backend_on_drone_scene(tmp_path):
    meta = DRONE_META.read_text()
    import json

    raw = json.loads(meta)
    ref = ImageRef(
        path=str(DRONE_IMAGE),
        camera=CameraModel(
            intrinsics=CameraIntrinsics(
                fx=50 / 36 * 3840,
                fy=50 / 36 * 3840,
                cx=1920.0,
                cy=1080.0,
                width=3840,
                height=2160,
            ),
            extrinsics=CameraExtrinsics(
                position_m=tuple(raw["camera_locations"][0]),
                rotation_euler_deg=tuple(raw["camera_rotations"][0]),
            ),
        ),
    )
    store = ArtifactStore(tmp_path / "runs", "backend-test")
    store.write_model("input", "image_ref", ref)
    ctx = StageContext(
        store=store,
        stage_name="pose",
        cfg={
            "backend_dir": str(REPO_ROOT / ".." / "ilu-pose-estimation"),
            "method": "upright-tier",
            "min_corners": 2,
            "_base_dir": str(REPO_ROOT),
        },
        inputs={"image": store.path_for("input", "image_ref")},
    )
    PoseBackendStage().run(ctx)
    boxes = store.read_model("pose", "pose_boxes", PoseBoxes)
    assert len(boxes.boxes) > 0
    with_world = [b for b in boxes.boxes if b.center_m is not None]
    assert len(with_world) > 0
    assert all(b.size_class in {"20", "40", "trailer", "unknown"} for b in boxes.boxes)
