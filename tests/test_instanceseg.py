import os
from pathlib import Path

import pytest

from orchestrator.registry import create_stage
from orchestrator.stage import StageContext
from orchestrator.stages.instanceseg import InstancesegBackendStage, instances_from_runner
from orchestrator.store import ArtifactStore
from schemas.image import ImageRef
from schemas.vision import Instances

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

CANNED_RUNNER = {
    "image": "img.jpg",
    "size": [3840, 2160],
    "weights": "models/ilu-seg-yolov8m-1024.pt",
    "imgsz": 1024,
    "conf": 0.25,
    "names": {"0": "container", "1": "tank_container", "2": "trailer"},
    "instances": [
        {
            "cls_id": 0,
            "cls": "container",
            "conf": 0.97,
            "bbox_xyxy": [712.3, 21.0, 1814.4, 731.8],
            "polygon": [[712.5, 21.0], [1814.0, 21.5], [1813.5, 731.0]],
        },
        {
            "cls_id": 2,
            "cls": "trailer",
            "conf": 0.55,
            "bbox_xyxy": [100.0, 200.0, 900.0, 800.0],
            "polygon": [[100.0, 200.0], [900.0, 200.0], [900.0, 800.0], [100.0, 800.0]],
        },
    ],
}


def test_registry_has_instanceseg():
    stage = create_stage("instanceseg")
    assert isinstance(stage, InstancesegBackendStage)
    assert stage.name == "instanceseg"


def test_instances_from_runner():
    result = instances_from_runner(CANNED_RUNNER, "img.jpg")
    assert isinstance(result, Instances)
    assert result.image_path == "img.jpg"
    assert len(result.instances) == 2
    first, second = result.instances
    assert first.instance_id == "seg-0"
    assert first.label == "container"
    assert first.confidence == pytest.approx(0.97)
    assert first.bbox_px == pytest.approx((712.3, 21.0, 1814.4, 731.8))
    assert len(first.polygon_px) == 3
    assert first.polygon_px[0] == pytest.approx((712.5, 21.0))
    assert second.label == "trailer"


def test_instances_from_runner_empty():
    result = instances_from_runner({**CANNED_RUNNER, "instances": []}, "img.jpg")
    assert result.instances == []


@pytest.mark.skipif(
    os.environ.get("RUN_BACKEND_TESTS") != "1",
    reason="needs GPU + ilu-detection backend (set RUN_BACKEND_TESTS=1)",
)
def test_instanceseg_backend_on_drone_scene(tmp_path):
    store = ArtifactStore(tmp_path / "runs", "backend-test")
    store.write_model("input", "image_ref", ImageRef(path=str(DRONE_IMAGE)))
    ctx = StageContext(
        store=store,
        stage_name="instanceseg",
        cfg={
            "backend_dir": str(REPO_ROOT / ".." / "ilu-detection"),
            "_base_dir": str(REPO_ROOT),
        },
        inputs={"image": store.path_for("input", "image_ref")},
    )
    InstancesegBackendStage().run(ctx)
    instances = store.read_model("instanceseg", "instances", Instances)
    assert len(instances.instances) > 0
    assert all(inst.label in {"container", "tank_container", "trailer"} for inst in instances.instances)
    assert all(0.0 <= inst.confidence <= 1.0 for inst in instances.instances)
    assert all(len(inst.polygon_px) >= 3 for inst in instances.instances)
