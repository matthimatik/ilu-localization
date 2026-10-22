import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from make_example_config import config_text  # noqa: E402

SCENE_ROOT = REPO_ROOT / ".." / "ilu-pose-estimation" / "datasets" / "synthetic-data-generation"


@pytest.mark.skipif(not SCENE_ROOT.is_dir(), reason="needs the sibling ilu-pose-estimation checkout")
def test_config_text_matches_the_scene_camera():
    scene = "v3_drone_0"
    meta = json.loads((SCENE_ROOT / scene / "meta.json").read_text())
    text = config_text(scene, SCENE_ROOT, "upright-tier", 2, "backends-example")

    position = meta["camera_locations"][0]
    rotation = meta["camera_rotations"][0]
    assert f"position_m: [{position[0]}, {position[1]}, {position[2]}]" in text
    assert f"rotation_euler_deg: [{rotation[0]}, {rotation[1]}, {rotation[2]}]" in text
    assert f"datasets/synthetic-data-generation/{scene}/rgb.jpg" in text
    assert "method: upright-tier" in text
    assert "min_corners: 2" in text


@pytest.mark.skipif(not SCENE_ROOT.is_dir(), reason="needs the sibling ilu-pose-estimation checkout")
def test_shipped_example_configs_load():
    from orchestrator.config import build_pipeline

    examples = sorted((REPO_ROOT / "configs" / "examples").glob("*.yaml"))
    assert examples
    for path in examples:
        pipeline = build_pipeline(path, REPO_ROOT)
        pose = next(s for s in pipeline.specs if s.name == "pose")
        assert pose.cfg["method"] == "upright-tier"
        assert pose.cfg["min_corners"] == 2
