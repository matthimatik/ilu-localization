import json
import os
from pathlib import Path

import pytest

from orchestrator.stage import StageContext
from orchestrator.stages.mmocr_backend import MmocrBackendStage, text_spots_from_runner
from orchestrator.store import ArtifactStore
from schemas.image import ImageRef
from schemas.vision import TextSpots

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
    "image": "img.png",
    "spots": [
        {
            "quad": [[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]],
            "text": "BMOU2372713",
            "det_score": 0.9,
            "rec_score": 0.95,
            "score": 0.855,
        },
        {
            "quad": [[0.0, 50.0], [60.0, 50.0], [60.0, 70.0], [0.0, 70.0]],
            "text": "COSCO",
            "det_score": 0.8,
            "rec_score": 0.9,
            "score": 0.72,
        },
    ],
}


def test_runner_output_translation():
    result = text_spots_from_runner(CANNED_RUNNER, "img.png")
    assert len(result.spots) == 2
    first, second = result.spots
    assert first.text == "BMOU2372713"
    assert first.text_type == "id_iso6346"
    assert first.confidence == pytest.approx(0.855)
    assert first.quad[0] == (0.0, 0.0)
    assert second.text_type == "unknown"


@pytest.mark.skipif(
    os.environ.get("RUN_BACKEND_TESTS") != "1",
    reason="needs GPU + mmocr backend (set RUN_BACKEND_TESTS=1)",
)
def test_mmocr_backend_on_drone_scene(tmp_path):
    store = ArtifactStore(tmp_path / "runs", "mmocr-test")
    store.write_model("input", "image_ref", ImageRef(path=str(DRONE_IMAGE)))
    ctx = StageContext(
        store=store,
        stage_name="textspots",
        cfg={
            "backend_dir": str(REPO_ROOT / ".." / "mmocr"),
            "_base_dir": str(REPO_ROOT),
        },
        inputs={"image": store.path_for("input", "image_ref")},
    )
    MmocrBackendStage().run(ctx)
    spots = store.read_model("textspots", "text_spots", TextSpots)
    assert len(spots.spots) > 0
    assert all(len(s.quad) == 4 for s in spots.spots)
    assert all(0.0 <= s.confidence <= 1.0 for s in spots.spots)
    assert any(s.text for s in spots.spots)
