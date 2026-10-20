import json
from pathlib import Path

import pytest

from orchestrator.stage import StageContext
from orchestrator.stages.identify import IdentifyStage, ilu_type_for, items_for, results_from_runner
from orchestrator.store import ArtifactStore
from schemas.association import Association, Associations
from schemas.identification import IdentifiedILUs
from schemas.vision import PoseBoxes, TextSpots
from tests.builders import make_box, make_poseboxes, make_spot, make_textspots

REPO_ROOT = Path(__file__).resolve().parents[1]
IDENT_BACKEND = REPO_ROOT / ".." / "ilu-identification"

CANNED_RUNNER = {
    "results": [
        {
            "ilu_id": "box-A",
            "ilu_type": "container",
            "result": {
                "schema_version": "1",
                "image_path": "img.png",
                "id_code": "BMOU2372713",
                "id_scheme": "iso6346",
                "id_valid": True,
                "id_confidence": 1.0,
                "method": "code",
                "candidates": [],
                "size_type": {
                    "code": "45G1",
                    "length_m": 12.192,
                    "height_m": 2.896,
                    "body_kind": "dry",
                    "high_cube": True,
                    "in_whitelist": True,
                },
                "embedding_match": None,
                "warnings": [],
            },
        },
        {
            "ilu_id": "box-B",
            "ilu_type": "trailer",
            "result": {
                "schema_version": "1",
                "image_path": "img.png",
                "id_code": None,
                "id_scheme": "none",
                "id_valid": False,
                "id_confidence": 0.0,
                "method": "unknown",
                "candidates": [],
                "size_type": None,
                "embedding_match": None,
                "warnings": [],
            },
        },
    ]
}


class TestIluTypeFor:
    def test_trailer_box(self):
        box = make_box(size_class="trailer")
        assert ilu_type_for(box, [make_spot(text="anything")]) == "trailer"

    def test_tank_via_text(self):
        box = make_box(size_class="40")
        spots = [make_spot(text="TANK UNIT", text_type="marking")]
        assert ilu_type_for(box, spots) == "tank_container"

    def test_default_container(self):
        box = make_box(size_class="40")
        assert ilu_type_for(box, [make_spot(text="BMOU2372713")]) == "container"


class TestItemsFor:
    def test_builds_payloads_and_placeholders(self):
        spots = make_textspots(
            [make_spot(text="BMOU2372713"), make_spot(text="45G1", center=(200, 100))]
        )
        boxes = make_poseboxes(
            [make_box(box_id="box-A"), make_box(box_id="box-B", size_class="trailer")]
        )
        assoc = Associations(
            associations=[
                Association(pose_box_id="box-A", text_spot_indices=[0, 1]),
                Association(pose_box_id="box-B", text_spot_indices=[]),
            ]
        )
        items, placeholders = items_for(assoc, spots, boxes)
        assert len(items) == 1
        assert items[0]["ilu_id"] == "box-A"
        assert items[0]["ilu_type"] == "container"
        assert len(items[0]["spots"]) == 2
        assert items[0]["spots"][0]["score"] == pytest.approx(0.9)
        assert len(placeholders) == 1
        assert placeholders[0].ilu_id == "box-B"
        assert placeholders[0].ilu_type == "trailer"
        assert placeholders[0].method == "unknown"

    def test_out_of_range_indices_ignored(self):
        spots = make_textspots([make_spot(text="BMOU2372713")])
        boxes = make_poseboxes([make_box(box_id="box-A")])
        assoc = Associations(
            associations=[Association(pose_box_id="box-A", text_spot_indices=[0, 99])]
        )
        items, _ = items_for(assoc, spots, boxes)
        assert len(items[0]["spots"]) == 1


class TestResultsFromRunner:
    def test_translation(self):
        artifact = results_from_runner(CANNED_RUNNER, "img.png")
        assert isinstance(artifact, IdentifiedILUs)
        assert artifact.image_path == "img.png"
        a, b = artifact.results
        assert a.ilu_id == "box-A"
        assert a.ilu_type == "container"
        assert a.id_code == "BMOU2372713"
        assert a.id_scheme == "iso6346"
        assert a.id_valid
        assert a.method == "code"
        assert a.size_type is not None
        assert a.size_type.code == "45G1"
        assert a.size_type.length_m == pytest.approx(12.192)
        assert a.size_type.high_cube
        assert b.ilu_id == "box-B"
        assert b.id_code is None
        assert b.size_type is None


@pytest.mark.skipif(
    not IDENT_BACKEND.is_dir(), reason="needs sibling ilu-identification checkout"
)
def test_identify_stage_end_to_end(tmp_path):
    store = ArtifactStore(tmp_path / "runs", "identify-e2e")
    spots = make_textspots(
        [
            make_spot(text="BMOU 237271 3", confidence=0.94),
            make_spot(text="45G1", center=(300, 100), confidence=0.87),
        ],
        image_path="img.png",
    )
    boxes = make_poseboxes([make_box(box_id="box-A", size_class="40")], image_path="img.png")
    assoc = Associations(
        associations=[Association(pose_box_id="box-A", text_spot_indices=[0, 1])]
    )
    store.write_model("assoc", "associations", assoc)
    store.write_model("spots", "text_spots", spots)
    store.write_model("boxes", "pose_boxes", boxes)

    ctx = StageContext(
        store=store,
        stage_name="identify",
        cfg={
            "backend_dir": str(IDENT_BACKEND.resolve()),
            "_base_dir": str(REPO_ROOT),
        },
        inputs={
            "associations": store.path_for("assoc", "associations"),
            "text_spots": store.path_for("spots", "text_spots"),
            "pose_boxes": store.path_for("boxes", "pose_boxes"),
        },
    )
    IdentifyStage().run(ctx)
    artifact = store.read_model("identify", "ilu_ids", IdentifiedILUs)
    assert len(artifact.results) == 1
    entry = artifact.results[0]
    assert entry.id_code == "BMOU2372713"
    assert entry.id_scheme == "iso6346"
    assert entry.id_valid
    assert entry.size_type is not None
    assert entry.size_type.code == "45G1"
