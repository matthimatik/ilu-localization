from schemas.association import Association, Associations
from schemas.ilu import IdentificationMethod, ILURecord, ILURecords
from schemas.image import CameraIntrinsics, CameraModel, ImageRef
from schemas.run import RunManifest, StageResult, StageStatus
from schemas.vision import Embeddings, PoseBoxes, TextSpots
from tests.builders import make_embeddings, make_poseboxes, make_spot, make_textspots


def test_image_ref_roundtrip():
    ref = ImageRef(
        path="tests/fixtures/image.png",
        camera=CameraModel(
            intrinsics=CameraIntrinsics(
                fx=1.0, fy=1.0, cx=2.0, cy=3.0, width=10, height=20
            )
        ),
    )
    again = ImageRef.model_validate_json(ref.model_dump_json())
    assert again == ref
    assert again.camera.intrinsics.width == 10


def test_vision_roundtrip():
    spots = make_textspots([make_spot()])
    boxes = make_poseboxes([])
    embs = make_embeddings([("box-A", [0.1, 0.2])])
    assert TextSpots.model_validate_json(spots.model_dump_json()) == spots
    assert PoseBoxes.model_validate_json(boxes.model_dump_json()) == boxes
    assert Embeddings.model_validate_json(embs.model_dump_json()) == embs


def test_ilu_records_roundtrip():
    records = ILURecords(
        image_path="img.png",
        records=[
            ILURecord(
                ilu_id="box-A",
                category="container",
                iso_code="BMOU2372713",
                size_class="40",
                identification_method=IdentificationMethod.code,
                confidences={"pose": 0.9},
            )
        ],
    )
    again = ILURecords.model_validate_json(records.model_dump_json())
    assert again == records


def test_associations_roundtrip():
    assoc = Associations(
        associations=[
            Association(
                pose_box_id="box-A", text_spot_indices=[0], scores={"pose": 1.0}
            )
        ]
    )
    assert Associations.model_validate_json(assoc.model_dump_json()) == assoc


def test_run_manifest_roundtrip():
    manifest = RunManifest(
        run_id="r1",
        pipeline="p",
        created_at="2026-01-01T00:00:00+00:00",
        stages=[
            StageResult(stage="s", status=StageStatus.success, duration_s=0.1)
        ],
    )
    again = RunManifest.model_validate_json(manifest.model_dump_json())
    assert again == manifest
    assert again.stages[0].status == StageStatus.success
