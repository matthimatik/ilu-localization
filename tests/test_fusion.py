import pytest

from core.fusion import fuse, validate_iso6346
from core.fusion.iso6346 import calculate_check_digit
from schemas.association import Association, Associations
from tests.builders import (
    make_box,
    make_embeddings,
    make_poseboxes,
    make_spot,
    make_textspots,
)


@pytest.mark.parametrize(
    "code",
    ["BMOU2372713", "LKWA9118436", "XINU1447400", "DBOU2192060", "DFSU1239631"],
)
def test_known_valid_codes(code):
    ok, reason = validate_iso6346(code)
    assert ok, reason


@pytest.mark.parametrize(
    "code",
    ["BMOU2372710", "BMOU2372711", "SHORT", "BMOU237271X", "1234U1234567"],
)
def test_invalid_codes(code):
    ok, _ = validate_iso6346(code)
    assert not ok


def test_check_digit_matches_known_values():
    assert calculate_check_digit("BMOU237271") == 3
    assert calculate_check_digit("XINU144740") == 0


def _fuse_with(spots, boxes, embeddings=None):
    indices = list(range(len(spots.spots)))
    assoc = Associations(
        associations=[
            Association(
                pose_box_id=boxes.boxes[0].box_id,
                text_spot_indices=indices,
                embedding_crop_id=(
                    embeddings.embeddings[0].crop_id if embeddings and embeddings.embeddings else None
                ),
                scores={"text_match": 0.9},
            )
        ]
    )
    return fuse(
        assoc,
        spots,
        boxes,
        embeddings or make_embeddings([]),
        image_path="img.png",
    )


def test_fuse_identifies_by_valid_code():
    spots = make_textspots([make_spot(text="BMOU2372713")])
    boxes = make_poseboxes([make_box(size_class="40")])
    records = _fuse_with(spots, boxes).records
    assert len(records) == 1
    assert records[0].iso_code == "BMOU2372713"
    assert records[0].identification_method.value == "code"
    assert records[0].category == "container"
    assert records[0].confidences["ocr"] == pytest.approx(0.9)


def test_fuse_rejects_bad_check_digit():
    spots = make_textspots([make_spot(text="BMOU2372710")])
    boxes = make_poseboxes([make_box()])
    records = _fuse_with(spots, boxes).records
    assert records[0].iso_code is None
    assert records[0].identification_method.value == "unknown"


def test_fuse_falls_back_to_embedding():
    spots = make_textspots([])
    boxes = make_poseboxes([make_box(box_id="box-B", size_class="trailer")])
    embs = make_embeddings([("box-B", [0.1, 0.2])])
    records = _fuse_with(spots, boxes, embs).records
    assert records[0].identification_method.value == "embedding_rank"
    assert records[0].embedding_crop_id == "box-B"
    assert records[0].category == "trailer"


def test_fuse_detects_tank_container_via_text():
    spots = make_textspots(
        [make_spot(text="TANK UNIT", text_type="marking", center=(100.0, 100.0))]
    )
    boxes = make_poseboxes([make_box(size_class="40")])
    records = _fuse_with(spots, boxes).records
    assert records[0].category == "tank_container"


def test_fuse_prefers_highest_confidence_valid_code():
    low = make_spot(text="XINU1447400", confidence=0.5, center=(50.0, 50.0))
    high = make_spot(text="BMOU2372713", confidence=0.95, center=(150.0, 150.0))
    boxes = make_poseboxes([make_box()])
    records = _fuse_with(make_textspots([low, high]), boxes).records
    assert records[0].iso_code == "BMOU2372713"
