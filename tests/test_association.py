from core.association import associate
from tests.builders import (
    make_box,
    make_embeddings,
    make_poseboxes,
    make_spot,
    make_textspots,
)


def test_spot_inside_box_is_matched():
    spots = make_textspots([make_spot(center=(100.0, 100.0))])
    boxes = make_poseboxes([make_box(x0=0.0, y0=0.0, x1=200.0, y1=200.0)])
    result = associate(spots, boxes)
    assert len(result.associations) == 1
    assert result.associations[0].pose_box_id == "box-A"
    assert result.associations[0].text_spot_indices == [0]


def test_spot_outside_all_boxes_is_unmatched():
    spots = make_textspots([make_spot(center=(500.0, 500.0))])
    boxes = make_poseboxes([make_box(x0=0.0, y0=0.0, x1=200.0, y1=200.0)])
    result = associate(spots, boxes)
    assert result.associations[0].text_spot_indices == []
    assert result.associations[0].scores["text_match"] == 0.0


def test_multiple_spots_sorted_by_confidence():
    low = make_spot(text="AAAA1111111", confidence=0.4, center=(50.0, 50.0))
    high = make_spot(text="BMOU2372713", confidence=0.9, center=(150.0, 150.0))
    boxes = make_poseboxes([make_box(x0=0.0, y0=0.0, x1=200.0, y1=200.0)])
    result = associate(make_textspots([low, high]), boxes)
    assert result.associations[0].text_spot_indices == [1, 0]


def test_margin_expands_match_region():
    spot = make_spot(center=(210.0, 100.0))
    boxes = make_poseboxes([make_box(x0=0.0, y0=0.0, x1=200.0, y1=200.0)])
    without = associate(make_textspots([spot]), boxes, margin_px=0.0)
    with_margin = associate(make_textspots([spot]), boxes, margin_px=20.0)
    assert without.associations[0].text_spot_indices == []
    assert with_margin.associations[0].text_spot_indices == [0]


def test_embedding_attached_by_crop_id():
    boxes = make_poseboxes([make_box(box_id="box-A"), make_box(box_id="box-B")])
    embs = make_embeddings([("box-B", [0.1, 0.2, 0.3])])
    result = associate(make_textspots([]), boxes, embs)
    by_id = {a.pose_box_id: a for a in result.associations}
    assert by_id["box-A"].embedding_crop_id is None
    assert by_id["box-B"].embedding_crop_id == "box-B"


def test_single_occluded_corner_still_matches():
    boxes = make_poseboxes([make_box(occlude_last=True)])
    spot = make_spot(center=(100.0, 100.0))
    result = associate(make_textspots([spot]), boxes)
    assert result.associations[0].text_spot_indices == [0]
