from schemas.vision import Corner, Embedding, Embeddings, PoseBox, PoseBoxes, TextSpot, TextSpots


def make_spot(
    text: str = "BMOU2372713",
    text_type: str = "id_iso6346",
    confidence: float = 0.9,
    center: tuple[float, float] = (100.0, 100.0),
    size: float = 40.0,
) -> TextSpot:
    x, y = center
    s = size / 2
    return TextSpot(
        quad=[
            (x - s, y - s),
            (x + s, y - s),
            (x + s, y + s),
            (x - s, y + s),
        ],
        text=text,
        text_type=text_type,
        confidence=confidence,
    )


def make_box(
    box_id: str = "box-A",
    x0: float = 0.0,
    y0: float = 0.0,
    x1: float = 200.0,
    y1: float = 200.0,
    size_class: str = "40",
    confidence: float = 0.9,
    occlude_last: bool = False,
) -> PoseBox:
    corners = [
        (x0, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
        (x0 + 1, y0 + 1),
        (x1 - 1, y0 + 1),
        (x1 - 1, y1 - 1),
        (x0 + 1, y1 - 1),
    ]
    built = [
        Corner(position_px=None if occlude_last and i == 7 else c, confidence=confidence)
        for i, c in enumerate(corners)
    ]
    return PoseBox(
        box_id=box_id, corners_px=built, size_class=size_class, confidence=confidence
    )


def make_textspots(spots: list[TextSpot], image_path: str = "img.png") -> TextSpots:
    return TextSpots(image_path=image_path, spots=spots)


def make_poseboxes(boxes: list[PoseBox], image_path: str = "img.png") -> PoseBoxes:
    return PoseBoxes(image_path=image_path, boxes=boxes)


def make_embeddings(pairs: list[tuple[str, list[float]]]) -> Embeddings:
    return Embeddings(
        embeddings=[Embedding(crop_id=cid, vector=vec) for cid, vec in pairs]
    )
