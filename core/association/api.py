from schemas.association import Association, Associations
from schemas.vision import Embeddings, Point2D, PoseBox, PoseBoxes, TextSpots


def quad_center(quad: list[Point2D]) -> Point2D:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def visible_bbox(
    box: PoseBox, margin_px: float = 0.0
) -> tuple[float, float, float, float] | None:
    pts = [c.position_px for c in box.corners_px if c.position_px is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (
        min(xs) - margin_px,
        min(ys) - margin_px,
        max(xs) + margin_px,
        max(ys) + margin_px,
    )


def bbox_contains(
    bbox: tuple[float, float, float, float], point: Point2D
) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def associate(
    text_spots: TextSpots,
    pose_boxes: PoseBoxes,
    embeddings: Embeddings | None = None,
    margin_px: float = 0.0,
) -> Associations:
    embeddings_by_crop: dict[str, str] = {}
    if embeddings is not None:
        for emb in embeddings.embeddings:
            embeddings_by_crop[emb.crop_id] = emb.crop_id

    associations: list[Association] = []
    for box in pose_boxes.boxes:
        bbox = visible_bbox(box, margin_px)
        matched: list[tuple[int, float]] = []
        if bbox is not None:
            for index, spot in enumerate(text_spots.spots):
                if bbox_contains(bbox, quad_center(spot.quad)):
                    matched.append((index, spot.confidence))
        matched.sort(key=lambda item: -item[1])
        crop_id = embeddings_by_crop.get(box.box_id)
        associations.append(
            Association(
                pose_box_id=box.box_id,
                text_spot_indices=[index for index, _ in matched],
                embedding_crop_id=crop_id,
                scores={
                    "text_match": matched[0][1] if matched else 0.0,
                    "pose": box.confidence,
                },
            )
        )
    return Associations(associations=associations)
