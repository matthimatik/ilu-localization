from core.fusion.iso6346 import validate_iso6346
from schemas.association import Associations
from schemas.ilu import IdentificationMethod, ILURecord, ILURecords
from schemas.vision import Embeddings, PoseBox, PoseBoxes, TextSpot, TextSpots


def _pick_iso_code(
    spots: TextSpots, indices: list[int]
) -> tuple[str | None, float]:
    candidates: list[tuple[float, str]] = []
    for index in indices:
        spot = spots.spots[index]
        ok, _ = validate_iso6346(spot.text)
        if ok:
            candidates.append((spot.confidence, normalize_code(spot.text)))
    if not candidates:
        return None, 0.0
    confidence, code = max(candidates)
    return code, confidence


def normalize_code(text: str) -> str:
    return text.replace(" ", "").upper()


def _category(box: PoseBox, spots: TextSpots, indices: list[int]) -> str:
    if box.size_class == "trailer":
        return "trailer"
    for index in indices:
        spot: TextSpot = spots.spots[index]
        haystack = f"{spot.text} {spot.text_type}".lower()
        if "tank" in haystack:
            return "tank_container"
    return "container"


def fuse(
    associations: Associations,
    text_spots: TextSpots,
    pose_boxes: PoseBoxes,
    embeddings: Embeddings,
    image_path: str = "",
) -> ILURecords:
    boxes_by_id = {box.box_id: box for box in pose_boxes.boxes}
    embeddings_by_crop = {e.crop_id: e for e in embeddings.embeddings}

    records: list[ILURecord] = []
    for assoc in associations.associations:
        box = boxes_by_id[assoc.pose_box_id]
        iso_code, ocr_confidence = _pick_iso_code(
            text_spots, assoc.text_spot_indices
        )
        embedding = (
            embeddings_by_crop.get(assoc.embedding_crop_id)
            if assoc.embedding_crop_id
            else None
        )
        if iso_code is not None:
            method = IdentificationMethod.code
        elif embedding is not None:
            method = IdentificationMethod.embedding_rank
        else:
            method = IdentificationMethod.unknown
        records.append(
            ILURecord(
                ilu_id=box.box_id,
                category=_category(box, text_spots, assoc.text_spot_indices),
                iso_code=iso_code,
                size_class=box.size_class,
                identification_method=method,
                pose=box,
                embedding_crop_id=embedding.crop_id if embedding else None,
                confidences={
                    "pose": box.confidence,
                    "ocr": ocr_confidence,
                    "text_match": assoc.scores.get("text_match", 0.0),
                },
            )
        )
    return ILURecords(image_path=image_path, records=records)
