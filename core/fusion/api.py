from core.fusion.iso6346 import validate_iso6346
from schemas.association import Associations
from schemas.identification import IdentifiedILUs
from schemas.ilu import IdentificationMethod, ILURecord, ILURecords
from schemas.vision import Embeddings, PoseBox, PoseBoxes, TextSpot, TextSpots


def category_for(box: PoseBox, spots: list[TextSpot]) -> str:
    if box.size_class == "trailer":
        return "trailer"
    for spot in spots:
        haystack = f"{spot.text} {spot.text_type}".lower()
        if "tank" in haystack:
            return "tank_container"
    return "container"


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


def fuse(
    associations: Associations,
    text_spots: TextSpots,
    pose_boxes: PoseBoxes,
    embeddings: Embeddings,
    image_path: str = "",
    identified: IdentifiedILUs | None = None,
) -> ILURecords:
    boxes_by_id = {box.box_id: box for box in pose_boxes.boxes}
    embeddings_by_crop = {e.crop_id: e for e in embeddings.embeddings}
    ids_by_id = (
        {entry.ilu_id: entry for entry in identified.results} if identified else {}
    )

    records: list[ILURecord] = []
    for assoc in associations.associations:
        box = boxes_by_id[assoc.pose_box_id]
        spots = [
            text_spots.spots[i]
            for i in assoc.text_spot_indices
            if 0 <= i < len(text_spots.spots)
        ]
        entry = ids_by_id.get(assoc.pose_box_id)
        embedding = (
            embeddings_by_crop.get(assoc.embedding_crop_id)
            if assoc.embedding_crop_id
            else None
        )

        if entry is not None and entry.id_valid and entry.id_code:
            iso_code = entry.id_code
            ocr_confidence = entry.id_confidence
            method = IdentificationMethod(entry.method)
            category = entry.ilu_type
            id_scheme = entry.id_scheme
            size_type = entry.size_type.code if entry.size_type else None
        else:
            iso_code, ocr_confidence = _pick_iso_code(
                text_spots, assoc.text_spot_indices
            )
            if iso_code is not None:
                method = IdentificationMethod.code
            elif embedding is not None:
                method = IdentificationMethod.embedding_rank
            else:
                method = IdentificationMethod.unknown
            category = category_for(box, spots)
            id_scheme = "iso6346" if iso_code is not None else None
            size_type = None

        records.append(
            ILURecord(
                ilu_id=box.box_id,
                category=category,
                iso_code=iso_code,
                id_scheme=id_scheme,
                size_class=box.size_class,
                size_type=size_type,
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
