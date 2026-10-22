import json
import re
from pathlib import Path

from schemas.evaluation import GTBox, GTBoxes

ISO_CODE_RE = re.compile(r"^[A-Z]{4}\d{7}$")  # owner + serial + check digit
ISO_HEAD_RE = re.compile(r"^[A-Z]{4}\d{6}$")  # owner + serial, no check digit


def parse_iso_from_label(label: str) -> str | None:
    """Extract an ISO 6346 code from a pose.jsonl label like `'KLJU237271 3'`.

    The generator puts the owner+serial code and the check digit in separate
    whitespace-separated tokens; the full 11-character code is returned.
    """
    tokens = label.split()
    if not tokens:
        return None
    code = tokens[0].replace(" ", "").upper()
    if len(code) == 10 and len(tokens) > 1 and tokens[1].isdigit():
        code += tokens[1]
    if ISO_CODE_RE.match(code):
        return code
    if ISO_HEAD_RE.match(code):
        return code
    return None


def load_pose_jsonl(path: Path | str) -> GTBoxes:
    """Parse a synthetic scene's `pose.jsonl` into ground-truth boxes.

    Each line has `corners` (8 entries with `world_position` `[idx, [x,y,z]]`
    and `image_position` `[u,v]` or null when occluded), `model` (size class)
    and `label` (ISO code + check digit).
    """
    path = Path(path)
    boxes: list[GTBox] = []
    for index, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        corners = row.get("corners") or []
        world_corners: list[tuple[float, float, float]] = []
        visible: list[bool] = []
        for corner in corners:
            _, position = corner["world_position"]
            world_corners.append(
                (float(position[0]), float(position[1]), float(position[2]))
            )
            visible.append(corner.get("image_position") is not None)
        label = row.get("label", "")
        boxes.append(
            GTBox(
                index=index,
                world_corners=world_corners,
                visible=visible,
                size_class=str(row["model"]) if row.get("model") is not None else None,
                label=label,
                iso_code=parse_iso_from_label(label),
            )
        )
    return GTBoxes(source=str(path), boxes=boxes)
