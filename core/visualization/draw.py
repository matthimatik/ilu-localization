"""2D overlay of the fused ILU records onto the source image (OpenCV, BGR).

Drawing idioms follow ../ilu-pose-estimation/keypoints/evaluate_pose.py:
resolution-scaled strokes, outlined text, a black title bar.

The image area carries only geometric annotations: instance-tinted 8-corner
wireframes with keypoints, OCR quad outlines and small id badges. All record
metadata lives in the table strip appended below the image (and in the HTML
report), keyed by the badge id.
"""

from typing import NamedTuple

import cv2
import numpy as np

from core.association.api import visible_bbox
from schemas.association import Associations
from schemas.ilu import IdentificationMethod, ILURecord, ILURecords
from schemas.vision import TextSpot, TextSpots

# Box edges between corners (0-based), same order as the pose backend.
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]
# BGR. One color per record; cycles so neighbouring boxes stay distinguishable.
INSTANCE_COLORS = [
    (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0),
    (255, 255, 255), (128, 128, 255), (255, 128, 0), (0, 165, 255),
]
# BGR, dark enough for white text. Label fill keyed by identification method
# (only used by the verbose `--labels all` mode).
METHOD_COLORS = {
    IdentificationMethod.code: (50, 140, 0),
    IdentificationMethod.embedding_rank: (150, 90, 0),
    IdentificationMethod.unknown: (64, 64, 64),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
TABLE_BG = (32, 32, 32)
TABLE_HEADER_BG = (48, 48, 48)
TABLE_ROW_ALT_BG = (40, 40, 40)
# Tint alpha for the instance-colour fill of each box footprint.
INSTANCE_FILL_ALPHA = 0.35
SPOT_FILL_ALPHA = 0.30


def stroke(img: np.ndarray) -> tuple[int, float, int]:
    """Line width and font scale/thickness, scaled with resolution."""
    lw = max(2, round(max(img.shape[:2]) / 600))
    return lw, lw / 3, max(lw - 1, 1)


def put_text(
    im: np.ndarray,
    text: str,
    org: tuple[float, float],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Text with a black outline so it stays readable on any background."""
    org = (int(org[0]), int(org[1]))
    outline = thickness + max(2, thickness)
    cv2.putText(im, text, org, FONT, scale, (0, 0, 0), outline, cv2.LINE_AA)
    cv2.putText(im, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def put_title(im: np.ndarray, title: str, lw: int) -> np.ndarray:
    fs = lw / 3 * 1.5
    th = cv2.getTextSize(title, FONT, fs, lw)[0][1]
    width = im.shape[1]
    cv2.rectangle(im, (0, 0), (width, th + 4 * lw), (0, 0, 0), -1)
    cv2.putText(
        im, title, (2 * lw, th + 2 * lw), FONT, fs, (255, 255, 255), lw, cv2.LINE_AA
    )
    return im


def record_color(index: int) -> tuple[int, int, int]:
    return INSTANCE_COLORS[index % len(INSTANCE_COLORS)]


def _size_part(record: ILURecord) -> str:
    """` size` suffix, dropped when the size class just repeats the category."""
    size = record.size_class or "?"
    return "" if size.lower() == record.category.lower() else f" {size}"


def _label_lines(record: ILURecord) -> tuple[str, str]:
    iso = record.iso_code or "no code"
    first = f"{record.ilu_id}  {record.category}{_size_part(record)}  {iso}"
    confs = record.confidences or {}
    parts = [record.identification_method.value]
    for key in ("pose", "ocr", "text_match"):
        if key in confs:
            parts.append(f"{key} {confs[key]:.2f}")
    return first, " · ".join(parts)


class Label(NamedTuple):
    """One planned annotation: a full label block or a compact id badge."""

    record: ILURecord
    index: int
    text: str
    sub: str | None
    badge: bool  # instance-color chip with just the id (keeps busy scenes readable)


def short_id(ilu_id: str) -> str:
    """Badge form of an ilu id: the part after the last hyphen (`det-0` → `0`)."""
    return ilu_id.rsplit("-", 1)[-1] if "-" in ilu_id else ilu_id


def record_label_items(records: list[ILURecord], mode: str = "ids") -> list[Label]:
    """What to draw per record.

    - `ids` (default): small id badge only; details live in the table below
    - `all`: full two-line label blocks incl. confidences (debugging)
    - `none`: nothing (image stays pure geometry)
    """
    if mode == "none":
        return []
    shorts = [short_id(record.ilu_id) for record in records]
    unique = len(set(shorts)) == len(shorts)
    labels: list[Label] = []
    for index, record in enumerate(records):
        if mode == "all":
            first, second = _label_lines(record)
            labels.append(Label(record, index, first, second, badge=False))
        else:
            text = shorts[index] if unique else record.ilu_id
            labels.append(Label(record, index, text, None, badge=True))
    return labels


def _overlaps(
    rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]
) -> bool:
    x0, y0, x1, y1 = rect
    return any(
        x0 < px1 and px0 < x1 and y0 < py1 and py0 < y1
        for px0, py0, px1, py1 in placed
    )


def _place_label(
    x0: int,
    y0: int,
    box_w: int,
    box_h: int,
    img_w: int,
    img_h: int,
    lw: int,
    placed: list[tuple[int, int, int, int]],
    min_y: int = 0,
) -> tuple[int, int]:
    """Nearest collision-free spot to the box top (searched up/out), else the default."""
    base_x = min(max(x0, 0), max(img_w - box_w, 0))
    above = y0 - box_h - lw
    below = y0 + lw
    base_y = above if above >= lw else min(below, img_h - box_h)
    step_y = box_h + lw
    step_x = box_w // 2 + lw
    dy_order = [0]
    for step in range(1, 13):
        dy_order += [-step, step]
    candidates: list[tuple[float, int, int]] = []
    for dy in dy_order:
        for dx in (0, -1, 1, -2, 2, -3, 3, -4, 4):
            ly0 = min(max(base_y + dy * step_y, min_y), max(img_h - box_h, min_y))
            lx0 = min(max(base_x + dx * step_x, 0), max(img_w - box_w, 0))
            dist = abs(dy) * 2.0 + abs(dx) * 3.0 + (1.0 if dy > 0 else 0.0)
            candidates.append((dist, lx0, ly0))
    for _, lx0, ly0 in sorted(candidates):
        rect = (lx0, ly0, min(lx0 + box_w, img_w), ly0 + box_h)
        if not _overlaps(rect, placed):
            return lx0, ly0
    return base_x, min(max(base_y, min_y), max(img_h - box_h, min_y))


class Placement(NamedTuple):
    label: Label
    rect: tuple[int, int, int, int]  # label block, clamped to the image
    bbox: tuple[int, int, int, int]  # visible box corners (anchor for leaders)


def plan_labels(
    img_w: int,
    img_h: int,
    labels: list[Label],
    title: str,
    lw: int,
    fs: float,
    tf: int,
    obstacles: list[tuple[int, int, int, int]] | None = None,
) -> list[Placement]:
    """Collision-free label rects; public so tests can assert they never overlap.

    `obstacles` are rects labels must stay off (OCR quads keep their highlight).
    """
    title_fs = lw / 3 * 1.5
    min_y = cv2.getTextSize(title, FONT, title_fs, lw)[0][1] + 5 * lw
    th = cv2.getTextSize("Ag", FONT, fs, tf)[0][1]
    th2 = cv2.getTextSize("Ag", FONT, fs * 0.8, tf)[0][1]
    placed: list[tuple[int, int, int, int]] = list(obstacles or [])
    placements: list[Placement] = []
    for label in labels:
        bbox = (
            visible_bbox(label.record.pose)
            if label.record.pose is not None
            else None
        )
        if bbox is None:
            continue
        (tw, _), _ = cv2.getTextSize(label.text, FONT, fs, tf)
        (tw2, _), _ = cv2.getTextSize(label.sub or "", FONT, fs * 0.8, tf)
        box_w = max(tw, tw2) + (3 if label.badge else 5) * lw
        box_h = (th + 4 * lw) if label.sub is None else (th + th2 + 4 * lw)
        lx0, ly0 = _place_label(
            int(bbox[0]), int(bbox[1]), box_w, box_h, img_w, img_h, lw, placed, min_y
        )
        rect = (lx0, ly0, min(lx0 + box_w, img_w), ly0 + box_h)
        placed.append(rect)
        placements.append(
            Placement(label, rect, tuple(map(int, bbox)))  # type: ignore[arg-type]
        )
    return placements


def _visible_corners(pose, kpt_conf_thres: float = 0.0) -> dict[int, tuple]:
    return {
        k: c.position_px
        for k, c in enumerate(pose.corners_px)
        if c.position_px is not None and c.confidence >= kpt_conf_thres
    }


def _tint_fill(im: np.ndarray, pts: np.ndarray, color, alpha: float) -> None:
    """Blend `color` over the convex hull of `pts` (instance-colour footprint)."""
    if len(pts) < 3:
        return
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 1, 2)
    mask = np.zeros(im.shape[:2], np.uint8)
    cv2.fillPoly(mask, [hull], 255)
    region = mask > 0
    base = im[region].astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    im[region] = (base * (1.0 - alpha) + tint * alpha).astype(np.uint8)


def draw_instances(
    im: np.ndarray,
    records: list[ILURecord],
    kpt_conf_thres: float = 0.0,
) -> np.ndarray:
    """Instance-tinted footprint fill, skeleton wireframe and keypoint dots."""
    lw, _, _ = stroke(im)
    out = im.copy()
    for index, record in enumerate(records):
        pose = record.pose
        if pose is None:
            continue
        color = record_color(index)
        visible = _visible_corners(pose, kpt_conf_thres)
        _tint_fill(out, np.array(list(visible.values()), dtype=np.float32), color, INSTANCE_FILL_ALPHA)
        for a, b in SKELETON:
            if a in visible and b in visible:
                cv2.line(
                    out,
                    tuple(map(int, visible[a])),
                    tuple(map(int, visible[b])),
                    color,
                    lw,
                    cv2.LINE_AA,
                )
        for p in visible.values():
            cv2.circle(out, (int(p[0]), int(p[1])), lw * 3, color, -1, cv2.LINE_AA)
    return out


def draw_labels(
    im: np.ndarray,
    records: list[ILURecord],
    title: str,
    label_mode: str = "ids",
    obstacles: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Small id badges (or verbose label blocks), then the title bar."""
    lw, fs, tf = stroke(im)
    out = im.copy()
    img_h, img_w = out.shape[:2]
    th = cv2.getTextSize("Ag", FONT, fs, tf)[0][1]
    th2 = cv2.getTextSize("Ag", FONT, fs * 0.8, tf)[0][1]
    placements = plan_labels(
        img_w, img_h, record_label_items(records, label_mode), title, lw, fs, tf,
        obstacles=obstacles,
    )

    # leader line when a verbose label had to move away from its box
    for placement in placements:
        if placement.label.badge:
            continue
        rx0, ry0, rx1, ry1 = placement.rect
        x0, y0, x1, y1 = placement.bbox
        gap_x = max(0, x0 - rx1, rx0 - x1)
        gap_y = max(0, y0 - ry1, ry0 - y1)
        if gap_x <= 3 * lw and gap_y <= 3 * lw:
            continue
        if ry1 <= y0:
            start, end = ((rx0 + rx1) // 2, ry1), ((x0 + x1) // 2, y0)
        elif ry0 >= y1:
            start, end = ((rx0 + rx1) // 2, ry0), ((x0 + x1) // 2, y1)
        elif rx1 <= x0:
            start, end = (rx1, (ry0 + ry1) // 2), (x0, (y0 + y1) // 2)
        else:
            start, end = (rx0, (ry0 + ry1) // 2), (x1, (y0 + y1) // 2)
        cv2.line(
            out, start, end, record_color(placement.label.index),
            max(1, lw - 2), cv2.LINE_AA,
        )

    for placement in placements:
        label = placement.label
        lx0, ly0, rx1, ry1 = placement.rect
        instance = record_color(label.index)
        if label.badge:
            fill = instance
            text_color = (0, 0, 0)
        else:
            fill = METHOD_COLORS.get(
                label.record.identification_method, (64, 64, 64)
            )
            text_color = (255, 255, 255)
        cv2.rectangle(out, (lx0, ly0), (rx1, ry1), fill, -1)
        if not label.badge:
            # instance-color strip keeps a relocated label tied to its box
            cv2.rectangle(out, (lx0, ly0), (lx0 + max(lw, 1), ry1), instance, -1)
        tx = lx0 + (2 if label.badge else 3) * lw
        ty = ly0 + 2 * lw + th
        cv2.putText(
            out, label.text, (tx, ty), FONT, fs, (0, 0, 0), tf + 1, cv2.LINE_AA
        )
        cv2.putText(
            out, label.text, (tx, ty), FONT, fs, text_color, tf, cv2.LINE_AA
        )
        if label.sub is not None:
            ty += th2 + 2 * lw
            cv2.putText(
                out, label.sub, (tx, ty), FONT, fs * 0.8, (0, 0, 0), tf + 1, cv2.LINE_AA
            )
            cv2.putText(
                out, label.sub, (tx, ty), FONT, fs * 0.8, (230, 230, 230), tf, cv2.LINE_AA
            )
    return put_title(out, title, lw)


def draw_text_spots(
    im: np.ndarray,
    text_spots: TextSpots | None,
    color_by_index: dict[int, tuple[int, int, int]] | None = None,
    mode: str = "associated",
) -> np.ndarray:
    """Highlight quads around the spotted text; no text printed on the image
    (recognized text lives in the table and the scene hover).

    `mode`: `associated` (only spots linked to a record, default), `all`, `none`.
    """
    if text_spots is None or mode == "none":
        return im
    lw, _, _ = stroke(im)
    out = im.copy()
    color_by_index = color_by_index or {}
    grey = (140, 140, 140)
    for index, spot in enumerate(text_spots.spots):
        associated = index in color_by_index
        if mode == "associated" and not associated:
            continue
        color = color_by_index.get(index, grey)
        quad = np.array(spot.quad, dtype=np.float32)
        if associated:
            _tint_fill(out, quad, color, SPOT_FILL_ALPHA)
        thickness = lw if associated else max(1, lw - 1)
        cv2.polylines(
            out, [quad.astype(np.int32)], True, color, thickness, cv2.LINE_AA
        )
    return out


def spot_obstacles(
    text_spots: TextSpots | None,
    color_by_index: dict[int, tuple[int, int, int]] | None,
    mode: str,
    lw: int,
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, int, int]]:
    """Padded quad rects of the drawn OCR spots (labels must stay off them)."""
    if text_spots is None or mode == "none":
        return []
    color_by_index = color_by_index or {}
    rects: list[tuple[int, int, int, int]] = []
    for index, spot in enumerate(text_spots.spots):
        if mode == "associated" and index not in color_by_index:
            continue
        xs = [p[0] for p in spot.quad]
        ys = [p[1] for p in spot.quad]
        rects.append(
            (
                max(int(min(xs)) - 2 * lw, 0),
                max(int(min(ys)) - 2 * lw, 0),
                min(int(max(xs)) + 2 * lw, img_w),
                min(int(max(ys)) + 2 * lw, img_h),
            )
        )
    return rects


def _fit_cell(text: str, fs: float, tf: int, avail: int) -> str:
    """Truncate `text` with an ellipsis so it fits into `avail` pixels."""
    if avail <= 0:
        return ""
    if cv2.getTextSize(text, FONT, fs, tf)[0][0] <= avail:
        return text
    while text:
        text = text[:-1]
        candidate = text + "…"
        if cv2.getTextSize(candidate, FONT, fs, tf)[0][0] <= avail:
            return candidate
    return ""


TABLE_HEADERS = [
    "id", "category", "size", "ISO 6346", "ident", "box",
    "pose", "ocr", "text_match", "#spots", "OCR texts",
]
# Last column: spots wrap into multiple lines and soak up the leftover width.
OCR_COLUMN = len(TABLE_HEADERS) - 1


def _ocr_items(cell: str) -> list[str]:
    """One `text confidence` item per associated spot."""
    return [] if cell in ("—", "") else cell.split(", ")


def _wrap_items(
    items: list[str], fs: float, tf: int, avail: int
) -> list[str]:
    """Greedy line wrapping of `text confidence` items into `avail` pixels."""
    lines: list[str] = []
    current = ""
    for item in items:
        item = _fit_cell(item, fs, tf, avail)
        candidate = f"{current}, {item}" if current else item
        if current and cv2.getTextSize(candidate, FONT, fs, tf)[0][0] > avail:
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _spots_by_id(
    text_spots: TextSpots | None, associations: Associations | None
) -> dict[str, list[TextSpot]]:
    if text_spots is None or associations is None:
        return {}
    return {
        assoc.pose_box_id: [
            text_spots.spots[i]
            for i in assoc.text_spot_indices
            if 0 <= i < len(text_spots.spots)
        ]
        for assoc in associations.associations
    }


def record_table_cells(
    records: list[ILURecord],
    text_spots: TextSpots | None = None,
    associations: Associations | None = None,
) -> tuple[list[str], list[list[str]]]:
    """Table header + one detail row per record (same data as report.html)."""
    spot_by_id = _spots_by_id(text_spots, associations)
    has_assoc = associations is not None

    def conf_text(record: ILURecord, key: str) -> str:
        value = (record.confidences or {}).get(key)
        return "—" if value is None else f"{value:.2f}"

    rows: list[list[str]] = []
    for record in records:
        spots = spot_by_id.get(record.ilu_id, [])
        ocr = (
            ", ".join(f"{spot.text} {spot.confidence:.2f}" for spot in spots)
            if spots
            else "—"
        )
        rows.append(
            [
                record.ilu_id,
                record.category,
                record.size_class or "—",
                record.iso_code or "—",
                record.identification_method.value,
                record.pose.method if record.pose is not None else "—",
                conf_text(record, "pose"),
                conf_text(record, "ocr"),
                conf_text(record, "text_match"),
                str(len(spots)) if has_assoc else "—",
                ocr if has_assoc else "—",
            ]
        )
    return list(TABLE_HEADERS), rows


def build_record_table(
    width: int,
    records: list[ILURecord],
    text_spots: TextSpots | None = None,
    associations: Associations | None = None,
    lw: int = 2,
) -> np.ndarray:
    """Detail table strip appended below the overlay image.

    One header + one row per record, id column swatched in the instance color
    so rows tie back to the badges and wireframes in the image above. The OCR
    texts column gets all leftover width and wraps onto extra lines, so long
    spot lists stay readable instead of being ellipsized.
    """
    u = min(12 * lw, width / 70)
    margin = int(u * 1.2)
    swatch = int(u * 0.5)
    headers, rows = record_table_cells(records, text_spots, associations)
    note = (
        "id ↔ badge on the image · colour swatch = instance · "
        "OCR texts = associated spots (text confidence) · box = pose method (pnp/yolo)"
    )
    fs = u * 0.75 / 22
    tf = max(1, round(u / 10))

    def text_px(text: str, scale: float, thickness: int, bold: bool = False) -> int:
        return cv2.getTextSize(text, FONT, scale, thickness + (1 if bold else 0))[0][0]

    def fixed_widths(scale: float, thickness: int) -> list[int]:
        """Widths of every column except the OCR texts column."""
        widths = []
        for column, header in enumerate(headers):
            if column == OCR_COLUMN:
                continue
            width_px = text_px(header, scale, thickness, bold=True)
            for row in rows:
                pad = swatch + int(u * 0.3) if column == 0 else 0
                width_px = max(width_px, text_px(row[column], scale, thickness) + pad)
            widths.append(width_px + int(u * 0.8))
        return widths

    def min_ocr_width(scale: float, thickness: int) -> int:
        widest = text_px(headers[OCR_COLUMN], scale, thickness, bold=True)
        for row in rows:
            for item in _ocr_items(row[OCR_COLUMN]):
                widest = max(widest, text_px(item, scale, thickness))
        return widest + int(u * 0.8)

    for _ in range(5):
        if sum(fixed_widths(fs, tf)) + min_ocr_width(fs, tf) + 2 * margin <= width:
            break
        fs *= 0.9
    fixed = fixed_widths(fs, tf)
    ocr_w = max(min_ocr_width(fs, tf), width - 2 * margin - sum(fixed))
    col_x = [margin]
    for w in fixed:
        col_x.append(col_x[-1] + w)

    text_h = cv2.getTextSize("Ag", FONT, fs, tf)[0][1]
    line_step = text_h + int(u * 0.55)
    ocr_avail = int(ocr_w - u * 0.4)
    row_lines = [
        _wrap_items(_ocr_items(row[OCR_COLUMN]), fs, tf, ocr_avail) or ["—"]
        for row in rows
    ]
    row_heights = [
        max(int(u * 1.25), len(lines) * line_step + int(u * 0.55))
        for lines in row_lines
    ]
    base_row_h = int(u * 1.25)
    header_h = base_row_h + int(u * 0.2)
    note_h = int(u * 1.1) if note else 0
    height = int(header_h + sum(row_heights) + u * 0.4 + note_h)
    im = np.full((height, width, 3), TABLE_BG, np.uint8)
    cv2.rectangle(im, (0, 0), (width, int(header_h)), TABLE_HEADER_BG, -1)

    header_baseline = int((header_h + text_h) / 2)
    for column, header in enumerate(headers):
        x = col_x[column] if column != OCR_COLUMN else col_x[-1]
        put_text(im, header, (x, header_baseline), fs, (255, 255, 255), tf + 1)

    y = header_h
    for r, row in enumerate(rows):
        row_h = row_heights[r]
        y0 = int(y)
        if r % 2 == 1:
            cv2.rectangle(im, (0, y0), (width, y0 + int(row_h)), TABLE_ROW_ALT_BG, -1)
        lines = row_lines[r]
        block_h = len(lines) * line_step
        first_baseline = y0 + max(int(u * 0.35), (row_h - block_h) // 2) + text_h
        cv2.circle(
            im,
            (col_x[0] + swatch // 2, first_baseline - text_h // 3),
            swatch // 2,
            record_color(r),
            -1,
            cv2.LINE_AA,
        )
        for column, cell in enumerate(row):
            if column == OCR_COLUMN:
                continue
            x = col_x[column] + (swatch + int(u * 0.3) if column == 0 else 0)
            put_text(im, cell, (x, first_baseline), fs, (230, 230, 230), tf)
        for i, line in enumerate(lines):
            put_text(
                im, line, (col_x[-1], first_baseline + i * line_step),
                fs, (230, 230, 230), tf,
            )
        y += row_h

    if note:
        note_y = int(y + u * 0.9)
        put_text(im, note, (margin, note_y), fs * 0.9, (200, 200, 200), tf)
    return im


def _summary(records: list[ILURecord]) -> str:
    coded = sum(1 for r in records if r.identification_method == IdentificationMethod.code)
    embedded = sum(
        1 for r in records if r.identification_method == IdentificationMethod.embedding_rank
    )
    unknown = len(records) - coded - embedded
    return (
        f"{len(records)} records · {coded} code · {embedded} embedding · "
        f"{unknown} unidentified"
    )


def build_overlay(
    image: np.ndarray,
    records: ILURecords,
    text_spots: TextSpots | None = None,
    associations: Associations | None = None,
    title: str | None = None,
    label_mode: str = "ids",
    spot_mode: str = "associated",
) -> np.ndarray:
    """End-result overlay: annotated image + detail table strip below.

    The image area stays annotation-only (instances, keypoints, OCR quads, id
    badges); every metadata field is in the table keyed by badge id.
    """
    if title is None:
        title = f"ILU localization · {records.image_path} · {_summary(records.records)}"
    color_by_index: dict[int, tuple[int, int, int]] = {}
    if associations is not None:
        index_by_id = {r.ilu_id: i for i, r in enumerate(records.records)}
        for assoc in associations.associations:
            match = index_by_id.get(assoc.pose_box_id)
            if match is None:
                continue
            color = record_color(match)
            for spot_index in assoc.text_spot_indices:
                color_by_index.setdefault(spot_index, color)

    lw, _, _ = stroke(image)
    img_h, img_w = image.shape[:2]
    obstacles = spot_obstacles(
        text_spots, color_by_index, spot_mode, lw, img_w, img_h
    )
    out = draw_instances(image, records.records)
    out = draw_text_spots(out, text_spots, color_by_index, mode=spot_mode)
    out = draw_labels(
        out, records.records, title, label_mode=label_mode, obstacles=obstacles
    )
    table = build_record_table(
        out.shape[1], records.records, text_spots, associations, lw
    )
    return np.vstack([out, table])
